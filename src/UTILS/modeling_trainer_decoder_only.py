"""
modeling_trainerdecoder_only.py — Distributed Trainer for PreLlama summarization.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter

from bert_score import BERTScorer
from rouge_score import rouge_scorer

from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.configuration_utils import PreTrainedConfig



class Trainer:
    def __init__(
        self,
        config:             PreTrainedConfig,
        tokenizer:          PreTrainedTokenizerFast,
        model:              torch.nn.Module,
        train_data:         DataLoader,
        eval_data:          DataLoader,
        optimizer:          torch.optim.Optimizer,
        gpu_id:             int,
        scheduler:          Optional[LRScheduler] = None,
        checkpoint_path:    Optional[str] = None,
        writer:             Optional[SummaryWriter] = None,
        max_patience:       int = 2,
        evaluation_timing:  int = 500,
    ) -> None:

        self.config           = config
        self.gpu_id           = gpu_id
        self.tokenizer        = tokenizer
        self.train_data       = train_data
        self.eval_data        = eval_data
        self.optimizer        = optimizer
        self.scheduler        = scheduler
        self.checkpoint_path  = checkpoint_path
        self.writer           = writer
        self.max_patience     = max_patience
        self.evaluation_timing= evaluation_timing

        self.best_loss  = float("inf")
        self.counter    = 0
        self.list_loss  = []
        self.dico_len_data = {
            "train": len(self.train_data),
            "eval":  len(self.eval_data),
        }

        # Special tokens to exclude when computing generated length
        self.exclude_ids = torch.tensor([config.pad_token_id], device=gpu_id)

        # Generation kwargs (set in train_prellama.py via task_specific_params)
        # Model is already wrapped in DDP before being passed in
        self.model = model
        self.generate_config_dict = (
            self.model.module.config.task_specific_params["summarization"]
            if isinstance(self.model, DDP)
            else self.model.config.task_specific_params["summarization"]
        )

        # ── Metrics ────────────────────────────────────────────────────────────
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

        local_bert = (
            str(Path(os.environ["DSDIR"]) / "HuggingFace_Models/bert-base-uncased")
            if "DSDIR" in os.environ
            else "bert-base-uncased"
        )
        self.bert_scorer = BERTScorer(
            model_type=local_bert,
            num_layers=12,
            device=f"cuda:{gpu_id}",
            rescale_with_baseline=False,
        )

    # ── Internals ──────────────────────────────────────────────────────────────

    @property
    def is_master(self) -> bool:
        return self.gpu_id == 0

    def _log_scalar(self, tag: str, value: float, step: int) -> None:
        """Write to TensorBoard only on rank 0."""
        if self.is_master and self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def _reduce_tensor(self, t: torch.Tensor, op=dist.ReduceOp.AVG) -> torch.Tensor:
        """In-place all_reduce — no-op if not distributed."""
        if dist.is_initialized():
            dist.all_reduce(t, op=op)
        return t

    # ── Generation & scoring ──────────────────────────────────────────────────

    def _run_generation(self, batch: dict, epoch: int, step: int, mode: str) -> None:
        """Generate summaries and compute ROUGE + BERTScore, log results."""
        references = batch["references"]
        model_ref  = self.model.module if isinstance(self.model, DDP) else self.model

        self.model.eval()
        with torch.no_grad():
            generated_ids = model_ref.generate(
                input_ids      = batch["input_ids"][:, :batch["max_input_len"]],
                attention_mask = batch["attention_mask"][:, :batch["max_input_len"]],
                max_input_len  = batch["max_input_len"],
                target_len     = batch["target_len"],
                pad_token_id   = self.config.pad_token_id,
                eos_token_id   = self.config.eos_token_id,
                **self.generate_config_dict,
            )

        # ── Length MAE ────────────────────────────────────────────────────────
        gen_tokens = generated_ids[:, batch["max_input_len"]:]
        valid_mask = ~torch.isin(gen_tokens, self.exclude_ids)
        # subtract 1 for BOS at start (already included in gen_tokens)
        generate_len = valid_mask.sum(dim=1).sub(1).clamp(min=0)
        MAE_score = torch.abs(
            batch["target_len"].float() - generate_len.float()
        ).mean().unsqueeze(0)

        # ── Decode ────────────────────────────────────────────────────────────────────
        generated_txts = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        # ── ROUGE (rouge_score library — computed per rank, reduced after) ────────────
        R1_sum, R2_sum, RL_sum = 0.0, 0.0, 0.0
        n = len(generated_txts)

        for pred, ref in zip(generated_txts, references):
            scores  = self.rouge.score(pred, ref)   # rouge_scorer API: .score(hypothesis, reference)
            R1_sum += scores["rouge1"].fmeasure
            R2_sum += scores["rouge2"].fmeasure
            RL_sum += scores["rougeLsum"].fmeasure     

        R1    = torch.tensor([R1_sum / n], device=self.gpu_id)
        R2    = torch.tensor([R2_sum / n], device=self.gpu_id)
        RLsum = torch.tensor([RL_sum / n], device=self.gpu_id)

        # ── BERTScore ─────────────────────────────────────────────────────────────────
        P, R, F1 = self.bert_scorer.score(generated_txts, references)
        bert_f1  = F1.mean().to(self.gpu_id).unsqueeze(0)

        # ── Write generation log (each rank writes its own) ───────────────────
        log_path = Path(os.environ.get("log_generation_txt", "./generation_logs"))
        log_path.mkdir(parents=True, exist_ok=True)
        with open(log_path / f"generations_log_rank{self.gpu_id}.txt", "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"{mode.upper()} | EPOCH: {epoch} | STEP: {step}\n")
            f.write(f"{'='*80}\n")
            for i, (gen, ref) in enumerate(zip(generated_txts, references)):
                f.write(f"\n--- Sample {i} ---\n")
                f.write(f"Target Length  : {batch['target_len'][i].item()}\n")
                f.write(f"Generated Len  : {generate_len[i].item()}\n")
                f.write(f"REFERENCE      :\n {ref}\n")
                f.write(f"GENERATED      :\n {gen}\n")
                f.write("-" * 40 + "\n")

        # ── All-reduce metrics across ranks ───────────────────────────────────
        for t in (MAE_score, R1, R2, RLsum, bert_f1):
            self._reduce_tensor(t, op=dist.ReduceOp.AVG)

        # ── TensorBoard (rank 0 only) ─────────────────────────────────────────
        gen_step = (
            epoch * (self.dico_len_data[mode] // self.evaluation_timing)
            + step // self.evaluation_timing
        )
        self._log_scalar(f"{mode}/gen_mae",     MAE_score.item(), gen_step)
        self._log_scalar(f"{mode}/gen_rouge1",  R1.item(),        gen_step)
        self._log_scalar(f"{mode}/gen_rouge2",  R2.item(),        gen_step)
        self._log_scalar(f"{mode}/gen_rougeL",  RLsum.item(),     gen_step)
        self._log_scalar(f"{mode}/gen_bert_f1", bert_f1.item(),   gen_step)

        if self.is_master:
            print(
                f"  GEN | R1={R1.item():.3f}  R2={R2.item():.3f}  "
                f"RL={RLsum.item():.3f}  BERTf1={bert_f1.item():.3f}  "
                f"MAE={MAE_score.item():.2f}"
            )

        del generated_ids, gen_tokens, valid_mask, generate_len

    # ── Batch ─────────────────────────────────────────────────────────────────

    def _run_batch(self, batch: dict, epoch: int, step: int, mode: str) -> float:

        references = batch.pop("references")

        # Move tensors to GPU
        batch = {
            k: v.to(self.gpu_id, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }


        # ── Forward ───────────────────────────────────────────────────────────
        if mode == "train":
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(**batch)
        else:
            self.model.eval()
            with torch.no_grad():
                output = self.model(**batch)

        loss   = getattr(output, "loss", None)
        labels = batch.get("labels", None)
        n_tokens = (labels != -100).sum() if labels is not None else torch.tensor(0)

        # ── Backward ──────────────────────────────────────────────────────────
        if mode == "train":
            if n_tokens > 0 and loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                safe_loss = loss.detach()
            else:
                # DDP requires all ranks to call backward — use dummy
                dummy = (output.logits * 0).sum()
                dummy.backward()
                safe_loss = torch.tensor(0.0, device=self.gpu_id)
                if self.is_master:
                    print(f"  ⚠️  Step {step}: empty/NaN batch — dummy gradient used.")
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
        else:
            safe_loss = loss.detach() if loss is not None else torch.tensor(0.0, device=self.gpu_id)

        # ── Reduce loss for logging ────────────────────────────────────────────
        reduced_loss = self._reduce_tensor(safe_loss.clone(), op=dist.ReduceOp.AVG)

        global_step = epoch * self.dico_len_data[mode] + step
        self._log_scalar(f"{mode}/batch_loss", reduced_loss.item(), global_step)
        
        if mode == "train":

            if self.scheduler is not None:
                lrs = self.scheduler.get_last_lr()
                for i, lr in enumerate(lrs):
                    self._log_scalar(f"{mode}/learning_rate_group{i}", lr, global_step)
            else:
                for i, group in enumerate(self.optimizer.param_groups):
                    lr = group["lr"]
                    self._log_scalar(f"{mode}/learning_rate_group{i}", lr, global_step)

            # ── PRE gate logging ──────────────────────────────────────────────
            if self.config.pre_status:
                model_ref = self.model.module if hasattr(self.model, "module") else self.model
                gate      = model_ref.model.pre_gate  

                gate_raw    = gate.data.item()                  # raw learned value
                gate_scaled = torch.tanh(gate).data.item()      # actual injection weight

                self._log_scalar(f"{mode}/pre_gate_raw",    gate_raw,    global_step)
                self._log_scalar(f"{mode}/pre_gate_scaled", gate_scaled, global_step)

                if gate.grad is not None:
                    self._log_scalar(f"{mode}/pre_gate_grad", gate.grad.item(), global_step)

        if self.is_master:
            print(
                f"{mode.upper()} | Loss: {reduced_loss.item():.4f} "
                f"| Epoch {epoch} | Step {step}/{self.dico_len_data[mode]}"
            )

        # ── Generation + metrics every evaluation_timing steps ────────────────
        if step % self.evaluation_timing == 0:
            # Put references back (not a tensor — used in generation only)
            batch["references"] = references
            self._run_generation(batch, epoch, step, mode)

        return float(reduced_loss.item())

    # ── Epoch ─────────────────────────────────────────────────────────────────

    def _run_epoch(self, epoch: int, mode: str = "train") -> Optional[bool]:
        total_local_loss = 0.0
        dataloader = self.train_data if mode == "train" else self.eval_data

        if mode == "train" and hasattr(dataloader, "sampler"):
            dataloader.sampler.set_epoch(epoch)

        for step, batch in enumerate(dataloader):
            total_local_loss += self._run_batch(batch, epoch, step, mode)

        # ── Epoch loss ────────────────────────────────────────────────────────
        epoch_loss = torch.tensor(
            total_local_loss / len(dataloader), device=self.gpu_id
        )
        self._reduce_tensor(epoch_loss, op=dist.ReduceOp.AVG)

        self._log_scalar(f"{mode}/epoch_loss", epoch_loss.item(), epoch)
        if self.is_master:
            print(f"### {mode.upper()} | Epoch {epoch} | Loss: {epoch_loss.item():.4f}")

        # ── Early stopping (eval only) ────────────────────────────────────────
        if mode == "eval":
            self.list_loss.append(epoch_loss.item())

            if epoch_loss.item() < self.best_loss:
                self.best_loss = epoch_loss.item()
                self.counter   = 0
            else:
                self.counter  += 1

            terminate = self.counter >= self.max_patience
            if terminate and self.is_master:
                print(
                    f"  Early stopping triggered. "
                    f"Best: {self.best_loss:.4f} | History: {self.list_loss}"
                )

            # Broadcast terminate decision from rank 0 to all ranks
            terminate_tensor = torch.tensor(int(terminate), device=self.gpu_id)
            self._reduce_tensor(terminate_tensor, op=dist.ReduceOp.MAX)
            return bool(terminate_tensor.item())

        return False

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int) -> None:
        if self.checkpoint_path is None:
            return
        model_ref = self.model.module if isinstance(self.model, DDP) else self.model
        model_ref.save_pretrained(self.checkpoint_path)
        self.tokenizer.save_pretrained(self.checkpoint_path)
        print(f"  Checkpoint saved → {self.checkpoint_path}  (epoch {epoch})")
        

    # ── Public ────────────────────────────────────────────────────────────────

    def train(self, max_epochs: int) -> None:
        for epoch in range(max_epochs):
            self._run_epoch(epoch, mode="train")
            terminate = self._run_epoch(epoch, mode="eval")

            # Save only when eval loss improved (rank 0 only)
            if self.is_master and self.list_loss[-1] <= self.best_loss:
                self._save_checkpoint(epoch)
            
            if dist.is_initialized():
                dist.barrier()

            if terminate:
                break