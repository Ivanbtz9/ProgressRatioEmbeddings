import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from typing import Optional


class Trainer:
    def __init__(self,
                 tokenizer: PreTrainedTokenizerFast,
                 model: torch.nn.Module,
                 train_data: DataLoader,
                 eval_data: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 gpu_id: int,
                 checkpoint_path: Optional[str] = None,
                 writer: Optional[SummaryWriter] = None,
                 max_patience: Optional[int] = 2
                 ) -> None:

        self.gpu_id = gpu_id
        self.tokenizer = tokenizer
        model.to(gpu_id)
        self.model = DDP(model, device_ids=[gpu_id])
        self.train_data = train_data
        self.eval_data = eval_data
        self.optimizer = optimizer
        self.checkpoint_path = checkpoint_path
        self.writer = writer
        self.best_loss = float('inf')
        self.counter = 0
        self.max_patience = max_patience
        self.list_loss = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _move_batch(self, batch: dict) -> dict:
        return {
            k: v.to(self.gpu_id, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

    def _compute_loss(self, batch: dict, step: int, mode: str) -> torch.Tensor:
        """Forward pass → normalized safe loss. Does NOT call backward."""
        output = self.model(**batch)
        loss = getattr(output, "loss", None)
        labels = batch.get("labels", None)
        batch_size = batch["input_ids"].size(0)

        n_tokens = torch.tensor(0, device=self.gpu_id, dtype=torch.float32)
        if torch.is_tensor(labels):
            n_tokens = (labels != -100).sum().to(torch.float32)

        if loss is None or n_tokens.item() == 0:
            print(f"[{mode}] only pad tokens or loss=None at step {step}")
            return torch.zeros((), device=self.gpu_id, dtype=torch.float32)

        return loss / batch_size

    def _reduce_loss(self, loss: torch.Tensor) -> float:
        """All-reduce a detached scalar loss across ranks → Python float."""
        reduced = loss.detach().clone()
        if dist.is_initialized():
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            reduced = reduced / dist.get_world_size()   # FIX: was missing
        return float(reduced.item())

    def _reduce_and_log_epoch(self, total_loss: float, mode: str, epoch: int) -> float:
        """Average loss across steps, reduce across ranks, log to tensorboard."""
        dataloader = self.train_data if mode == "train" else self.eval_data
        avg = torch.tensor(total_loss / len(dataloader), device=self.gpu_id)

        if dist.is_initialized():
            dist.all_reduce(avg, op=dist.ReduceOp.SUM)
            avg = avg / dist.get_world_size()

        scalar = float(avg.item())
        if self.gpu_id == 0 and self.writer is not None:
            self.writer.add_scalar(f"{mode}/epoch_loss", scalar, epoch)
            print(f"### {mode.upper()} | Epoch {epoch} | Loss: {scalar:.4f}")
        return scalar

    # ------------------------------------------------------------------
    # Batch steps
    # ------------------------------------------------------------------

    def _run_batch_train(self, batch: dict, epoch: int, step: int) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        batch = self._move_batch(batch)
        safe_loss = self._compute_loss(batch, step, mode="train")
        safe_loss.backward()
        self.optimizer.step()

        loss_val = self._reduce_loss(safe_loss)
        if self.gpu_id == 0 and self.writer is not None:
            gs = epoch * len(self.train_data) + step
            self.writer.add_scalar("train/batch_loss", loss_val, gs)
            print(f"TRAIN | Loss: {loss_val:.4f} | Epoch {epoch} | Step {step}/{len(self.train_data)}")
        return loss_val

    @torch.no_grad()
    def _run_batch_eval(self, batch: dict, epoch: int, step: int) -> float:
        self.model.eval()

        batch = self._move_batch(batch)
        safe_loss = self._compute_loss(batch, step, mode="eval")
        loss_val = self._reduce_loss(safe_loss)

        if self.gpu_id == 0 and self.writer is not None:
            gs = epoch * len(self.eval_data) + step
            self.writer.add_scalar("eval/batch_loss", loss_val, gs)
            print(f"EVAL  | Loss: {loss_val:.4f} | Epoch {epoch} | Step {step}/{len(self.eval_data)}")
        return loss_val

    # ------------------------------------------------------------------
    # Epoch runners 
    # ------------------------------------------------------------------

    def _run_train_epoch(self, epoch: int) -> None:
        self.train_data.sampler.set_epoch(epoch)
        total = sum(
            self._run_batch_train(batch, epoch, step)
            for step, batch in enumerate(self.train_data)
        )
        self._reduce_and_log_epoch(total, mode="train", epoch=epoch)

    def _run_eval_epoch(self, epoch: int) -> bool:
        """Returns True when early stopping should fire."""
        total = sum(
            self._run_batch_eval(batch, epoch, step)
            for step, batch in enumerate(self.eval_data)
        )
        epoch_loss = self._reduce_and_log_epoch(total, mode="eval", epoch=epoch)
        self.list_loss.append(epoch_loss)

        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self.counter = 0
            self._save_checkpoint(epoch)
        else:
            self.counter += 1

        terminate = self.counter >= self.max_patience
        if self.gpu_id == 0 and terminate:
            print(f"Early stopping | best: {self.best_loss:.4f} | history: {self.list_loss}")

        # Broadcast decision so every rank agrees
        if dist.is_initialized():
            t = torch.tensor(int(terminate), device=self.gpu_id)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            terminate = bool(t.item())

        return terminate

    # ------------------------------------------------------------------
    # Checkpoint — rank-0 guard lives here, not in the caller
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int) -> None:
        if self.gpu_id != 0 or self.checkpoint_path is None:
            return
        self.tokenizer.save_pretrained(self.checkpoint_path)
        self.model.module.save_pretrained(self.checkpoint_path)
        print(f"Epoch {epoch} | Checkpoint saved at {self.checkpoint_path}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self, max_epochs: int) -> None:
        for epoch in range(max_epochs):
            self._run_train_epoch(epoch)
            if self._run_eval_epoch(epoch):  
                break
