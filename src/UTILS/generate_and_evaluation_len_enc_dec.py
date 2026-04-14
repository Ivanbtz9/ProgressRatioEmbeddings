import os, sys
import json
from pathlib import Path

import re

from datetime import datetime 
from datetime import timedelta

import tqdm
from functools import partial

import pandas as pd
import numpy as np
import random as rd

import argparse
from typing import Optional

# Load the ROUGE metric
from rouge_score import rouge_scorer
# import evaluate

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence

import torch.multiprocessing as mp

import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from transformers import BertTokenizer, BertModel
from bert_score import BERTScorer

def ddp_setup(timeout:int=3600):

    # ---------------------------------------------------------
    # 1. Get environment variables automatically set by torchrun
    # ---------------------------------------------------------

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    MASTER_ADDR = os.environ.get("MASTER_ADDR", "localhost")
    MASTER_PORT = os.environ.get("MASTER_PORT", "12355")

    # ---------------------------------------------------------
    # 2. DDP setup
    # ---------------------------------------------------------
    torch.cuda.set_device(local_rank)

    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_DEBUG"] = "WARN"

    init_process_group(
        backend="nccl", 
        rank=rank, 
        world_size=world_size, 
        timeout=timedelta(seconds=timeout))

    return world_size, local_rank, rank

def get_collate_fn(tokenizer,state_distrib:bool=True):
    # Pre-compile regex for speed
    id_pattern = re.compile(r'(\d).*?(\d)')
    
    def collate_fn(batch):

        ids = [item['id'] for item in batch]
        
        input_ids = [torch.as_tensor(item['input_ids'], dtype=torch.long) for item in batch]
        attention_mask = [torch.as_tensor(item['input_mask'], dtype=torch.long) for item in batch]
        
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        input_len = torch.tensor([item['input_len'] for item in batch], dtype=torch.long)

        if state_distrib:
            target_len = torch.tensor([item['target_len'] for item in batch], dtype=torch.long)    
        else:
            # Deterministic Ratio Extraction
            ratios_list = [
                float(m.group(1) + m.group(2)) if (m := id_pattern.search(id_doc)) else 50.0
                for id_doc in ids
            ]

            ratios = torch.as_tensor(ratios_list, dtype=torch.float).clamp(20, 95) / 100.0
            target_len = (ratios * input_len).clamp(200, 1000).long()

        return {
            'id': ids,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'input_len': input_len,
            'target_len': target_len,
            'highlights': [item['highlights'] for item in batch]
        }
    
    return collate_fn

def prepare_dataloader(dataset, batch_size, tokenizer, state_distrib=True, num_workers=4):
    sampler = DistributedSampler(dataset)
    # Instantiate the closure
    collate_fn = get_collate_fn(tokenizer,state_distrib)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=True, #performance booster for GPU training.
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers #how many separate CPU processes are spawned to load and preprocess your data
    )

class Evaluator:
    def __init__(self,
        model:torch.nn.Module,
        rank:int,
        task_name:str="summarization") -> None:

        self.rank = rank
        model.to(rank)

        task_params = (
            getattr(model.config, "task_specific_params", {}) or {}
        ).get(task_name)

        if task_params is None:
            task_params = (
                getattr(model.generation_config, "task_specific_params", {}) or {}
            ).get(task_name)

        if task_params is None:
            task_params = {
                "early_stopping": True,
                "length_penalty": 1.5,
                "max_length": 1024,
                "min_length": 12,
                "num_beams": 4,
            }

        self.generate_config_dict = task_params

                
        self.model = DDP(model, device_ids=[rank])#, find_unused_parameters=True
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
        
        if 'DSDIR' in os.environ:
            assert os.environ['DSDIR'] == "/lustre/fsmisc/dataset"
            LOCAL_BERT = str(Path(os.environ['DSDIR']) / "HuggingFace_Models/bert-base-uncased")
        else:
            LOCAL_BERT = "bert-base-uncased"  # only works if you have internet or cached files

        self.bert_scorer = BERTScorer(
            model_type=LOCAL_BERT, 
            num_layers=12,
            device=f"cuda:{rank}",
            rescale_with_baseline=False
        )
        self.txt_rows = []
        self.metrics_rows = []

    

    @torch.no_grad()
    def evaluate(self, data_loader, tokenizer):
        exclude_ids = torch.tensor([tokenizer.pad_token_id, tokenizer.unk_token_id, tokenizer.mask_token_id]).to(self.rank) #skip spécial token to skip <PAD>, <UNK>, <MASK>
        self.model.eval()
        for idx,batch in enumerate(data_loader,0):
            batch = {k: v.to(self.rank, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

            generated_ids = self.model.module.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                target_len=batch['target_len'],
                **self.generate_config_dict)
            
                
            mask = ~torch.isin(generated_ids, exclude_ids)
            generate_len = mask.sum(dim=1)-1 #substract the "decoder_start_token_id"=2 at the begining
            generated_txt = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for i in range(len(batch["id"])):
                rouge_scores = self.rouge.score(generated_txt[i],batch["highlights"][i])
                rouge_scores_dict = {key: score.fmeasure for key, score in rouge_scores.items()}
                P, R, F1 = self.bert_scorer.score([generated_txt[i]], [batch["highlights"][i]])
                # try:
                #     bleu_score = self.bleu.compute(predictions=[generated_txt[i]],references=[[batch["highlights"][i]]],)["bleu"]
                # except:
                #     bleu_score = 0


                # Save to DataFrame-like lists
                self.metrics_rows.append({
                    "id": batch["id"][i],
                    "input_len": batch["input_len"][i].item(),
                    "target_len": batch["target_len"][i].item(),
                    "generate_len": generate_len[i].item(),
                    "rouge1": rouge_scores_dict["rouge1"],
                    "rouge2": rouge_scores_dict["rouge2"],
                    "rougeL": rouge_scores_dict["rougeL"],
                    "rougeLsum": rouge_scores_dict["rougeLsum"],
                    # "bleu_score":bleu_score,
                    "bert_score_P":P.item(),
                    "bert_score_R":R.item(),
                    "bert_score_F1":F1.item()         
                })

                self.txt_rows.append({
                    "id": batch["id"][i],
                    "highlights": batch["highlights"][i],
                    "generated_txt": generated_txt[i]
                })
            
            print(f"RANK : {self.rank} | Batch {idx}/{len(data_loader)}")

        return (pd.DataFrame(self.metrics_rows), pd.DataFrame(self.txt_rows))


def main(model, tokenizer, dataset, batch_size, results_dir, state_distrib=True, task_name="summarization")->None:
    
    _, _, rank = ddp_setup()

    data_loader = prepare_dataloader(dataset, batch_size, tokenizer, state_distrib)

    evaluator = Evaluator(model,rank,task_name) 
    
    df_metrics_rows, df_txt_rows = evaluator.evaluate(data_loader,tokenizer)

    df_metrics_rows.to_csv(results_dir / f"metrics_rows_{rank}.csv",index=False)
    df_txt_rows.to_csv(results_dir / f"txt_rows_{rank}.csv",index=False)

    dist.barrier()

    