#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Évalue un modèle fusionné (non quantifié) sur un corpus JSONL (data/test.jsonl).
Calcule la loss moyenne et la perplexité via Trainer.evaluate().

Exemple :
    python evaluate_merged.py \\
        --model gguf_out/phi4_merged/merged_hf \\
        --data_file data/test.jsonl \\
        --assistant_tag "<|assistant|>:" \\
        --max_len 512 \\
        --per_device_batch 1
"""

import argparse
import os
import logging
from typing import Any, Dict, List

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments


# --- Allocateur PyTorch (avant tout import torch) ----------------------------
# Option robuste et compatible : on ne met que expandable_segments:True.
# (Les autres options peuvent varier selon les builds et casser silencieusement.)
_ALLOC_CONF = "expandable_segments:True"
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", _ALLOC_CONF)     # ROCm
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", _ALLOC_CONF)    # inoffensif si CUDA
# ROCm iGPU (ex. Ryzen 680M) a parfois besoin de l'override pour matcher l'arch.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")


# Logger vers stdout + fichier
LOG_PATH = "evaluate_merged.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, encoding="utf-8", mode="a"),
    ],
)
log = logging.getLogger("evaluate_merged")

def messages_to_text(ex: Dict[str, Any]) -> Dict[str, str]:
    msgs = ex.get("messages", [])
    text = "\n".join(f"<|{m.get('role','user')}|>: {m.get('content','')}" for m in msgs)
    return {"text": text}


def build_tokenize_and_label_fn(tokenizer, assistant_tag: str, max_len: int):
    tpl_ids: List[int] = tokenizer.encode(assistant_tag, add_special_tokens=False)

    def tok_and_mask(batch):
        t = tokenizer(batch["text"], truncation=True, max_length=max_len)
        labels = []
        for ids, attn in zip(t["input_ids"], t["attention_mask"]):
            lab = [-100] * len(ids)
            last = -1
            L = len(tpl_ids)
            for j in range(0, len(ids) - L + 1):
                if ids[j:j+L] == tpl_ids:
                    last = j
            if last >= 0:
                start = last + L
                end = max(i for i, a in enumerate(attn) if a == 1) + 1
                lab[start:end] = ids[start:end]
            labels.append(lab)
        t["labels"] = labels
        return t

    return tok_and_mask


class SimpleCausalCollator:
    """Pad (input_ids, attention_mask, labels) à la même longueur."""

    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tok = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        labels = [f["labels"] for f in features]
        feats_wo_labels = [{k: v for k, v in f.items() if k != "labels"} for f in features]
        batch = self.tok.pad(
            feats_wo_labels,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].size(1)
        padded_labels = []
        for lab in labels:
            if len(lab) < max_len:
                lab = lab + [-100] * (max_len - len(lab))
            else:
                lab = lab[:max_len]
            padded_labels.append(lab)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def parse_args():
    p = argparse.ArgumentParser("Évaluation d'un modèle fusionné (HF) sur un JSONL.")
    p.add_argument("--model", required=True, help="Répertoire du modèle fusionné (HF).")
    p.add_argument("--data_file", type=str, default="data/test.jsonl")
    p.add_argument("--assistant_tag", type=str, default="<|assistant|>:", help="Début de la réponse.")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--per_device_batch", type=int, default=1)
    p.add_argument("--num_proc", type=int, default=2)
    p.add_argument("--bf16", action="store_true", help="bf16 si supporté, sinon fp16.")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    log.info("=== Évaluation modèle fusionné ===")
    log.info(f"model={args.model} data_file={args.data_file} device={args.device} bf16={args.bf16}")

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    # Dataset
    ds = load_dataset("json", data_files=args.data_file)
    ds_txt = ds.map(messages_to_text, num_proc=args.num_proc)
    tok_and_mask = build_tokenize_and_label_fn(tok, args.assistant_tag, args.max_len)
    remove_cols = [c for c in ds_txt["train"].column_names if c != "text"]
    ds_tok = ds_txt.map(tok_and_mask, batched=True, num_proc=args.num_proc, remove_columns=remove_cols)
    eval_dataset = ds_tok["train"]
    log.info(f"Dataset chargé : {len(eval_dataset)} exemples")

    # dtype et device
    have_cuda = torch.cuda.is_available()
    bf16_supported = False
    if have_cuda:
        try:
            bf16_supported = torch.cuda.is_bf16_supported()
        except Exception:
            bf16_supported = False
    use_bf16 = args.bf16 and have_cuda and bf16_supported
    torch_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if have_cuda else torch.float32)
    device_map = args.device if args.device != "auto" else ("auto" if have_cuda else "cpu")
    log.info(f"dtype={torch_dtype} device_map={device_map}")

    # Modèle
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        local_files_only=True,
    )
    model.config.use_cache = False

    # Trainer pour eval
    collator = SimpleCausalCollator(tok, pad_to_multiple_of=8)
    targs = TrainingArguments(
        output_dir=os.path.join(args.model, "eval_tmp"),
        per_device_eval_batch_size=args.per_device_batch,
        report_to="none",
        fp16=(torch_dtype == torch.float16),
        bf16=(torch_dtype == torch.bfloat16),
        dataloader_pin_memory=False,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tok,
    )

    log.info("Début de l'évaluation…")
    metrics = trainer.evaluate()
    loss = metrics.get("eval_loss")
    if loss is not None:
        ppl = torch.exp(torch.tensor(loss)).item()
        log.info(f"eval_loss={loss:.4f}  perplexity={ppl:.4f}")
    log.info(f"Fin. Détails : {metrics}")


if __name__ == "__main__":
    main()
