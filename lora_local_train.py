#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA training (classique) — ROCm friendly, autonome.
Aucun export à faire dans le shell : le script pose tout ce qu'il faut.
"""

import os, argparse
from typing import Dict, Any, List

# --- Allocateur PyTorch (avant tout import torch) ----------------------------
# Option robuste et compatible : on ne met que expandable_segments:True.
# (Les autres options peuvent varier selon les builds et casser silencieusement.)
_ALLOC_CONF = "expandable_segments:True"
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", _ALLOC_CONF)     # ROCm
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", _ALLOC_CONF)    # inoffensif si CUDA
# ROCm iGPU (ex. Ryzen 680M) a parfois besoin de l'override pour matcher l'arch.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

from datasets import load_dataset, Dataset, DatasetDict
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType

# ------------------------- Utils -------------------------

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
            # On prend la DERNIÈRE occurrence du tag assistant comme début des cibles
            last = -1
            L = len(tpl_ids)
            for j in range(0, len(ids) - L + 1):
                if ids[j:j+L] == tpl_ids:
                    last = j
            if last >= 0:
                start = last + L
                # borne fin = dernière position "active" de l'attention
                end = max(i for i, a in enumerate(attn) if a == 1) + 1
                lab[start:end] = ids[start:end]
            labels.append(lab)
        t["labels"] = labels
        return t
    return tok_and_mask

def pack_constant_length(ds_split: Dataset, max_len: int) -> Dataset:
    big_ids, big_mask, big_lab = [], [], []
    for rec in ds_split:
        big_ids.extend(rec["input_ids"])
        big_mask.extend(rec["attention_mask"])
        big_lab.extend(rec["labels"])
    L = min(len(big_ids), len(big_mask), len(big_lab))
    L = (L // max_len) * max_len
    big_ids, big_mask, big_lab = big_ids[:L], big_mask[:L], big_lab[:L]
    chunks = []
    for i in range(0, L, max_len):
        chunks.append({
            "input_ids": big_ids[i:i+max_len],
            "attention_mask": big_mask[i:i+max_len],
            "labels": big_lab[i:i+max_len],
        })
    return Dataset.from_list(chunks)

class SimpleCausalCollator:
    """Pad (input_ids, attention_mask, labels) à la même longueur.
    Labels sont paddés à -100 pour ignorer la perte sur le padding.
    """
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tok = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        # on sépare labels pour que tokenizer.pad ne les touche pas
        labels = [f["labels"] for f in features]
        feats_wo_labels = [{k: v for k, v in f.items() if k != "labels"} for f in features]
        batch = self.tok.pad(
            feats_wo_labels,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].size(1)
        import torch as _T
        padded_labels = []
        for lab in labels:
            if len(lab) < max_len:
                lab = lab + [-100] * (max_len - len(lab))
            else:
                lab = lab[:max_len]
            padded_labels.append(lab)
        batch["labels"] = _T.tensor(padded_labels, dtype=_T.long)
        return batch

class LossLoggerCallback:
    """Callback simple pour logguer la perte à chaque step voulu."""
    def __init__(self, log_path: str, every: int = 1):
        self.log_path = log_path
        self.every = max(1, every)
        # On ouvre le fichier en append pour conserver l'historique
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self._fh = open(log_path, "a", encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # Un log est émis toutes les logging_steps par Trainer
        if "loss" in logs:
            line = f"step={state.global_step} loss={logs['loss']:.4f}"
            if "learning_rate" in logs:
                line += f" lr={logs['learning_rate']:.2e}"
            print(line, flush=True)
            self._fh.write(line + "\n")
            self._fh.flush()

    def on_train_end(self, args, state, control, **kwargs):
        try:
            self._fh.close()
        except Exception:
            pass

# ------------------------- CLI -------------------------

def parse_args():
    p = argparse.ArgumentParser("LoRA training (classique) – ROCm friendly (autonome)")

    # Defaults calqués sur votre commande
    p.add_argument("--model_path", type=str, default="models/phi4")
    p.add_argument("--data_file", type=str, default="data/train.jsonl",
                   help="JSONL avec colonne 'messages' (role/content)")
    p.add_argument("--output_dir", type=str, default="checkpoints_phi4_lora")

    p.add_argument("--assistant_tag", type=str, default="<|assistant|>:", help="Préfixe de la réponse")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--packing", action="store_true", default=True, help="Constant-length packing")
    p.add_argument("--num_proc", type=int, default=4)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--per_device_batch", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--optimizer", type=str, default="adamw_torch")

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Dispositif / dtype / mémoire
    p.add_argument("--bf16", action="store_true", default=False,
                   help="Utiliser bfloat16 si supporté, sinon bascule auto en float16.")
    p.add_argument("--gpu_mem_gib", type=int, default=12, help="Budget VRAM pour device_map=auto")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto","cuda","cpu"],
                   help="Forcer l'emplacement du modèle. 'auto' laisse Accelerate décider.")

    # Divers
    p.add_argument("--save_strategy", type=str, default="epoch", choices=["epoch","steps","no"])
    p.add_argument("--save_steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--log_every", type=int, default=1, help="Nombre de pas entre deux logs (perte).")
    p.add_argument("--log_file", type=str, default="training.log", help="Fichier texte de log des pertes.")
    return p.parse_args()

# ------------------------- Main -------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    have_cuda = torch.cuda.is_available()
    on_hip = getattr(torch.version, "hip", None) is not None

    # 1) Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    # 2) Dataset -> text -> tokenisation + labels -> (packing optionnel)
    ds = load_dataset("json", data_files=args.data_file)
    ds_txt = ds.map(messages_to_text, num_proc=args.num_proc)
    tok_and_mask = build_tokenize_and_label_fn(tok, args.assistant_tag, args.max_len)
    remove_cols = [c for c in ds_txt["train"].column_names if c != "text"]
    ds_tok = {k: ds_txt[k].map(tok_and_mask, batched=True, num_proc=args.num_proc, remove_columns=remove_cols)
              for k in ds_txt.keys()}
    ds_tok = DatasetDict(ds_tok)

    if args.packing:
        ds_packed = {k: pack_constant_length(ds_tok[k], args.max_len) for k in ds_tok.keys()}
        ds_packed = DatasetDict(ds_packed)
        train_dataset_final = ds_packed["train"]
    else:
        train_dataset_final = ds_tok["train"]

    # 3) Modèle (chargement contrôlé)
    # Sélection dtype: bf16 si demandé ET supporté, sinon fp16 (CPU: fp32)
    bf16_supported = False
    if have_cuda:
        try:
            bf16_supported = torch.cuda.is_bf16_supported()
        except Exception:
            bf16_supported = False
    want_bf16 = args.bf16 and have_cuda and bf16_supported

    torch_dtype = torch.float32 if not have_cuda else torch.float16
    if want_bf16:
        torch_dtype = torch.bfloat16
    elif args.bf16 and have_cuda and not bf16_supported:
        print("bf16 demandé mais non supporté par ce GPU ROCm -> bascule en float16.")

    max_memory = {0: f"{args.gpu_mem_gib}GiB", "cpu": "256GiB"} if have_cuda else None
    device_map = "auto" if have_cuda else "cpu"

    if not have_cuda:
        print("Aucun GPU détecté par PyTorch (torch.cuda.is_available() == False). "
              "Le modèle sera chargé sur CPU. "
              "Installez un build PyTorch ROCm si vous souhaitez utiliser la carte AMD.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=device_map,
        max_memory=max_memory,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    # Eager attn si dispo, et cache off pour l'entraînement
    try:
        model.config.attn_implementation = "eager"
    except Exception:
        pass
    model.config.use_cache = False

    # 4) LoRA
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # 5) Entraînement (Trainer)
    collator = SimpleCausalCollator(tok, pad_to_multiple_of=8)
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=args.log_every,
        logging_strategy="steps",
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        report_to="none",
        bf16=(torch_dtype == torch.bfloat16),
        fp16=(torch_dtype == torch.float16),
        optim=args.optimizer,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=False,
        eval_strategy="no",
        seed=args.seed,
    )
    loss_logger = LossLoggerCallback(args.log_file, args.log_every)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_dataset_final,
        eval_dataset=None,
        data_collator=collator,
        tokenizer=tok,
        callbacks=[loss_logger],
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from)
    print(train_result)
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
