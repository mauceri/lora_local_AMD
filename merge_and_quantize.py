#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusionne un adaptateur LoRA avec le modèle de base puis (optionnellement)
produit des fichiers GGUF quantifiés via llama.cpp.

Exemple (merge uniquement) :
    python merge_and_quantize.py \\
        --base models/phi4 \\
        --lora checkpoints/phi4-lora-vtest \\
        --out gguf_out/phi4_merged/merged_hf

Exemple (merge + GGUF Q4/Q5/Q6) :
    python merge_and_quantize.py \\
        --base models/phi4 \\
        --lora checkpoints/phi4-lora-vtest \\
        --out gguf_out/phi4_merged/merged_hf \\
        --gguf-dir gguf_out/phi4_merged \\
        --llama-cpp /path/to/llama.cpp \\
        --quantize-types Q4_K_M Q5_K_M Q6_K
"""

import argparse
import os
from pathlib import Path
import subprocess
import logging
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Logger simple vers stdout + fichier
LOG_PATH = "merge_and_quantize.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, encoding="utf-8", mode="a"),
    ],
)
log = logging.getLogger("merge_and_quantize")


def merge_lora(base: str, lora: str, out: str, dtype: str = "float16") -> None:
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    log.info(f"[merge] Chargement base={base} dtype={dtype}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch_dtype,
        device_map="cpu",
        local_files_only=True,
    )
    log.info(f"[merge] Chargement LoRA depuis {lora}")
    lora_model = PeftModel.from_pretrained(base_model, lora, local_files_only=True)
    merged = lora_model.merge_and_unload()

    log.info(f"[merge] Sauvegarde dans {out}")
    os.makedirs(out, exist_ok=True)
    merged.save_pretrained(out)
    tok = AutoTokenizer.from_pretrained(base, use_fast=True, local_files_only=True)
    tok.save_pretrained(out)
    log.info("[merge] Terminé.")


def run(cmd: List[str], cwd: str = None) -> None:
    log.info("[cmd] " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def quantize_gguf(merged_dir: str, gguf_dir: str, llama_cpp: str, qtypes: List[str]) -> None:
    llama_cpp = llama_cpp.rstrip("/")

    convert_script = os.path.join(llama_cpp, "convert_hf_to_gguf.py")
    quant_bin = os.path.join(llama_cpp, "build/bin/llama-quantize")
    print("llama_cpp =", repr(llama_cpp))
    print("convert_script =", repr(convert_script))
    print("quant_bin =", repr(quant_bin))
    print(llama_cpp)

    if not os.path.isfile(convert_script):
        raise FileNotFoundError(f"{convert_script} introuvable dans {llama_cpp}")
    if not os.path.isfile(quant_bin):
        raise FileNotFoundError(f"{quant_bin} introuvable dans {llama_cpp}. Compilez llama.cpp avant.")

    os.makedirs(gguf_dir, exist_ok=True)
    f16_path = os.path.join(gguf_dir, "model-f16.gguf")

    print(f"[gguf] Conversion HF -> GGUF (f16) vers {f16_path}")
    log.info(f"[gguf] Conversion HF -> GGUF (f16) vers {f16_path}")
    run(["python3", convert_script, "--outtype", "f16", "--outfile", f16_path, merged_dir])

    for q in qtypes:
        out_path = os.path.join(gguf_dir, f"model-{q}.gguf")
        log.info(f"[gguf] Quantization {q} -> {out_path}")
        run([quant_bin, f16_path, out_path, q])
    log.info("[gguf] Terminé.")


def parse_args():
    p = argparse.ArgumentParser("Fusion LoRA + quantization GGUF (optionnel)")
    p.add_argument("--base", required=True, help="Chemin du modèle de base (HF, local).")
    p.add_argument("--lora", required=True, help="Chemin du checkpoint LoRA (sortie Trainer).")
    p.add_argument("--out", required=True, help="Répertoire de sortie HF fusionné.")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"],
                   help="Dtype pour charger le modèle avant fusion.")
    p.add_argument("--gguf-dir", help="Répertoire cible pour les GGUF. Si absent, on ne quantize pas.")
    p.add_argument("--llama-cpp", help="Chemin vers la racine de llama.cpp (avec convert-hf-to-gguf.py et llama-quantize).")
    p.add_argument("--quantize-types", nargs="+", default=[],
                   help="Ex: Q4_K_M Q5_K_M Q6_K. Nécessite --gguf-dir et --llama-cpp.")
    return p.parse_args()


def main():
    args = parse_args()
    merge_lora(args.base, args.lora, args.out, args.dtype)

    if args.gguf_dir:
        if not args.llama_cpp:
            raise ValueError("Merci de fournir --llama-cpp pour produire les GGUF.")
        quantize_gguf(args.out, args.gguf_dir, args.llama_cpp, args.quantize_types)
    else:
        log.info("[info] Pas de répertoire GGUF fourni, aucune quantization réalisée.")


if __name__ == "__main__":
    main()
