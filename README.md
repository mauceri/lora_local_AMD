# LoRA local (AMD/ROCm)

Script autonome pour fine-tuner un adaptateur LoRA sur un modèle type Phi-4-mini-instruct en environnement ROCm (AMD). Les données attendues sont au format JSONL avec une colonne `messages` (role/content) dans `data/train.jsonl` (par défaut).

## Pré-requis
- Python 3.12
- PyTorch ROCm (testé avec `torch 2.5.1+rocm6.2`)
- `transformers`, `datasets`, `peft`, etc. (voir `requirements.txt`)

## Jeu de données attendu
- `data/train.jsonl` (et `data/test.jsonl` si besoin) avec des entrées de la forme :
```json
{"messages": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Réponse"}]}
```

## Modèle de base
Placez le modèle de base sous `models/phi4` (ou indiquez un autre chemin via `--model_path`). Le script charge en local (`local_files_only=True`), sans téléchargement.

## Lancement minimal
```bash
python lora_local_train.py \
  --model_path models/phi4 \
  --data_file data/train.jsonl \
  --output_dir checkpoints/phi4-lora-vtest
```

Paramètres utiles :
- `--max_len`: longueur de séquence (512 par défaut)
- `--epochs`: nombre d’époques (1 par défaut)
- `--per_device_batch`: batch par device (1)
- `--grad_accum`: accumulation de gradient (16)
- `--lr`: taux d’apprentissage (2e-5)
- `--bf16`: active bf16 si supporté (désactivé par défaut)
- `--device`: `auto|cuda|cpu` (auto par défaut)
- `--log_every`: pas de logging de la perte (1 = chaque step d’optimisation)
- `--log_file`: fichier texte pour les pertes (`training.log` par défaut)

## Particularités ROCm
- Le script fixe `HSA_OVERRIDE_GFX_VERSION=10.3.0` et `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` pour les iGPU 680M/GFX1030.
- Si `torch.cuda.is_bf16_supported()` est faux, le script bascule en float16.

## Sorties
- L’adaptateur LoRA et le tokenizer sauvegardés dans `--output_dir`.
- Checkpoints intermédiaires selon `--save_strategy`.
- Un logging des pertes pas-à-pas (stdout + fichier configurable).

## Fusion + quantization (merge_and_quantize.py)
- Fusionne l’adaptateur avec le modèle de base en HF.
- Optionnel : conversion GGUF via llama.cpp et quantization (ex. Q4_K_M, Q5_K_M, Q6_K).
Exemple fusion seule :
```bash
python merge_and_quantize.py \
  --base models/phi4 \
  --lora checkpoints/phi4-lora-vtest \
  --out gguf_out/phi4_merged/merged_hf
```
Fusion + GGUF :
```bash
python merge_and_quantize.py \
  --base models/phi4 \
  --lora checkpoints/phi4-lora-vtest \
  --out gguf_out/phi4_merged/merged_hf \
  --gguf-dir gguf_out/phi4_merged \
  --llama-cpp /chemin/vers/llama.cpp \
  --quantize-types Q4_K_M Q5_K_M Q6_K
```
Les logs sont écrits dans `merge_and_quantize.log`.

## Évaluation du modèle fusionné (evaluate_merged.py)
Calcule loss et perplexité sur un JSONL (par défaut `data/test.jsonl`).
```bash
python evaluate_merged.py \
  --model gguf_out/phi4_merged/merged_hf \
  --data_file data/test.jsonl \
  --assistant_tag "<|assistant|>:" \
  --max_len 512 \
  --per_device_batch 1
```
Logs console + `evaluate_merged.log`.

## Notes
- Le dépôt ignore les poids volumineux (models/, checkpoints/) et garde le dossier `data/`.
- Le notebook `LoRA_optimized.ipynb` pourra être documenté plus tard pour l’export/quantization.
