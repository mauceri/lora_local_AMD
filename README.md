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

## Particularités ROCm
- Le script fixe `HSA_OVERRIDE_GFX_VERSION=10.3.0` et `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` pour les iGPU 680M/GFX1030.
- Si `torch.cuda.is_bf16_supported()` est faux, le script bascule en float16.

## Sorties
- L’adaptateur LoRA et le tokenizer sauvegardés dans `--output_dir`.
- Checkpoints intermédiaires selon `--save_strategy`.

## Notes
- Le dépôt ignore les poids volumineux (models/, checkpoints/) et garde le dossier `data/`.
- Le notebook `LoRA_optimized.ipynb` pourra être documenté plus tard pour l’export/quantization.
