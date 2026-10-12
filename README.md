# Exploring Spectral Concentration of LoRA Backdoors in Vision Transformers

This repository trains and evaluates clean and backdoored ViT models with several fine-tuning methods, then compresses LoRA query weights with SVD and verifies the compressed backdoor performance.

## 0. Configure Runtime Environment

Create and activate a Python environment:

```bash
conda create -n lora-vit python=3.13 -y
conda activate lora-vit
```

Install PyTorch according to your CUDA version. For example, for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining Python dependencies:

```bash
pip install transformers peft safetensors pillow tqdm matplotlib huggingface_hub
```

If you are using a different CUDA version or CPU-only environment, replace the PyTorch install command with the one generated from the official PyTorch installation page.

## 1. Prepare Raw Datasets

Put raw datasets under `datasets_raw/` with the following structure.

### ImageNet-1k

Official dataset link: https://image-net.org/challenges/LSVRC/2017/2017-downloads

```text
datasets_raw/imagenet-1k/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
├── val/
│   ├── ILSVRC2012_val_00000001.JPEG
│   └── ...
├── ImageNet_val_label.txt
└── ImageNet_label.json
```

`ImageNet_val_label.txt` should contain one sample per line:

```text
image_name.JPEG wordnet_id
```

`ImageNet_label.json` should map WordNet IDs to integer labels:

```json
{
  "n01440764": 0
}
```

### ImageNet-V2

Official dataset link: https://github.com/modestyachts/ImageNetV2

```text
datasets_raw/imagenet-v2/imagenetv2-matched-frequency/
├── 0/
├── 1/
├── 2/
└── ...
```

The ImageNet-V2 class folders are expected to be integer class IDs.

## 2. Prepare ViT Models

The code defaults to these local model directories:

```text
models/ViT-base-16-224/
models/ViT-large-16-384/
```

Download the models from HuggingFace:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/vit-base-patch16-224",
    local_dir="models/ViT-base-16-224",
    local_dir_use_symlinks=False,
)

snapshot_download(
    repo_id="google/vit-large-patch16-384",
    local_dir="models/ViT-large-16-384",
    local_dir_use_symlinks=False,
)
```

Equivalent one-shot command:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

models = {
    "google/vit-base-patch16-224": "models/ViT-base-16-224",
    "google/vit-large-patch16-384": "models/ViT-large-16-384",
}

for repo_id, local_dir in models.items():
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
PY
```

## 3. Prepare Processed Data

Run both preprocessing scripts:

```bash
python deal_imagenet_1k.py
python deal_imagenet_v2.py
```

Expected ImageNet-1k outputs:

```text
datasets_processed/imagenet-1k/
├── train_c/
├── train_p/
├── val_p/
├── ImageNet10_c_train_label.txt
├── ImageNet10_p_train_label.txt
├── ImageNet_val_label.txt
└── ImageNet_label.json
```

Expected ImageNet-V2 outputs:

```text
datasets_processed/imagenet-v2/
├── ImageNet_v2_train_c/
├── ImageNet_v2_train_p/
├── ImageNet_v2_val_c/
├── ImageNet_v2_val_p/
├── train_c_label.txt
├── train_p_label.txt
├── val_c_label.txt
└── val_p_label.txt
```

Check processed data completeness:

```bash
for p in \
  datasets_processed/imagenet-1k/train_c \
  datasets_processed/imagenet-1k/train_p \
  datasets_processed/imagenet-1k/val_p \
  datasets_processed/imagenet-1k/ImageNet10_c_train_label.txt \
  datasets_processed/imagenet-1k/ImageNet10_p_train_label.txt \
  datasets_processed/imagenet-1k/ImageNet_val_label.txt \
  datasets_processed/imagenet-1k/ImageNet_label.json \
  datasets_processed/imagenet-v2/ImageNet_v2_train_c \
  datasets_processed/imagenet-v2/ImageNet_v2_train_p \
  datasets_processed/imagenet-v2/ImageNet_v2_val_c \
  datasets_processed/imagenet-v2/ImageNet_v2_val_p \
  datasets_processed/imagenet-v2/train_c_label.txt \
  datasets_processed/imagenet-v2/train_p_label.txt \
  datasets_processed/imagenet-v2/val_c_label.txt \
  datasets_processed/imagenet-v2/val_p_label.txt; do
  [ -e "$p" ] && echo "OK   $p" || echo "MISS $p"
done
```


## 4. Train and Evaluate with `main.py`

Run the default full experiment grid:

```bash
python main.py
```

The default grid runs:

- algorithms: `adapter-h`, `adapter-l`, `lora-o`, `lora-l`, `bitfit`, `ft`
- model types: `base`, `large`
- data types: `1k`, `v2`

Outputs are saved under `results/`:

```text
results/exp_<model_type>_<data_type>/
├── ckpt_<algorithm>/
└── log_<algorithm>/
```

Clean accuracy and poison ASR logs are written to files such as:

```text
results/exp_base_1k/logs_lora-o/clean_eval_log.txt
results/exp_base_1k/logs_lora-o/poison_eval_log.txt
```

Run a smaller example:

```bash
python main.py \
  --algorithm lora-o \
  --model-type base \
  --data-type 1k \
  --epochs 30 \
  --eval-batch-size 64
```

Useful options:

```bash
--output-root results
--model-base-path models/ViT-base-16-224/
--model-large-path models/ViT-large-16-384/
--imagenet-1k-root datasets_processed/imagenet-1k
--imagenet-v2-root datasets_processed/imagenet-v2
--resume-from-checkpoint <checkpoint_dir>
```

## 5. Compress Backdoored LoRA and Verify Performance

Use `svd.py` on a trained LoRA checkpoint directory that contains `adapter_model.safetensors`.

Example:

```bash
python svd.py \
  --lora-ckpt-path results/exp_base_1k/ckpt_lora-o/checkpoint-4710 \
  --energy-threshold 0.95 \
  --model-path models/ViT-base-16-224/ \
  --data-type 1k \
  --eval-batch-size 64
```

`svd.py` automatically creates a compressed checkpoint directory by appending `-c`:

```text
results/exp_base_1k/ckpt_lora-o/checkpoint-4710-c/
```

It first copies all files from the original checkpoint directory, then overwrites only:

```text
adapter_model.safetensors
```

The final output contains four evaluation results:

```text
ACC before compression
ACC after compression
ASR before compression
ASR after compression
```

Run compression only:

```bash
python svd.py \
  --lora-ckpt-path results/exp_base_1k/ckpt_lora-o/checkpoint-4710 \
  --energy-threshold 0.95 \
  --no-eval
```

Run evaluation only when the `-c` checkpoint already exists:

```bash
python svd.py \
  --lora-ckpt-path results/exp_base_1k/ckpt_lora-o/checkpoint-4710 \
  --no-compress
```
