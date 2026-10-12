import argparse
import os
import re
import copy
import shutil
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from utils import compute_clean_accuracy, compute_asr
from utils import (ImageNet1kTrainDataset, CleanPoisonEvalCallback, compute_clean_accuracy, ImageNet1kValDataset,
                   ImageNetv2TrainDataset, ImageNetv2ValDataset)
from transformers import (
    ViTForImageClassification,
    AutoImageProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, PeftModel


def compress_all_lora_q_by_energy(
    lora_ckpt_path,
    energy_threshold=0.95,
):
    """
    Process query LoRA weights in every ViT block from the LoRA safetensors file:
    - Extract delta_W = B @ A layer by layer
    - Apply SVD decomposition
    - Keep only singular values whose cumulative energy ratio reaches energy_threshold
    - Reconstruct with low rank
    - Replace the corresponding lora_A / lora_B tensors in place
    - Save the new safetensors file

    Parameters:
        lora_ckpt_path: str
            LoRA checkpoint directory path; it must contain adapter_model.safetensors
        energy_threshold: float
            Cumulative singular-value energy threshold (default: 0.95)

    Returns:
        summary: dict
            key = layer_id
            value = {
                "original_rank": r,
                "kept_rank": k,
                "energy_ratio": float,
                "kept_singular_values": list[float]
            }
    """

    src_dir = Path(lora_ckpt_path)
    if not src_dir.exists():
        raise FileNotFoundError(lora_ckpt_path)
    if not src_dir.is_dir():
        raise NotADirectoryError(lora_ckpt_path)

    adapter_path = src_dir / "adapter_model.safetensors"
    if not adapter_path.exists():
        raise FileNotFoundError(adapter_path)

    state = load_file(adapter_path)
    new_state = dict(state)

    # --------- 1. Automatically parse all layer IDs ---------
    layer_pattern = re.compile(r"layer\.(\d+)")
    layer_ids = sorted({
        int(layer_pattern.search(k).group(1))
        for k in state.keys()
        if layer_pattern.search(k) and "query" in k
    })

    if len(layer_ids) == 0:
        raise RuntimeError("No ViT layers found in LoRA checkpoint.")

    summary = {}

    # --------- 2. Process query LoRA layer by layer ---------
    for layer_id in layer_ids:
        keys = [
            k for k in state.keys()
            if f"layer.{layer_id}" in k
            and "query" in k
            and ("lora_A" in k or "lora_B" in k)
        ]

        if len(keys) == 0:
            continue

        key_A = next((k for k in keys if "lora_A" in k), None)
        key_B = next((k for k in keys if "lora_B" in k), None)

        if key_A is None or key_B is None:
            continue

        A = state[key_A]  # [r, in_dim]
        B = state[key_B]  # [out_dim, r]
        r = A.shape[0]

        # ---------- ΔW ----------
        delta_W = B @ A

        # ---------- SVD ----------
        U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)

        sigma2 = S ** 2
        energy_cum = torch.cumsum(sigma2, dim=0) / sigma2.sum()

        # ---------- rank-k ----------
        k = int((energy_cum >= energy_threshold).nonzero()[0]) + 1

        U_k = U[:, :k]
        S_k = S[:k]
        Vh_k = Vh[:k, :]

        # ---------- Reconstruct with low rank ----------
        S_sqrt = torch.sqrt(S_k)

        B_k = (U_k * S_sqrt.unsqueeze(0)).contiguous()  # [out_dim, k]
        A_k = (S_sqrt.unsqueeze(1) * Vh_k).contiguous()  # [k, in_dim]

        # ---------- Pad back to the original rank r ----------
        in_dim, out_dim = A_k.shape[1], B_k.shape[0]
        B_new = torch.zeros(
            (out_dim, r),
            device=B.device,
            dtype=B.dtype,
        )
        A_new = torch.zeros(
            (r, in_dim),
            device=A.device,
            dtype=A.dtype,
        )

        B_new[:, :k] = B_k
        A_new[:k, :] = A_k

        # Ensure contiguous tensors for safetensors
        B_new = B_new.contiguous()
        A_new = A_new.contiguous()

        # ---------- Replace ----------
        new_state[key_A] = A_new
        new_state[key_B] = B_new

        summary[layer_id] = {
            "original_rank": r,
            "kept_rank": k,
            "energy_ratio": float(energy_cum[k - 1]),
            "kept_singular_values": S_k.tolist(),
        }

    # --------- 3. Copy the checkpoint directory and overwrite only adapter_model.safetensors ----------
    dst_dir = src_dir.with_name(f"{src_dir.name}-c")

    if src_dir.resolve() == dst_dir.resolve():
        raise ValueError("Compressed LoRA checkpoint directory must differ from the original directory.")

    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
    save_file(new_state, dst_dir / "adapter_model.safetensors")

    return summary


def load_and_merge_lora(trainer, base_model, lora_path, copy_base_model=True):
    """
    Load LoRA adapter weights into base_model, merge them into the base weights,
    and replace trainer.model with the merged model.

    copy_base_model=True keeps the original base_model unchanged, so the same
    base model can be reused for before/after compression evaluation.
    """
    model_to_merge = copy.deepcopy(base_model) if copy_base_model else base_model
    peft_model = PeftModel.from_pretrained(
        model_to_merge,
        lora_path,
    )
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()
    trainer.model = merged_model
    return merged_model


def get_compressed_lora_path(lora_ckpt_path):
    src_dir = Path(lora_ckpt_path)
    return str(src_dir.with_name(f"{src_dir.name}-c"))


def build_eval_datasets(data_type, processor, args):
    if data_type == "1k":
        processed_root = args.imagenet_1k_root
        raw_root = args.imagenet_1k_raw_root
        dataset_val_c = ImageNet1kValDataset(
            root=os.path.join(raw_root, "val"),
            label_file=os.path.join(raw_root, "ImageNet_val_label.txt"),
            label_map_file=os.path.join(processed_root, "ImageNet_label.json"),
            transform=processor,
        )
        dataset_val_p = ImageNet1kValDataset(
            root=os.path.join(processed_root, "val_p"),
            label_file=os.path.join(processed_root, "ImageNet_val_label.txt"),
            label_map_file=os.path.join(processed_root, "ImageNet_label.json"),
            transform=processor,
        )
        return dataset_val_c, dataset_val_p

    if data_type == "v2":
        root = args.imagenet_v2_root
        dataset_val_c = ImageNetv2ValDataset(
            root=os.path.join(root, "ImageNet_v2_val_c"),
            label_file=os.path.join(root, "val_c_label.txt"),
            transform=processor,
        )
        dataset_val_p = ImageNetv2ValDataset(
            root=os.path.join(root, "ImageNet_v2_val_p"),
            label_file=os.path.join(root, "val_p_label.txt"),
            transform=processor,
        )
        return dataset_val_c, dataset_val_p

    raise ValueError(f"Unsupported data type: {data_type}")


def evaluate_lora(base_model, trainer, lora_path, metric_name):
    load_and_merge_lora(
        trainer,
        base_model,
        lora_path,
    )
    metrics = trainer.evaluate()
    print(f"{metric_name}: {metrics}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Compress LoRA query weights with SVD and evaluate before/after metrics.")
    parser.add_argument(
        "--lora-ckpt-path",
        default="results/exp_base_1k/ckpt_lora-o/checkpoint-4710",
        help="LoRA checkpoint directory containing adapter_model.safetensors.",
    )
    parser.add_argument("--energy-threshold", type=float, default=0.95)
    parser.add_argument("--model-path", default="models/ViT-base-16-224/")
    parser.add_argument("--data-type", choices=("1k", "v2"), default="1k")
    parser.add_argument("--imagenet-1k-root", default="datasets_processed/imagenet-1k")
    parser.add_argument("--imagenet-1k-raw-root", default="datasets_raw/imagenet-1k")
    parser.add_argument("--imagenet-v2-root", default="datasets_processed/imagenet-v2")
    parser.add_argument("--num-labels", type=int, default=1000)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--no-compress", action="store_true", help="Skip compression and only evaluate existing checkpoints.")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation after optional compression.")
    return parser.parse_args()

def main():
    args = parse_args()
    lora_path_before_compact = args.lora_ckpt_path
    lora_path_after_compact = get_compressed_lora_path(lora_path_before_compact)

    if not args.no_compress:
        summary = compress_all_lora_q_by_energy(
            lora_ckpt_path=lora_path_before_compact,
            energy_threshold=args.energy_threshold,
        )
        print("Compression summary:", summary)

    if args.no_eval:
        print("Evaluation skipped.")
        return

    model = ViTForImageClassification.from_pretrained(args.model_path, num_labels=args.num_labels)
    processor = AutoImageProcessor.from_pretrained(args.model_path, use_fast=True)
    dataset_val_c, dataset_val_p = build_eval_datasets(args.data_type, processor, args)

    training_args = TrainingArguments(
        output_dir="results/svd_eval",
        per_device_eval_batch_size=args.eval_batch_size,
    )

    trainer_eval_clean = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset_val_c,
        compute_metrics=compute_clean_accuracy,
    )

    trainer_eval_poison = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset_val_p,
        compute_metrics=compute_asr,
    )

    results = {
        "acc_before": evaluate_lora(
            model,
            trainer_eval_clean,
            lora_path_before_compact,
            "ACC before compression",
        ),
        "acc_after": evaluate_lora(
            model,
            trainer_eval_clean,
            lora_path_after_compact,
            "ACC after compression",
        ),
        "asr_before": evaluate_lora(
            model,
            trainer_eval_poison,
            lora_path_before_compact,
            "ASR before compression",
        ),
        "asr_after": evaluate_lora(
            model,
            trainer_eval_poison,
            lora_path_after_compact,
            "ASR after compression",
        ),
    }
    print("All metrics:", results)


if __name__ == "__main__":
    main()
