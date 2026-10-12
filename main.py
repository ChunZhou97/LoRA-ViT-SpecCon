import argparse
import os
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoImageProcessor, Trainer, TrainingArguments, ViTForImageClassification

from utils import (
    CleanPoisonEvalCallback,
    ImageNet1kTrainDataset,
    ImageNet1kValDataset,
    ImageNetv2TrainDataset,
    ImageNetv2ValDataset,
    compute_clean_accuracy,
)


MODEL_PATHS = {
    "base": "models/ViT-base-16-224/",
    "large": "models/ViT-large-16-384/",
}

ALGORITHMS = ("ft", "bitfit", "adapter-h", "adapter-l", "lora-l", "lora-o")


@dataclass(frozen=True)
class AlgorithmConfig:
    output_name: str
    log_name: str
    train_batch_size: int = 32
    eval_batch_size: int = 32
    eval_every: int = 2
    lora_rank: int | None = None
    lora_alpha: int | None = None
    adapter_dim: int = 2
    dataloader_num_workers: int = 0


ALGORITHM_CONFIGS = {
    "ft": AlgorithmConfig(output_name="ckpt_ft", log_name="log_ft"),
    "bitfit": AlgorithmConfig(output_name="ckpt_bitfit", log_name="log_bitfit"),
    "adapter-h": AlgorithmConfig(output_name="ckpt_adapter-h", log_name="log_adapter-h"),
    "adapter-l": AlgorithmConfig(output_name="ckpt_adapter-l", log_name="log_adapter-l"),
    "lora-l": AlgorithmConfig(
        output_name="ckpt_lora-l",
        log_name="log_lora-l",
        lora_rank=2,
        lora_alpha=4,
    ),
    "lora-o": AlgorithmConfig(
        output_name="ckpt_lora-o",
        log_name="logs_lora-o",
        lora_rank=16,
        lora_alpha=32
    ),
}


class SimpleAdapter(nn.Module):
    def __init__(self, hidden_dim: int, adapter_dim: int = 16):
        super().__init__()
        self.down = nn.Linear(hidden_dim, adapter_dim, bias=False)
        self.act = nn.ReLU()
        self.up = nn.Linear(adapter_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.act(self.down(x)))


class AdapterLayerL(nn.Module):
    def __init__(self, base_layer: nn.Module, hidden_dim: int, adapter_dim: int):
        super().__init__()
        self.base_layer = base_layer
        self.adapter = SimpleAdapter(hidden_dim, adapter_dim)

    def forward(self, hidden_states, *args, **kwargs):
        outputs = self.base_layer(hidden_states, *args, **kwargs)
        if isinstance(outputs, torch.Tensor):
            return outputs + self.adapter(outputs)
        if isinstance(outputs, tuple):
            hidden_states = outputs[0] + self.adapter(outputs[0])
            return (hidden_states,) + outputs[1:]
        hidden_states = outputs.hidden_states + self.adapter(outputs.hidden_states)
        outputs.hidden_states = hidden_states
        return outputs


class AdapterViTLayerH(nn.Module):
    def __init__(self, base_layer: nn.Module, hidden_dim: int, adapter_dim: int):
        super().__init__()
        self.base_layer = base_layer
        self.adapter = SimpleAdapter(hidden_dim, adapter_dim)

    def forward(self, hidden_states, head_mask=None):
        attention_output = self.base_layer.attention(
            self.base_layer.layernorm_before(hidden_states),
            head_mask,
        )
        hidden_states = hidden_states + attention_output
        ln_states = self.base_layer.layernorm_after(hidden_states)
        ln_states = ln_states + self.adapter(ln_states)
        intermediate_output = self.base_layer.intermediate(ln_states)
        return self.base_layer.output(intermediate_output, hidden_states)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified ViT training entry for FT, BitFit, Adapter, and LoRA algorithms."
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        nargs="+",
        choices=ALGORITHMS,
        # default=["adapter-h"],
        default=["adapter-h", "adapter-l", "lora-o", "lora-l", "bitfit", "ft"],
        help="Algorithms to run. Use multiple values to run them sequentially.",

    )
    parser.add_argument(
        "--model-type",
        nargs="+",
        choices=MODEL_PATHS.keys(),
        default=["base", "large"],
        help="ViT model sizes to run.",
    )
    parser.add_argument(
        "--data-type",
        nargs="+",
        choices=("1k", "v2"),
        default=["1k", "v2"],
        help="Dataset variants to run.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--adapter-dim", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--target-modules", nargs="+", default=["query"])
    parser.add_argument("--dataloader-num-workers", type=int, default=None)
    parser.add_argument("--output-root", default="results", help="Root directory for experiment outputs.")
    parser.add_argument("--model-base-path", default=MODEL_PATHS["base"])
    parser.add_argument("--model-large-path", default=MODEL_PATHS["large"])
    parser.add_argument("--imagenet-1k-root", default="datasets_processed/imagenet-1k")
    parser.add_argument("--imagenet-v2-root", default="datasets_processed/imagenet-v2")
    parser.add_argument("--num-labels", type=int, default=1000)
    parser.add_argument("--eval-before-train", action="store_true")
    parser.add_argument("--no-train", action="store_true", help="Only build the trainer and run optional evaluation.")
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def get_model_name(model_type: str, args: argparse.Namespace) -> str:
    if model_type == "base":
        return args.model_base_path
    if model_type == "large":
        return args.model_large_path
    raise ValueError(f"Unsupported model type: {model_type}")


def load_datasets(data_type: str, processor, args: argparse.Namespace):
    if data_type == "1k":
        root = args.imagenet_1k_root
        dataset_train_c = ImageNet1kTrainDataset(
            root=os.path.join(root, "train_c"),
            label_file=os.path.join(root, "ImageNet10_c_train_label.txt"),
            label_map_file=os.path.join(root, "ImageNet_label.json"),
            transform=processor,
        )
        dataset_train_p = ImageNet1kTrainDataset(
            root=os.path.join(root, "train_p"),
            label_file=os.path.join(root, "ImageNet10_p_train_label.txt"),
            label_map_file=os.path.join(root, "ImageNet_label.json"),
            transform=processor,
        )
        dataset_val_c = ImageNet1kValDataset(
            root=os.path.join("datasets_raw/imagenet-1k", "val"),
            label_file=os.path.join("datasets_raw/imagenet-1k", "ImageNet_val_label.txt"),
            label_map_file=os.path.join(root, "ImageNet_label.json"),
            transform=processor,
        )
        dataset_val_p = ImageNet1kValDataset(
            root=os.path.join(root, "val_p"),
            label_file=os.path.join(root, "ImageNet_val_label.txt"),
            label_map_file=os.path.join(root, "ImageNet_label.json"),
            transform=processor,
        )
        return dataset_train_c, dataset_train_p, dataset_val_c, dataset_val_p

    if data_type == "v2":
        root = args.imagenet_v2_root
        dataset_train_c = ImageNetv2TrainDataset(
            root=os.path.join(root, "ImageNet_v2_train_c"),
            label_file=os.path.join(root, "train_c_label.txt"),
            transform=processor,
        )
        dataset_train_p = ImageNetv2TrainDataset(
            root=os.path.join(root, "ImageNet_v2_train_p"),
            label_file=os.path.join(root, "train_p_label.txt"),
            transform=processor,
        )
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
        return dataset_train_c, dataset_train_p, dataset_val_c, dataset_val_p

    raise ValueError(f"Unsupported data type: {data_type}")


def add_low_adapter_to_vit(model: ViTForImageClassification, adapter_dim: int) -> None:
    hidden_dim = model.config.hidden_size

    for index, layer in enumerate(model.vit.encoder.layer):
        model.vit.encoder.layer[index] = AdapterLayerL(layer, hidden_dim, adapter_dim)


def add_high_adapter_to_vit(model: ViTForImageClassification, adapter_dim: int) -> None:
    hidden_dim = model.config.hidden_size

    for index, layer in enumerate(model.vit.encoder.layer):
        model.vit.encoder.layer[index] = AdapterViTLayerH(layer, hidden_dim, adapter_dim)


def freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def configure_algorithm(model: ViTForImageClassification, algorithm: str, config: AlgorithmConfig, args: argparse.Namespace):
    total_params = sum(p.numel() for p in model.parameters())
    print("total params original: ", total_params)

    if algorithm == "ft":
        freeze_all(model)
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif algorithm == "bitfit":
        freeze_all(model)
        for name, param in model.named_parameters():
            if name.endswith(".bias"):
                param.requires_grad = True
    elif algorithm in {"adapter-h", "adapter-l"}:
        adapter_dim = args.adapter_dim or config.adapter_dim
        if algorithm == "adapter-h":
            add_high_adapter_to_vit(model, adapter_dim)
        else:
            add_low_adapter_to_vit(model, adapter_dim)
        freeze_all(model)
        for layer in model.vit.encoder.layer:
            for param in layer.adapter.parameters():
                param.requires_grad = True
    elif algorithm in {"lora-l", "lora-o"}:
        rank = args.lora_rank or config.lora_rank
        alpha = args.lora_alpha or config.lora_alpha
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        model = get_peft_model(model, lora_config)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable_params:,}")
    print(f"Total params:     {total_params:,}")
    print(f"Trainable ratio:  {100 * trainable_params / total_params:.2f}%")
    return model


def build_output_dirs(algorithm: str, model_type: str, data_type: str, args: argparse.Namespace) -> tuple[str, str]:
    config = ALGORITHM_CONFIGS[algorithm]
    exp_dir = os.path.join(args.output_root, f"exp_{model_type}_{data_type}")
    output_dir = os.path.join(exp_dir, config.output_name)
    log_dir = os.path.join(exp_dir, config.log_name)
    os.makedirs(log_dir, exist_ok=True)
    return output_dir, log_dir


def run_one(algorithm: str, model_type: str, data_type: str, args: argparse.Namespace) -> None:
    config = ALGORITHM_CONFIGS[algorithm]
    print("+++++++++++++++++++++++++++Training Begin!!+++++++++++++++++++++++++++", model_type, data_type, algorithm)

    output_dir, log_dir = build_output_dirs(algorithm, model_type, data_type, args)
    model_name = get_model_name(model_type, args)
    model = ViTForImageClassification.from_pretrained(model_name, num_labels=args.num_labels)
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = configure_algorithm(model, algorithm, config, args)

    _, dataset_train_p, dataset_val_c, dataset_val_p = load_datasets(data_type, processor, args)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.train_batch_size or config.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size or config.eval_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        remove_unused_columns=False,
        logging_steps=args.logging_steps,
        dataloader_num_workers=(
            args.dataloader_num_workers
            if args.dataloader_num_workers is not None
            else config.dataloader_num_workers
        ),
    )

    callback = CleanPoisonEvalCallback(
        dataset_val_c,
        dataset_val_p,
        eval_every=args.eval_every or config.eval_every,
        log_dir=log_dir,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train_p,
        eval_dataset=dataset_val_c,
        compute_metrics=compute_clean_accuracy,
        callbacks=[callback],
    )
    callback.trainer = trainer

    if args.eval_before_train:
        clean_metrics = trainer.evaluate(eval_dataset=dataset_val_c)
        print("[Original Clean Accuracy:", clean_metrics)

    if not args.no_train:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


def iter_runs(args: argparse.Namespace) -> Iterable[tuple[str, str, str]]:
    for algorithm in args.algorithm:
        for model_type in args.model_type:
            for data_type in args.data_type:
                yield algorithm, model_type, data_type


def main() -> None:
    args = parse_args()
    for algorithm, model_type, data_type in iter_runs(args):
        run_one(algorithm, model_type, data_type, args)


if __name__ == "__main__":
    main()
