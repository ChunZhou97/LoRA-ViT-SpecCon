from pathlib import Path
from transformers import TrainerCallback
import torch
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# -------------------------------------------------------
# Define two metric functions
# -------------------------------------------------------
def compute_clean_accuracy(eval_pred):
    logits, labels = eval_pred
    preds = torch.tensor(logits).argmax(-1)
    labels = torch.tensor(labels)
    acc = (preds == labels).float().mean().item()
    return {"accuracy": acc}

def compute_asr(eval_pred):
    target_label = 0
    logits, labels = eval_pred
    preds = torch.tensor(logits).argmax(-1)
    asr = (preds == target_label).float().mean().item()
    return {"attack_success_rate": asr}


# -------------------------------------------------------
# Callback: automatically evaluate clean and poisoned sets every N epochs
# -------------------------------------------------------
class CleanPoisonEvalCallback(TrainerCallback):
    def __init__(self, clean_dataset, poison_dataset, eval_every=5, log_dir="log_c"):
        self.clean_dataset = clean_dataset
        self.poison_dataset = poison_dataset
        self.eval_every = eval_every
        self.log_dir = log_dir
        self.trainer = None

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)

        if epoch % self.eval_every == 0:
            if self.trainer is None:
                raise RuntimeError("Trainer not initialized in callback.")

            orig_compute_metrics = self.trainer.compute_metrics

            # ------- Clean -------
            self.trainer.compute_metrics = compute_clean_accuracy
            clean_metrics = self.trainer.evaluate(
                eval_dataset=self.clean_dataset,
                metric_key_prefix=f"clean_epoch{epoch}",
            )
            print(f"[Epoch {epoch}] Clean Accuracy:", clean_metrics)
            # Write to file
            with open(f"{self.log_dir}/clean_eval_log.txt", "a") as f:
                f.write(f"[Epoch {epoch}] {clean_metrics}\n")

            # ------- Poison -------
            self.trainer.compute_metrics = compute_asr
            poison_metrics = self.trainer.evaluate(
                eval_dataset=self.poison_dataset,
                metric_key_prefix=f"poison_epoch{epoch}",
            )
            print(f"[Epoch {epoch}] Poison ASR:", poison_metrics)
            # Write to file
            with open(f"{self.log_dir}/poison_eval_log.txt", "a") as f:
                f.write(f"[Epoch {epoch}] {poison_metrics}\n")

            # Restore
            self.trainer.compute_metrics = orig_compute_metrics

            control.should_save = True
            control.should_log = True

        return control


class ImageNet1kTrainDataset(Dataset):
    """
    Read the official ImageNet-1k validation set format:
    root/xxx.JPEG  +  val.txt  ->  (image, label)
    """
    def __init__(self, root: str, label_file: str, label_map_file: str, transform=None):
        self.root = Path(root)
        with open(label_map_file, 'r',
                  encoding='utf-8') as f:
            self.label_dic = json.load(f)

        self.transform = transform
        with open(label_file) as f:
            lines = [l.strip().split() for l in f if l.strip()]
        self.samples_path_int = [(self.root / x[0], int(self.label_dic[x[1]])) for x in lines]


        self.samples_name_int = [(x[0], x[1]) for x in lines]

    def __len__(self):
        return len(self.samples_path_int)

    def __getitem__(self, idx):
        path, label = self.samples_path_int[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img, return_tensors='pt')['pixel_values'].squeeze(0)
        return {"pixel_values":img, "labels": label}


class ImageNet1kValDataset(Dataset):
    """
    Read the official ImageNet-1k validation set format:
    root/xxx.JPEG  +  val.txt  ->  (image, label)
    """
    def __init__(self, root: str, label_file: str, label_map_file: str, transform=None):
        self.root = Path(root)
        with open(label_map_file, 'r',
                  encoding='utf-8') as f:
            self.label_dic = json.load(f)

        self.transform = transform
        with open(label_file) as f:
            lines = [l.strip().split() for l in f if l.strip()]
        self.samples_path_int = [(self.root / x[0], int(self.label_dic[x[1]])) for x in lines]

        # Use a subset of the validation set
        self.samples_path_int = self.samples_path_int[:10000]

        self.samples_name_int = [(x[0], x[1]) for x in lines]

    def __len__(self):
        return len(self.samples_path_int)

    def __getitem__(self, idx):
        path, label = self.samples_path_int[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img, return_tensors='pt')['pixel_values'].squeeze(0)
        return {"pixel_values":img, "labels": label}


class ImageNetv2TrainDataset(Dataset):
    """
    Read the official ImageNet-1k validation set format:
    root/xxx.JPEG  +  val.txt  ->  (image, label)
    """
    def __init__(self, root: str, label_file: str, transform=None):
        self.root = Path(root)

        self.transform = transform
        with open(label_file) as f:
            lines = [l.strip().split() for l in f if l.strip()]
        self.samples_path_int = [(self.root / x[0], int(x[1])) for x in lines]

    def __len__(self):
        return len(self.samples_path_int)

    def __getitem__(self, idx):
        path, label = self.samples_path_int[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img, return_tensors='pt')['pixel_values'].squeeze(0)
        return {"pixel_values":img, "labels": label}


class ImageNetv2ValDataset(Dataset):
    """
    Read the official ImageNet-1k validation set format:
    root/xxx.JPEG  +  val.txt  ->  (image, label)
    """
    def __init__(self, root: str, label_file: str, transform=None):
        self.root = Path(root)

        self.transform = transform
        with open(label_file) as f:
            lines = [l.strip().split() for l in f if l.strip()]
        self.samples_path_int = [(self.root / x[0], int(x[1])) for x in lines]

    def __len__(self):
        return len(self.samples_path_int)

    def __getitem__(self, idx):
        path, label = self.samples_path_int[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img, return_tensors='pt')['pixel_values'].squeeze(0)
        return {"pixel_values":img, "labels": label}


if __name__ == '__main__':


    print("ok")
