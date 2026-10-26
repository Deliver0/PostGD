"""DINOv3 teacher-student distillation on the complete stage-one dataset.

One epoch is defined by one complete pass through ``data-dir``.  The default
run therefore requires 200,000 images and does not hold out a validation split:
all generated images participate in every training epoch.

The historical training objective is retained:

    L = MSE(proj(student(v1)), teacher(v1)) + beta * (1 - cos(student(v1), student(v2)))

The formal configuration is a DINOv3 ViT-L/16 teacher (1024-d) distilled into a
DINOv3 ViT-B/16 student (768-d) through a ``Linear(768, 1024)`` projection.
**Both models are initialised from pretrained weights.**  A randomly initialised
student collapses under this objective -- with only synthetic images and no
natural-image pretraining the student embeddings fall onto a single direction
(pairwise cosine ~0.99, effective rank 13/384) and every downstream task then
scores at chance.

``lr`` is tied to ``SCHEDULER_REFERENCE_BATCH_SIZE`` (64) images: the formal run
uses a global batch of 640 across 8 processes and steps the cosine schedule
``global_batch // 64`` times per optimizer step, so ``lr=1e-4`` keeps the same
meaning at any world size.  Layer-wise learning-rate decay (BEiT/MAE convention)
and an EMA copy of the student are both part of the formal recipe.

Models are loaded through timm so the local ``vit_base_patch16_dinov3`` weights
can be used without torch-hub's DINOv2 fallback.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# ``lr`` is defined for a batch of this many images.  Larger batches step the
# cosine schedule proportionally instead of scaling the learning rate.
SCHEDULER_REFERENCE_BATCH_SIZE = 64


@dataclass
class DistillConfig:
    data_dir: str
    output_dir: str = "./runs/distill"
    teacher_model: str = "vit_large_patch16_dinov3"
    student_model: str = "vit_base_patch16_dinov3"
    teacher_weights: Optional[str] = None
    student_weights: Optional[str] = None
    pretrained: bool = True
    img_size: int = 224
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_steps: int = 500
    beta: float = 0.1
    layer_decay: float = 0.75
    ema_decay: float = 0.999
    use_ema: bool = True
    aug_strength: float = 1.0
    device: str = "cuda"
    num_workers: int = 4
    expected_images: int = 200000
    validation_fraction: float = 0.0
    seed: int = 42
    resume: Optional[str] = None


def _image_files(data_dir: Path) -> list[Path]:
    files = []
    for path in data_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        name = path.name.lower()
        if "_synth" in name or name.startswith("synth_"):
            files.append(path)
    return sorted(files)


class SyntheticDataset(Dataset):
    def __init__(self, data_dir: str, transform):
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"data directory does not exist: {self.data_dir}")
        self.files = _image_files(self.data_dir)
        if not self.files:
            raise FileNotFoundError(f"no synth images found in {self.data_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            image = image.convert("RGB")
            view1 = self.transform(image)
            view2 = self.transform(image)
        return view1, view2


class StrongAugment:
    def __init__(self, img_size: int, strength: float):
        s = float(strength)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4 * s,
                contrast=0.4 * s,
                saturation=0.2 * s,
                hue=0.1 * s,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))],
                p=0.5,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __call__(self, image):
        return self.transform(image)


def _load_weights(model: nn.Module, path: Optional[str]) -> nn.Module:
    if not path:
        return model
    weight_path = Path(path)
    if weight_path.is_dir():
        candidates = (
            weight_path / "pytorch_model.bin",
            weight_path / "model.safetensors",
        )
        weight_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if weight_path is None:
            raise FileNotFoundError(
                f"no pytorch_model.bin or model.safetensors found in {path}"
            )
    if weight_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                "loading .safetensors weights requires safetensors"
            ) from exc
        checkpoint = load_file(str(weight_path), device="cpu")
    else:
        checkpoint = torch.load(weight_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state = checkpoint.get("state_dict", checkpoint)
        state = state.get("model", state) if isinstance(state, dict) else state
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise ValueError(f"unsupported checkpoint format: {path}")
    state = {
        key.removeprefix("module.").removeprefix("backbone."): value
        for key, value in state.items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"[Distill] loaded {weight_path}; "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    return model


def build_model(name: str, weights: Optional[str], pretrained: bool) -> nn.Module:
    if weights and Path(weights).is_dir():
        # DINOv3 checkpoints exported by timm use timm/EVA keys rather than
        # Hugging Face ViT keys.  Prefer timm when the requested architecture
        # is registered, then fall back to a local Hugging Face directory.
        try:
            import timm
            model = timm.create_model(name, pretrained=False, num_classes=0)
            print(f"[Distill] loading local timm model: {weights}")
            return _load_weights(model, weights)
        except (ImportError, RuntimeError, ValueError, KeyError):
            pass
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "loading a local Hugging Face model directory requires transformers"
            ) from exc
        print(f"[Distill] loading local Hugging Face model: {weights}")
        return AutoModel.from_pretrained(
            weights, local_files_only=True, use_safetensors=False
        )
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("distillation requires timm") from exc
    model = timm.create_model(
        name,
        pretrained=bool(pretrained and not weights),
        num_classes=0,
    )
    return _load_weights(model, weights)


def extract_feature(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Return a CLS/global embedding for timm and Hugging Face style outputs."""
    if hasattr(model, "forward_features"):
        output = model.forward_features(image)
    else:
        output = model(image)
    if isinstance(output, dict):
        for key in ("x_norm_clstoken", "cls_token", "last_hidden_state", "x"):
            if key in output:
                output = output[key]
                break
    if hasattr(output, "last_hidden_state"):
        output = output.last_hidden_state
    if not torch.is_tensor(output):
        raise TypeError(f"unsupported model output type: {type(output)!r}")
    if output.ndim == 3:
        return output[:, 0]
    if output.ndim != 2:
        raise ValueError(f"expected [B,D] or [B,T,D], got {tuple(output.shape)}")
    return output


def kd_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student, teacher)


def consistency_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = F.normalize(first, dim=-1)
    second = F.normalize(second, dim=-1)
    return 1.0 - (first * second).sum(dim=-1).mean()


def _autocast(device: str):
    if device.startswith("cuda"):
        return torch.cuda.amp.autocast(enabled=True)
    return torch.autocast(device_type="cpu", enabled=False)


# ---------------------------------------------------------------------
# Layer-wise LR decay (BEiT / MAE convention)
# ---------------------------------------------------------------------

def _layer_id(name: str, num_layers: int) -> int:
    if name.startswith("blocks."):
        try:
            return int(name.split(".")[1]) + 1
        except (IndexError, ValueError):
            return num_layers + 1
    if name.startswith(("patch_embed", "cls_token", "reg_token", "pos_embed",
                        "mask_token", "storage_tokens")):
        return 0
    return num_layers + 1


def _no_weight_decay(name: str, param: nn.Parameter) -> bool:
    if param.ndim <= 1:
        return True
    if name.endswith(".bias"):
        return True
    if "gamma" in name:  # LayerScale
        return True
    if name in ("cls_token", "reg_token", "pos_embed", "mask_token"):
        return True
    if "pos_embed" in name or "cls_token" in name or "reg_token" in name:
        return True
    return False


def build_param_groups(model: nn.Module, base_lr: float, weight_decay: float,
                       layer_decay: float, num_layers: int,
                       extra_max_layer: int = 0) -> list[dict]:
    """Return AdamW param groups with layer-wise lr scaling and a no-decay split.

    ``lr_scale = layer_decay ** (max_layer - layer_id)`` so the patch embedding
    gets the smallest learning rate and the final norm the largest.
    """
    max_layer = num_layers + 1 + extra_max_layer
    groups: dict[tuple[int, float], dict] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lid = _layer_id(name, num_layers)
        if lid > num_layers + 1:
            lid = num_layers + 1
        scale = layer_decay ** (max_layer - lid)
        wd = 0.0 if _no_weight_decay(name, param) else weight_decay
        key = (lid, wd)
        groups.setdefault(key, {"params": [], "lr": base_lr * scale,
                                "weight_decay": wd, "layer_id": lid,
                                "lr_scale": scale})
        groups[key]["params"].append(param)
    return [groups[k] for k in sorted(groups, key=lambda k: (k[0], k[1]))]


# ---------------------------------------------------------------------
# EMA student
# ---------------------------------------------------------------------

class EMAStudent(nn.Module):
    """Exponential moving average of the student, with a warm-up ramp.

    ``decay_t = min(decay, (1 + step) / (10 + step))`` so the EMA does not lag
    the freshly-initialised student at the very beginning of training.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        super().__init__()
        import copy
        self.module = copy.deepcopy(model)
        for param in self.module.parameters():
            param.requires_grad = False
        self.decay = float(decay)
        self.step = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step += 1
        d = self.current_decay()
        for ema_param, param in zip(self.module.parameters(), model.parameters()):
            ema_param.mul_(d).add_(param.detach(), alpha=1.0 - d)
        for ema_buf, buf in zip(self.module.buffers(), model.buffers()):
            ema_buf.copy_(buf)

    def current_decay(self) -> float:
        return min(self.decay, (1.0 + self.step) / (10.0 + self.step))


def train_one_epoch(student, teacher, projector, loader, optimizer, scheduler,
                    scaler, cfg: DistillConfig, epoch: int, ema=None) -> dict:
    student.train()
    projector.train()
    teacher.eval()
    # Keep ``lr`` tied to SCHEDULER_REFERENCE_BATCH_SIZE: a larger batch takes
    # proportionally more scheduler steps per optimizer step.
    scheduler_steps = max(1, int(cfg.batch_size) // SCHEDULER_REFERENCE_BATCH_SIZE)
    totals = {"loss": 0.0, "kd": 0.0, "cons": 0.0, "samples": 0}
    progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")
    for view1, view2 in progress:
        view1 = view1.to(cfg.device, non_blocking=True)
        view2 = view2.to(cfg.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(cfg.device):
            student_first = projector(extract_feature(student, view1))
            student_second = projector(extract_feature(student, view2))
            with torch.no_grad():
                teacher_feature = extract_feature(teacher, view1)
            loss_kd = kd_loss(student_first, teacher_feature)
            loss_cons = consistency_loss(student_first, student_second)
            loss = loss_kd + cfg.beta * loss_cons
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if ema is not None:
            ema.update(student)
        if scheduler is not None:
            for _ in range(scheduler_steps):
                scheduler.step()
        batch_size = int(view1.shape[0])
        totals["loss"] += loss.item() * batch_size
        totals["kd"] += loss_kd.item() * batch_size
        totals["cons"] += loss_cons.item() * batch_size
        totals["samples"] += batch_size
        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )
    if totals["samples"] != len(loader.dataset):
        raise RuntimeError(
            f"epoch consumed {totals['samples']} samples, "
            f"expected {len(loader.dataset)}"
        )
    return {
        key: value / totals["samples"]
        for key, value in totals.items()
        if key != "samples"
    } | {"samples": totals["samples"], "batches": len(loader)}


def _make_scheduler(optimizer, total_steps: int, warmup_steps: int):
    warmup_steps = min(max(0, int(warmup_steps)), max(0, total_steps - 1))

    def scale(step):
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / remaining))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DINOv3 distillation on stage-one images")
    parser.add_argument("--data-dir", "--data_dir", required=True)
    parser.add_argument("--output-dir", "--output_dir", default="./runs/distill")
    parser.add_argument("--teacher-model", "--teacher",
                        default="vit_large_patch16_dinov3")
    parser.add_argument("--student-model", "--student",
                        default="vit_base_patch16_dinov3")
    parser.add_argument("--teacher-weights")
    parser.add_argument("--student-weights")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="initialise the student randomly; not recommended, "
                             "the student collapses under this objective")
    parser.add_argument("--img-size", "--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=64,
                        help="images per optimizer step; lr is tied to "
                             f"{SCHEDULER_REFERENCE_BATCH_SIZE}")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", "--warmup_steps", type=int, default=500)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--layer-decay", "--layer_decay", type=float, default=0.75,
                        help="BEiT/MAE layer-wise lr decay")
    parser.add_argument("--ema-decay", "--ema_decay", type=float, default=0.999)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--aug-strength", "--aug_strength", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", "--num_workers", type=int, default=4)
    parser.add_argument("--expected-images", "--expected_images", type=int, default=200000)
    parser.add_argument("--validation-fraction", "--validation_fraction",
                        type=float, default=0.0,
                        help="0 keeps all images in every training epoch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume")
    return parser


def run(cfg: DistillConfig) -> None:
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if cfg.expected_images < 0:
        raise ValueError("expected_images must be >= 0")
    if not 0 <= cfg.validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = SyntheticDataset(
        cfg.data_dir, StrongAugment(cfg.img_size, cfg.aug_strength)
    )
    if cfg.expected_images and len(dataset) != cfg.expected_images:
        raise RuntimeError(
            f"expected {cfg.expected_images} images, found {len(dataset)} in {cfg.data_dir}"
        )

    train_dataset = dataset
    val_dataset = None
    if cfg.validation_fraction > 0:
        n_val = max(1, int(len(dataset) * cfg.validation_fraction))
        n_train = len(dataset) - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(cfg.seed),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.startswith("cuda"),
        persistent_workers=cfg.num_workers > 0,
    )
    print(
        f"[Distill] {len(train_dataset)} images/epoch, "
        f"{len(train_loader)} batches/epoch, batch_size={cfg.batch_size}"
    )

    teacher = build_model(cfg.teacher_model, cfg.teacher_weights, cfg.pretrained)
    student = build_model(cfg.student_model, cfg.student_weights, cfg.pretrained)
    teacher.to(cfg.device).eval()
    student.to(cfg.device).train()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    teacher_dim = int(getattr(teacher, "num_features", 0))
    student_dim = int(getattr(student, "num_features", 0))
    if not teacher_dim or not student_dim:
        with torch.no_grad():
            probe = torch.zeros(1, 3, cfg.img_size, cfg.img_size, device=cfg.device)
            teacher_dim = int(extract_feature(teacher, probe).shape[-1])
            student_dim = int(extract_feature(student, probe).shape[-1])
    projector = (nn.Identity() if student_dim == teacher_dim
                 else nn.Linear(student_dim, teacher_dim)).to(cfg.device)

    num_layers = len(getattr(student, "blocks", []))
    param_groups = build_param_groups(
        student, cfg.lr, cfg.weight_decay, cfg.layer_decay, num_layers
    )
    if not isinstance(projector, nn.Identity):
        param_groups += build_param_groups(
            projector, cfg.lr, cfg.weight_decay, cfg.layer_decay, num_layers,
        )
    print(
        f"[Distill] teacher_dim={teacher_dim} student_dim={student_dim} "
        f"layer_decay={cfg.layer_decay} num_layers={num_layers} "
        f"param_groups={len(param_groups)}"
    )

    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler_steps_per_epoch = -(-len(train_dataset) // SCHEDULER_REFERENCE_BATCH_SIZE)
    total_steps = max(1, cfg.epochs * scheduler_steps_per_epoch)
    scheduler = _make_scheduler(optimizer, total_steps, cfg.warmup_steps)
    scaler = torch.cuda.amp.GradScaler() if cfg.device.startswith("cuda") else None

    ema = EMAStudent(student, cfg.ema_decay).to(cfg.device) if cfg.use_ema else None
    if ema is not None:
        print(f"[Distill] EMA enabled, decay={cfg.ema_decay}")

    start_epoch = 0
    if cfg.resume:
        checkpoint = torch.load(cfg.resume, map_location=cfg.device)
        student.load_state_dict(checkpoint["student"])
        projector.load_state_dict(checkpoint["projector"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if ema is not None and "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"[Distill] resumed at epoch {start_epoch}")

    (output_dir / "config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics_path = output_dir / "epoch_metrics.jsonl"
    start_time = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        train_metrics = train_one_epoch(
            student, teacher, projector, train_loader, optimizer, scheduler,
            scaler, cfg, epoch, ema=ema
        )
        record = {"epoch": epoch + 1, "train": train_metrics}
        if ema is not None:
            record["ema_decay"] = ema.current_decay()
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        checkpoint = {
            "epoch": epoch,
            "student": student.state_dict(),
            "projector": projector.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_metrics": train_metrics,
            "config": asdict(cfg),
        }
        if ema is not None:
            checkpoint["ema"] = ema.state_dict()
        torch.save(checkpoint, output_dir / "last_checkpoint.pt")
        print(
            f"Epoch {epoch + 1}/{cfg.epochs} | "
            f"samples={train_metrics['samples']} "
            f"loss={train_metrics['loss']:.5f} "
            f"kd={train_metrics['kd']:.5f} "
            f"cons={train_metrics['cons']:.5f} | "
            f"time_min={(time.time() - start_time) / 60:.1f}"
        )

    torch.save(student.state_dict(), output_dir / "student_final.pth")
    torch.save(projector.state_dict(), output_dir / "student_to_teacher_projection.pth")
    if ema is not None:
        torch.save(ema.module.state_dict(), output_dir / "student_ema_final.pth")
    print(f"[Distill] complete: {output_dir}")


def main() -> None:
    args = _parser().parse_args()
    run(DistillConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        teacher_weights=args.teacher_weights,
        student_weights=args.student_weights,
        pretrained=not args.no_pretrained,
        img_size=args.img_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        beta=args.beta,
        layer_decay=args.layer_decay,
        ema_decay=args.ema_decay,
        use_ema=not args.no_ema,
        aug_strength=args.aug_strength,
        device=args.device,
        num_workers=args.num_workers,
        expected_images=args.expected_images,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        resume=args.resume,
    ))


if __name__ == "__main__":
    main()
