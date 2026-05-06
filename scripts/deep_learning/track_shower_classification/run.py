try:
    from IPython import get_ipython
    if 'IPKernelApp' in get_ipython().config:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
        import matplotlib
        matplotlib.use('Agg')
        
except Exception:
    from tqdm import tqdm
    import matplotlib
    matplotlib.use('Agg')

from torch.utils.data import DataLoader, random_split, Subset
from dataset import *
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from training import *
from network import *


def confusion_matrix_figure(preds, labels, class_names, title):
    """
    Returns a figure of a normalised confusion matrix.
    Normalised by true class (rows sum to 1).
    """
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    cm_norm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}\n({cm[i,j]})",
                    ha='center', va='center', fontsize=8,
                    color='white' if cm_norm[i, j] > thresh else 'black')

    fig.tight_layout()
    return fig


def get_gpu_memory_stats(device):
    """
    Returns a dict of current GPU memory usage in MB for the given device.
    Returns empty dict if CUDA is not available or device is CPU.
    """
    if not torch.cuda.is_available() or device.type == 'cpu':
        return {}

    allocated  = torch.cuda.memory_allocated(device)  / 1024**2
    reserved   = torch.cuda.memory_reserved(device)   / 1024**2
    max_alloc  = torch.cuda.max_memory_allocated(device) / 1024**2

    return {
        "allocated_MB":  allocated,   # memory currently held by tensors
        "reserved_MB":   reserved,    # memory held by pytorch allocator (includes free blocks)
        "peak_alloc_MB": max_alloc,   # high-water mark since last reset
    }


def save_checkpoint(state, filename):
    torch.save(state, filename)


def load_checkpoint(path, model, optimizer, scheduler, device):
    """
    Loads a checkpoint saved by save_checkpoint.
    Restores model, optimizer, and scheduler state in-place.
    Returns the epoch to resume from and the best validation accuracy seen so far.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch    = checkpoint["epoch"] + 1
    best_val_acc  = checkpoint["best_val_acc"]
    print(f"Resumed from {path} — starting at epoch {start_epoch}, best val acc {best_val_acc:.4f}")
    return start_epoch, best_val_acc


if __name__ == "__main__":
    CLASS_NAMES = ["mip", "hip", "shower", "lowe"]
    dataset = LArTPCSequenceDataset("data.h5")
    #dataset = Subset(dataset, range(0, 640))
    train_frac = 0.6
    n_total = len(dataset)
    n_train = int(train_frac * n_total)
    n_val = n_total - n_train

    # --- Resume ---
    # Set resume_checkpoint to a path string to resume, or None to start fresh.
    # e.g. resume_checkpoint = "checkpoints/best_model.pt"
    resume_checkpoint = None
    
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fn_pad, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn_pad, pin_memory=True)

    device = torch.device("cuda:0")
    num_classes = 4    # mip, hip, shower, lowe
    class_weights = compute_class_weights(train_loader, num_classes, device=device)

    total_epochs = 175
    model = LArTPCTransformer(
        input_dim=11,        # [x_rel, z_rel, x_abs, z_abs, width, adc, r, cosθ, sinθ, wire_pitch, wire_angle]
        embed_dim=128, num_heads=8, ff_dim=256, num_layers=4,
        num_classes=num_classes,
        dropout=0.1)
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=total_epochs, pct_start=0.1, anneal_strategy='cos')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=total_epochs, T_mult=1, eta_min=1e-6)

    from torch.utils.tensorboard import SummaryWriter
    
    writer = SummaryWriter(log_dir="runs/lar_tpc_experiment")
    global_step = 0
    best_val_acc = float(0)
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_epoch = 0

    if resume_checkpoint is not None:
        start_epoch, best_val_acc = load_checkpoint(resume_checkpoint, model, optimizer, scheduler, device)

    run_epochs = 100
    finish_epoch = min(start_epoch + run_epochs, total_epochs)

    for epoch in tqdm(range(start_epoch, finish_epoch), "Training"):
        train_loss, train_acc, train_preds, train_labels = train_one_epoch(model, train_loader, optimizer, criterion, device, writer=writer, epoch=epoch)
        val_loss, val_acc, val_preds, val_labels = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step(epoch)
        torch.cuda.reset_peak_memory_stats(device)
    
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
        if writer is not None:
            writer.add_scalar("Loss/Train_epoch", train_loss, epoch)
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            writer.add_scalar("Accuracy/Train", train_acc, epoch)
            writer.add_scalar("Accuracy/Validation", val_acc, epoch)

            gpu_stats = get_gpu_memory_stats(device)
            for name, value in gpu_stats.items():
                writer.add_scalar(f"GPU/{name}", value, epoch)
    
            train_fig = confusion_matrix_figure(
                train_preds, train_labels, CLASS_NAMES,
                title=f"Train confusion — epoch {epoch+1}"
            )
            val_fig = confusion_matrix_figure(
                val_preds, val_labels, CLASS_NAMES,
                title=f"Val confusion — epoch {epoch+1}"
            )
            writer.add_figure("Confusion/Train", train_fig, epoch)
            writer.add_figure("Confusion/Val",   val_fig,   epoch)
            plt.close(train_fig)
            plt.close(val_fig)
    
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
        }

        # --- Save best model ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint["best_val_acc"] = best_val_acc
            epoch_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt")
            save_checkpoint(checkpoint, epoch_path)
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            save_checkpoint(checkpoint, best_path)

