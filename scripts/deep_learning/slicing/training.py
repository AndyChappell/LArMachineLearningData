import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm


def train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=None, max_grad_norm=1.0):
    model.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc="[Train]", unit="batch")

    for batch in pbar:
        hits = batch["hits"].to(device)                     # (B, N, F)
        slice_labels = batch["slice_labels"].to(device)     # (B, N)
        cp_labels = batch["cp_labels"].to(device)           # (B, N)

        optimizer.zero_grad()

        # ----------------------------
        # Forward (with AMP if enabled)
        # ----------------------------
        if scaler is not None:
            with autocast():
                pred = model(hits)
                loss = criterion(pred, slice_labels, cp_labels)
        else:
            pred = model(hits)
            loss = criterion(pred, slice_labels, cp_labels)

        # ----------------------------
        # Backward
        # ----------------------------
        if scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    return running_loss / len(train_loader)


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0

    pbar = tqdm(val_loader, desc="[Val]", unit="batch")

    for batch in pbar:
        hits = batch["hits"].to(device)
        slice_labels = batch["slice_labels"].to(device)
        cp_labels = batch["cp_labels"].to(device)

        pred = model(hits)
        loss = criterion(pred, slice_labels, cp_labels)

        running_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    return running_loss / len(val_loader)


def save_checkpoint(model, optimizer, epoch, path):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(ckpt, path)
    print(f"✓ Saved checkpoint to {path}")


def load_or_initialize_model(model_class, model_kwargs, checkpoint_path=None, device="cuda"):
    """
    model_class:   the class object, e.g. MyTransformer
    model_kwargs:  dict of kwargs to instantiate the model
    checkpoint_path: path to a saved checkpoint .pt/.pth
    """
    # Instantiate model
    model = model_class(**model_kwargs).to(device)

    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print("✓ Model weights restored.")
    else:
        print("✓ Initializing new model.")

    return model