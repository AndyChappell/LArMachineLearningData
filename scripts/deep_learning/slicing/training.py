import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from tqdm.auto import tqdm


def train_one_epoch(
        epoch, model, train_loader, optimizer, criterion,
        device, writer=None, scaler=None, max_grad_norm=1.0):

    model.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f"[Train {epoch}]", unit="batch")

    for batch_idx, batch in enumerate(pbar):
        hits = batch["hits"].to(device)               # (B,N,F)
        slice_labels = batch["slice_labels"].to(device)
        cp_labels = batch["cp_labels"].to(device)     # must be {0,1}

        is_cp = cp_labels                             # <-- no conversion needed

        optimizer.zero_grad()

        # -----------------------
        # Forward
        # -----------------------
        if scaler is not None:
            with autocast(device_type=device.type):
                pred = model(hits)
                loss, extras = criterion(pred, slice_labels, is_cp)
        else:
            pred = model(hits)
            loss, extras = criterion(pred, slice_labels, is_cp)

        # -----------------------
        # Backward
        # -----------------------
        if scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # -----------------------
        # TensorBoard logging
        # -----------------------
        if writer is not None and batch_idx % 10 == 0:
            global_step = (epoch - 1) * len(train_loader) + batch_idx

            writer.add_scalar("loss/total", loss.item(), global_step)
            writer.add_scalar("loss/beta", extras["beta_loss"].item(), global_step)
            writer.add_scalar("loss/attr", extras["attr_loss"].item(), global_step)
            writer.add_scalar("loss/repl", extras["repl_loss"].item(), global_step)

            with torch.no_grad():
                beta_prob = torch.sigmoid(pred["beta"]).squeeze(-1)   # (B,N)
                beta_flat = beta_prob.cpu().flatten()

                writer.add_histogram("beta/all", beta_flat, global_step)
                writer.add_scalar("beta/mean", beta_flat.mean().item(), global_step)

                cp_mask = (cp_labels.flatten() == 1).cpu()
                noncp_mask = ~cp_mask

                if cp_mask.any():
                    writer.add_histogram("beta/cp", beta_flat[cp_mask], global_step)
                if noncp_mask.any():
                    writer.add_histogram("beta/noncp", beta_flat[noncp_mask], global_step)

                pred_cp = (beta_flat > 0.8).sum().item()
                true_cp = cp_mask.sum().item()

                writer.add_scalar("cp/pred_count", pred_cp, global_step)
                writer.add_scalar("cp/true_count", true_cp, global_step)

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
        loss, extras = criterion(pred, slice_labels, cp_labels)

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