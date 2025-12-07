import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from tqdm.auto import tqdm


def train_one_epoch(epoch, model, train_loader, optimizer, criterion, device, writer=None, scaler=None, max_grad_norm=1.0):
    model.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc="[Train]", unit="batch")

    for batch_idx, batch in enumerate(pbar):
        hits = batch["hits"].to(device)                     # (B, N, F)
        slice_labels = batch["slice_labels"].to(device)     # (B, N)
        cp_labels = batch["cp_labels"].to(device)           # (B, N)

        optimizer.zero_grad()

        # ----------------------------
        # Forward pass
        # ----------------------------
        if scaler is not None:
            with autocast(device.type):
                pred = model(hits)
                loss, extras = criterion(pred, slice_labels, cp_labels)
        else:
            pred = model(hits)
            loss, extras = criterion(pred, slice_labels, cp_labels)

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

        global_step = (epoch - 1) * len(train_loader) + batch_idx
        # ======================================================
        # -------------- TensorBoard Logging -------------------
        # ======================================================
        if writer is not None and batch_idx % 10 == 0:
            global_step = (epoch - 1) * len(train_loader) + batch_idx

            # Per-component scalar logs
            writer.add_scalar("loss/total", loss.item(), global_step)
            writer.add_scalar("loss/beta_loss", extras["beta_loss"].item(), global_step)
            writer.add_scalar("loss/attr_loss", extras["attr_loss"].item(), global_step)
            writer.add_scalar("loss/repl_loss", extras["repl_loss"].item(), global_step)
            writer.add_scalar("loss/beta_ce", extras["beta_ce"].item(), global_step)
            writer.add_scalar("loss/rank_loss", extras["rank_loss"].item(), global_step)
            writer.add_scalar("loss/cp_elev", extras["cp_elev"].item(), global_step)
            writer.add_scalar("loss/ncp_supp", extras["ncp_supp"].item(), global_step)

            with torch.no_grad():
                # (B, N, 1) -> (B, N) -> (B*N,)
                beta_prob = torch.sigmoid(pred["beta"]).squeeze(-1)      # (B, N)
                beta_flat = beta_prob.detach().cpu().reshape(-1)         # (B*N,)

                writer.add_histogram("beta_prob/all", beta_flat, global_step)
                writer.add_scalar("beta_prob/mean", beta_flat.mean().item(), global_step)

                # Ground truth CP / non-CP / background masks
                cp_flat    = (cp_labels.reshape(-1).cpu() == 1)
                noncp_flat = (cp_labels.reshape(-1).cpu() == 0)
                bg_flat    = (cp_labels.reshape(-1).cpu() == -1)

                # True CP count
                true_cp_count = cp_flat.sum().item()
                writer.add_scalar("debug/true_cp_count", true_cp_count, global_step)

                # Predicted CP count at threshold 0.5
                pred_cp_count = (beta_flat > 0.5).sum().item()
                writer.add_scalar("debug/pred_cp_count", pred_cp_count, global_step)

                # Optional: separate histograms
                if cp_flat.any():
                    writer.add_histogram("beta_prob/cp_hits", beta_flat[cp_flat], global_step)
                if noncp_flat.any():
                    writer.add_histogram("beta_prob/noncp_hits", beta_flat[noncp_flat], global_step)
                if bg_flat.any():
                    writer.add_histogram("beta_prob/background", beta_flat[bg_flat], global_step)

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