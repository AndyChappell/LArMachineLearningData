import torch

def train_one_epoch(model, dataloader, optimizer, criterion, device, scheduler=None, writer=None, epoch=0):
    model = model.to(device)
    model.train()

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    all_preds = []
    all_labels = []

    for batch_idx, batch in enumerate(dataloader):
        hits = batch["hits"].to(device)         # (B, N, F)
        labels = batch["labels"].to(device)     # (B, N)
        mask = batch["mask"].to(device)         # (B, N)

        optimizer.zero_grad()

        outputs = model(hits, mask=mask)        # (B, N, C)

        # Flatten for loss
        B, N, C = outputs.shape
        outputs = outputs.view(B * N, C)
        labels = labels.view(B * N)
        mask = mask.view(B * N)

        # Only compute loss on valid tokens
        outputs = outputs[mask]
        labels = labels[mask]

        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        preds = outputs.argmax(dim=-1)
        correct = (preds == labels).sum().item()
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

        num_tokens = mask.sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        total_correct += correct

        if writer is not None:
            step = epoch * len(dataloader) + batch_idx
            writer.add_scalar("Loss/Train_batch", loss.item(), step)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], step)

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return avg_loss, accuracy, all_preds, all_labels


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device):
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        hits = batch["hits"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)

        outputs = model(hits, mask=mask)  # (B, N, C)

        B, N, C = outputs.shape
        outputs_flat = outputs.view(B * N, C)
        labels_flat = labels.view(B * N)
        mask_flat = mask.view(B * N)

        outputs_valid = outputs_flat[mask_flat]
        labels_valid = labels_flat[mask_flat]

        loss = criterion(outputs_valid, labels_valid)

        # Accuracy
        preds = outputs_valid.argmax(dim=-1)
        correct = (preds == labels_valid).sum().item()
        all_preds.append(preds.cpu())
        all_labels.append(labels_valid.cpu())

        num_tokens = mask_flat.sum().item()

        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        total_correct += correct

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return avg_loss, accuracy, all_preds, all_labels
