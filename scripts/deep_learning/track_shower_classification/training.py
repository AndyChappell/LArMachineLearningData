import torch

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model = model.to(device)
    model.train()

    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
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
        optimizer.step()

        num_tokens = mask.sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    return avg_loss


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device):
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

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

        num_tokens = mask_flat.sum().item()

        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        total_correct += correct

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens

    return avg_loss, accuracy