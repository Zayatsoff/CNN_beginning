import torch
import torch.nn.functional as F
import time


def save_checkpoint(model, optimizer, filename="checkpoint.pt"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(filename, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(filename, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # set old lr to new lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# Taken from https://github.com/rasbt/deeplearning-models
def compute_epoch_loss(model, data_loader, device):
    model.eval()
    curr_loss, num_examples = 0.0, 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            loss = F.cross_entropy(logits, targets, reduction="sum")
            num_examples += targets.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss


# Taken from https://github.com/rasbt/deeplearning-models
def compute_accuracy(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100


# Modified from https://github.com/rasbt/deeplearning-models
def train_classifier(
    num_epochs,
    model,
    optimizer,
    device,
    train_loader,
    valid_loader=None,
    loss_fn=None,
    logging_interval=100,
    skip_epoch_stats=False,
):

    log_dict = {
        "train_loss_per_batch": [],
        "train_acc_per_epoch": [],
        "train_loss_per_epoch": [],
        "valid_acc_per_epoch": [],
        "valid_loss_per_epoch": [],
    }

    if loss_fn is None:
        loss_fn = F.cross_entropy

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):

            features = features.to(device)
            targets = targets.to(device)

            # FORWARD AND BACK PROP
            logits = model(features)
            loss = loss_fn(logits, targets)
            optimizer.zero_grad()

            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            log_dict["train_loss_per_batch"].append(loss.item())

            if not batch_idx % logging_interval:
                print(
                    "Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f"
                    % (epoch + 1, num_epochs, batch_idx, len(train_loader), loss)
                )

        if not skip_epoch_stats:
            model.eval()

            with torch.set_grad_enabled(False):  # save memory during inference

                train_acc = compute_accuracy(model, train_loader, device)
                train_loss = compute_epoch_loss(model, train_loader, device)
                print(
                    "***Epoch: %03d/%03d | Train. Acc.: %.3f%% | Loss: %.3f"
                    % (epoch + 1, num_epochs, train_acc, train_loss)
                )
                log_dict["train_loss_per_epoch"].append(train_loss.item())
                log_dict["train_acc_per_epoch"].append(train_acc.item())

                if valid_loader is not None:
                    valid_acc = compute_accuracy(model, valid_loader, device)
                    valid_loss = compute_epoch_loss(model, valid_loader, device)
                    print(
                        "***Epoch: %03d/%03d | Valid. Acc.: %.3f%% | Loss: %.3f"
                        % (epoch + 1, num_epochs, valid_acc, valid_loss)
                    )
                    log_dict["valid_loss_per_epoch"].append(valid_loss.item())
                    log_dict["valid_acc_per_epoch"].append(valid_acc.item())

        print("Time elapsed: %.2f min" % ((time.time() - start_time) / 60))
        save_checkpoint(model, optimizer)
        print("Saved!")

    print("Total Training Time: %.2f min" % ((time.time() - start_time) / 60))

    return
