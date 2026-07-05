import os
import torch
from torch.optim import AdamW

import config
from src.data_utils import get_batch


def compute_accuracy(logits, labels):
    """
    Token-level accuracy: fraction of non-padding positions where the
    predicted token matches the target token.

    Args:
        logits: model output logits, shape (batch, seq_len, vocab_size)
        labels: target token ids, shape (batch, seq_len), padding = -100
    """
    preds   = torch.argmax(logits, dim=-1)
    mask    = labels != -100
    correct = (preds == labels) & mask
    return correct.sum().item() / mask.sum().item()


def run_epoch(model, src_data, tgt_data, tokenizer, device,
              batch_size, optimizer=None, train=True):
    """
    Runs one epoch of training or validation.

    Args:
        optimizer: required if train=True, unused if train=False
        train: True for training phase (backprop), False for validation

    Returns:
        (avg_loss, avg_accuracy)
    """
    model.train() if train else model.eval()

    total_loss, total_acc, steps = 0.0, 0.0, 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for i in range(0, len(src_data), batch_size):
            input_ids, attention_mask, labels = get_batch(
                src_data, tgt_data, i, batch_size, tokenizer, device
            )

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            acc  = compute_accuracy(outputs.logits, labels)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            total_acc  += acc
            steps      += 1

            if train and steps % config.LOG_EVERY_N_STEPS == 0:
                print(f"    Step {steps} | Loss: {loss.item():.4f} | Acc: {acc*100:.2f}%")

    return total_loss / steps, total_acc / steps


def train_model(model, tokenizer, train_src, train_tgt, val_src, val_tgt, device):
    """
    Full training loop across config.EPOCHS epochs.
    Saves the best model (by validation loss) to config.SAVE_PATH.
    """
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    best_val_loss = float("inf")

    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")

        avg_train, avg_train_acc = run_epoch(
            model, train_src, train_tgt, tokenizer, device,
            config.BATCH_SIZE, optimizer=optimizer, train=True,
        )

        avg_val, avg_val_acc = run_epoch(
            model, val_src, val_tgt, tokenizer, device,
            config.BATCH_SIZE, train=False,
        )

        print(f"Epoch {epoch + 1} | Train Loss: {avg_train:.4f} | Train Acc: {avg_train_acc*100:.2f}% "
              f"| Val Loss: {avg_val:.4f} | Val Acc: {avg_val_acc*100:.2f}%")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), config.SAVE_PATH)
            print(f"  ✓ Saved! (best val loss: {avg_val:.4f}, val acc: {avg_val_acc*100:.2f}%)")

    return model
