import os
import random
import warnings
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import GPT2Tokenizer, XCLIPProcessor
from peft import LoraConfig, TaskType

from dataloader import VideoQADataset, collate_qa_clipwise
from model import PitVQAGen

warnings.filterwarnings("ignore")

SYSTEM_MESSAGE = (
    "You are a surgical assistant AI for endonasal pituitary surgery. "
    "Rely on visual and textual input to deliver accurate, clinically relevant answers. "
    "Use proper surgical terminology. There are 3 phases, 15 steps, 18 instruments and 14 surgical activities. "
    "Time is measured in minutes. Only short sentence answers."
)

# helpers & utilities
def adjust_learning_rate(optimizer, shrink_factor):
    for g in optimizer.param_groups:
        g["lr"] *= shrink_factor
    print(f"\nDECAYING learning rate → {optimizer.param_groups[0]['lr']:.2e}\n")

def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# training / validation loops
def train_epoch(args, loader, model, criterion, optimizer, tokenizer, device, epoch):
    model.train()
    losses = []

    for images, questions, answers in loader:
        # build prompts
        prompt = [
            f"{SYSTEM_MESSAGE}\nQuestion: {q}\nAnswer: {a}"
            for q, a in zip(questions, answers)
        ]
        inputs = tokenizer(
            prompt, truncation=True, padding="max_length", max_length=args.seq_length, return_tensors="pt"
        )

        # label masking (mask question & padding)
        labels = inputs["input_ids"].clone()
        for idx, q in enumerate(questions):
            q_len = len(tokenizer(f"{SYSTEM_MESSAGE}\nQuestion: {q}\nAnswer: ")["input_ids"]) - 1
            labels[idx, :q_len] = -100
            eos_mask = labels[idx] == tokenizer.eos_token_id
            if eos_mask.sum() > 1:
                first_eos = eos_mask.nonzero()[0].item()
                labels[idx, first_eos + 1 :] = -100

        # forward
        logits = model(
            image=images,
            qa_inputs_ids=inputs["input_ids"].to(device),
            qa_att_mask=inputs["attention_mask"].to(device),
        )

        # shift for autoregressive loss
        loss = criterion(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"Epoch {epoch}: train‑loss = {np.mean(losses):.6f}")

@torch.no_grad()
def validate_epoch(args, loader, model, criterion, tokenizer, device, epoch):
    model.eval()
    losses = []

    for images, questions, answers in loader:
        prompt = [
            f"{SYSTEM_MESSAGE}\nQuestion: {q}\nAnswer: {a}"
            for q, a in zip(questions, answers)
        ]
        inputs = tokenizer(prompt, truncation=True, padding="max_length", max_length=args.seq_length, return_tensors="pt")
        labels = inputs["input_ids"].clone()

        for idx, q in enumerate(questions):
            q_len = len(tokenizer(f"{SYSTEM_MESSAGE}\nQuestion: {q}\nAnswer: ")["input_ids"]) - 1
            labels[idx, :q_len] = -100
            eos_mask = labels[idx] == tokenizer.eos_token_id
            if eos_mask.sum() > 1:
                first_eos = eos_mask.nonzero()[0].item()
                labels[idx, first_eos + 1 :] = -100

        logits = model(
            image=images,
            qa_inputs_ids=inputs["input_ids"].to(device),
            qa_att_mask=inputs["attention_mask"].to(device),
        )
        loss = criterion(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
        )
        losses.append(loss.item())

    avg_loss = np.mean(losses)
    print(f"Epoch {epoch}: val‑loss = {avg_loss:.6f}")
    return avg_loss

# CLI
def parse_args():
    p = argparse.ArgumentParser("PitVQAGen training")

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--seq_length", type=int, default=120)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seq_frames", type = int, default = 8)
    p.add_argument("--dataset", default="endo")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--checkpoint_dir", default="\Your\Directory")
    p.add_argument("--best_ckpt_name", default="best_model_path.pth")

    return p.parse_args()

def main():
    args = parse_args()
    seed_everything(args.random_seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    train_seq = ['01', '03', '04', '05', '07', '08', '09', '10', '11', '14',
                     '15', '16', '17', '18', '19', '20', '21', '22', '23', '25']
    val_seq   = ["02", "06", "12", "13", "24"]
    image_root = "/SAN/medic/surgicalLLM/content/PitVQA/datasets/PitVQA_Anticipation-25/images"
    qa_root    = "/SAN/medic/surgicalLLM/content/PitVQA/datasets/PitVQA_Anticipation-25/QA_Anticipation"

    processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32")

    train_ds = VideoQADataset(image_root, qa_root, "train", train_seq, val_seq, processor, sequence_length=args.seq_frames)
    val_ds   = VideoQADataset(image_root, qa_root, "val",   train_seq, val_seq, processor, sequence_length=args.seq_frames)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers,
                              pin_memory=True, collate_fn=collate_qa_clipwise)

    val_loader = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=True, collate_fn=collate_qa_clipwise)

    # model
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],
    )
    model = PitVQAGen(peft_config=lora_cfg).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100).to(device)

    best_val = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        if epochs_no_improve and epochs_no_improve % 5 == 0:
            adjust_learning_rate(optimizer, 0.8)

        train_epoch(args, train_loader, model, criterion, optimizer, tokenizer=model.tokenizer, device=device, epoch=epoch)
        val_loss = validate_epoch(args, val_loader, model, criterion, tokenizer=model.tokenizer, device=device, epoch=epoch)

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            ckpt_path = os.path.join(args.checkpoint_dir, args.best_ckpt_name)
            torch.save(model.state_dict(), ckpt_path)
            model.tokenizer.save_pretrained(args.checkpoint_dir)
            print("★ Saved new best model.")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= 5:
            print("Early stopping – moving to inference.")
            break

if __name__ == "__main__":
    main()
