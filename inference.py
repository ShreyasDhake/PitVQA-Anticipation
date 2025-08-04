import os
import random
import re
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate
import argparse
from peft import LoraConfig, TaskType
from transformers import GPT2Tokenizer, XCLIPProcessor

from dataloader import VideoQADataset, collate_qa_clipwise
from model import PitVQAGen
from main import argparse as args
SYSTEM_MESSAGE = (
    "You are a surgical assistant AI for endonasal pituitary surgery. "
    "Rely on visual and textual input to deliver accurate, clinically relevant answers. "
    "Use proper surgical terminology. There are 3 phases, 15 steps, 18 instruments and 14 surgical activities. "
    "Time is measured in minutes. Only short sentence answers."
)



# greedy generation  
def batch_greedy(images, questions, model, tokenizer, max_len, device):
    B = len(questions)
    model.eval()
    with torch.no_grad():
        prompts = [f"{SYSTEM_MESSAGE}\nQuestion: {q}\nAnswer:" for q in questions]
        enc = tokenizer(prompts, return_tensors="pt", padding="longest", add_special_tokens=False)
        ids = torch.zeros((B, max_len), dtype=torch.long, device=device)
        attn = torch.zeros_like(ids)
        L0 = enc["input_ids"].size(1)
        ids[:, :L0] = enc["input_ids"]
        attn[:, :L0] = enc["attention_mask"]
        generated = torch.empty((B, 0), dtype=torch.long, device=device)

        finished = torch.zeros(B, dtype=torch.bool, device=device)
        valid_lens = attn.sum(dim=1).long()
        batch_idx = torch.arange(B, device=device)

        for _ in range(max_len - L0):
            logits = model(
                image=images,
                qa_inputs_ids=ids[:, : valid_lens.max()],
                qa_att_mask=attn[:, : valid_lens.max()],
            )
            next_tok = logits[batch_idx, valid_lens - 1].argmax(dim=-1)
            ids[batch_idx, valid_lens] = next_tok
            attn[batch_idx, valid_lens] = 1
            valid_lens += 1

            generated = torch.cat([generated, next_tok.unsqueeze(1)], dim=1)
            finished |= next_tok.eq(tokenizer.eos_token_id)
            if finished.all():
                break

        # decode
        answers = []
        for seq in generated.cpu().tolist():
            try:
                eos = seq.index(tokenizer.eos_token_id)
                seq = seq[:eos]
            except ValueError:
                pass
            answers.append(tokenizer.decode(seq, skip_special_tokens=True).strip())
        return answers

def main():
    # paths & hyper‑params
    checkpoint_dir = "\Your\Directory"
    best_ckpt_name = "best_model_path.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset 
    train_seq = ['01', '03', '04', '05', '07', '08', '09', '10', '11', '14',
                     '15', '16', '17', '18', '19', '20', '21', '22', '23', '25']
    val_seq   = ["02", "06", "12", "13", "24"]
    image_root = "/SAN/medic/surgicalLLM/content/PitVQA/datasets/PitVQA_Anticipation-25/images"
    qa_root    = "/SAN/medic/surgicalLLM/content/PitVQA/datasets/PitVQA_Anticipation-25/QA_Anticipation"
        
    processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32")

    val_ds = VideoQADataset(image_root, qa_root, "val", train_seq, val_seq, processor,sequence_length=args.seq_frames)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=8,
                            pin_memory=True, collate_fn=collate_qa_clipwise)

    # model
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],
    )
    model = PitVQAGen(peft_config=lora_cfg)
    ckpt = os.path.join(checkpoint_dir, best_ckpt_name)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()

    tokenizer = model.tokenizer

    references, hypotheses = [], []

    for images, questions, answers in tqdm(val_loader, desc="Infer"):
        images = images.to(device)
        hyp = batch_greedy(images, questions, model, tokenizer, max_len=120, device=device)
        references.extend(answers)
        hypotheses.extend(hyp)

    # metrics 
    bleu = evaluate.load("bleu").compute(predictions=hypotheses, references=references)
    rouge = evaluate.load("rouge").compute(predictions=hypotheses, references=references)
    meteor = evaluate.load("meteor").compute(predictions=hypotheses, references=references)

    print("\n=== NLP metrics ===")
    print({k: v for k, v in rouge.items() if k in ("rouge1", "rougeL")})
    print({"BLEU": bleu["bleu"], "Meteor": meteor["meteor"]})

    # show some samples
    print("\n=== Random samples ===")
    idxs = random.sample(range(len(references)), k=min(7, len(references)))
    for i, idx in enumerate(idxs, 1):
        print(f"[{i}] REF: {references[idx]}")
        print(f"    HYP: {hypotheses[idx]}\n")

if __name__ == "__main__":
    main()
