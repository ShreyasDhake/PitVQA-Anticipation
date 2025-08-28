import os
import torch
import torch.nn as nn
import argparse
import torch.utils.data
import numpy as np
import random
import warnings

from transformers import GPT2Tokenizer, XCLIPProcessor
from peft import get_peft_model, TaskType, LoraConfig

from torch.utils.data import DataLoader
from dataloader import VideoQADataset, collate_qa_clipwise
from model import PitVQAGen

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def adjust_learning_rate(optimizer, shrink_factor):
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

def train(args, train_dataloader, model, criterion, optimizer, epoch, tokenizer, device):
    model.train()
    total_loss = []

    for i, (images, questions, answers) in enumerate(train_dataloader, 0):
        system_message = (
            "You are a surgical assistant AI for endonasal pituitary surgery. "
            "Rely on visual and textual input to deliver accurate, clinically relevant answers. "
            "Use proper surgical terminology. There are 3 phases, 15 steps, 18 instruments and 14 surgical activities. "
            "Time is measured in minutes. Only short sentence answers."
        )
        qa_prompt = [f'{system_message}\nQuestion: {q}\nAnswer: {a}' for q, a in zip(questions, answers)]
        qa_prompt_inputs = tokenizer(qa_prompt, truncation=True, padding="max_length", max_length=int(args.seq_length), return_tensors="pt")

        labels = qa_prompt_inputs['input_ids'].clone()
        labels = labels.to(device)

        for idx, q in enumerate(questions):
            q_prompt = f"{system_message}\nQuestion: {q}\nAnswer: "
            q_length = len(tokenizer(q_prompt)["input_ids"]) - 1

            labels[idx, :q_length] = -100
            eos_mask = (labels[idx] == tokenizer.eos_token_id)
            if eos_mask.sum() > 1:
                first_eos_pos = eos_mask.nonzero()[0].item()
                labels[idx, (first_eos_pos+1):] = -100

        logits = model(
                image=images,
                qa_inputs_ids=qa_prompt_inputs['input_ids'].to(device),
                qa_att_mask=qa_prompt_inputs['attention_mask'].to(device)
        )

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        loss = criterion(shift_logits, shift_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
    print("Training - Epoch: {}/{}, AVG Loss: {:.6f}".format(epoch, args.epochs, np.array(total_loss).mean()))

def validate(args, val_loader, model, criterion, epoch, tokenizer, device):
    total_loss = []
    model.eval()
    with torch.no_grad():
        for i, (images, questions, answers) in enumerate(val_loader, 0):
            system_message = (
                "You are a surgical assistant AI for endonasal pituitary surgery. "
                "Rely on visual and textual input to deliver accurate, clinically relevant answers. "
                "Use proper surgical terminology. There are 3 phases, 15 steps, 18 instruments and 14 surgical activities. "
                "Time is measured in minutes. Only short sentence answers."
            )
            qa_prompt = [f'{system_message}\nQuestion: {q}\nAnswer: {a}' for q, a in zip(questions, answers)]
            qa_prompt_inputs = tokenizer(qa_prompt, truncation=True, padding="max_length", max_length=int(args.seq_length), return_tensors="pt")

            labels = qa_prompt_inputs['input_ids'].clone()
            labels = labels.to(device)

            answer_starts = []
            answer_ends = []
            for idx, q in enumerate(questions):
                q_prompt = f"{system_message}\nQuestion: {q}\nAnswer: "
                q_length = len(tokenizer(q_prompt)["input_ids"]) - 1
                answer_starts.append(q_length+1)

                labels[idx, :q_length] = -100
                eos_mask = (labels[idx] == tokenizer.eos_token_id)
                if eos_mask.sum() > 1:
                    first_eos_pos = eos_mask.nonzero()[0].item()
                    labels[idx, (first_eos_pos+1):] = -100
                    answer_ends.append(first_eos_pos)

            logits = model(
                image=images,
                qa_inputs_ids=qa_prompt_inputs['input_ids'].to(device),
                qa_att_mask=qa_prompt_inputs['attention_mask'].to(device)
            )

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            loss = criterion(shift_logits, shift_labels)
            total_loss.append(loss.item())

    return np.array(total_loss).mean()

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_arg():
    parser = argparse.ArgumentParser(description='VisualQuestionAnswerGeneration')
    parser.add_argument('--epochs',         type=int,   default=60,   help='number of epochs to train for')
    parser.add_argument('--batch_size',     type=int,   default=2,   help='batch size')
    parser.add_argument('--workers',        type=int,   default=8,    help='for data-loading')
    parser.add_argument('--random_seed',    type=int,   default=42,   help='random seed')
    parser.add_argument('--seq_length',     type=int,   default=120,   help='sequence length for question and answer')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    parser.add_argument('--dataset',        default='endo',  help='endo / pit')
    parser.add_argument('--lr',             type=float, default=2e-5,  help='0.0000001, 0.00000005')
    parser.add_argument('--checkpoint_dir', default='Cross_Attention/',  help='path to checkpoint')
    parser.add_argument('--best_ckpt_name', default='SurgicalViVQA.pth', help='best checkpoint filename')

    args = parser.parse_args([])
    return args

if __name__ == '__main__':
    args = get_arg()
    seed_everything(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'Batch size: {args.batch_size}')
    print(f'Learning rate: {args.lr}')
    print(f'Random seed: {args.random_seed}')
    print(f'Sequence length: {args.seq_length}')
    print(f'Dropout: {args.dropout}')
    os.makedirs(args.checkpoint_dir, exist_ok = True)
    start_epoch = 1
    epochs_since_improvement = 0
    best_val_loss = float('inf')

    print(f'Dataset: {args.dataset}')
    train_dataloader = None
    val_dataloader = None

    # data location - PitVQA Anticipation dataset
    train_seq = ['01', '03', '04', '05', '07', '08', '09', '10', '11', '14','15', '16', '17', '18', '19', '20', '21', '22', '23', '25']
    val_seq = ['02', '06', '12','13', '24']  
    image_root = '/SAN/medic/surgicalLLM/content/PitVQA/datasets/PitVQA_Anticipation-25/images'
    qa_root = '/SAN/medic/surgicalLLM/content/PitVQA/datasets/PitVQA_Anticipation-25/QA_Anticipation'

    processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32")
    
    train_dataset = VideoQADataset(
        image_root=image_root,
        qa_root=qa_root,
        split='train',
        train_seq=train_seq,
        val_seq=val_seq,
        processor=processor
    )
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.workers, pin_memory=True,
                                collate_fn=collate_qa_clipwise)
    val_dataset = VideoQADataset(
        image_root=image_root,
        qa_root=qa_root,
        split='val',
        train_seq=train_seq,
        val_seq=val_seq,
        processor=processor
    )
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers, pin_memory=True,
                                collate_fn=collate_qa_clipwise)
    
    print('training samples:', len(train_dataset), 'validation samples:', len(val_dataset))

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"]
    )

    model = PitVQAGen(peft_config=lora_config)
    model = model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('model params: ', pytorch_total_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100).to(device)

    print('Start training.')
    for epoch in range(start_epoch, args.epochs+1):
        if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
            adjust_learning_rate(optimizer, 0.8)

        train(args, train_dataloader=train_dataloader, model=model, criterion=criterion, optimizer=optimizer,
              epoch=epoch, tokenizer=tokenizer, device=device)
        val_loss = validate(args, val_loader=val_dataloader, model=model, criterion=criterion,
                            epoch=epoch, tokenizer=tokenizer, device=device)

        if val_loss < best_val_loss:
            epochs_since_improvement = 0
            best_val_loss = val_loss
            save_dir = f'{args.checkpoint_dir}/{args.best_ckpt_name}'
            torch.save(model.state_dict(), save_dir)
            model.tokenizer.save_pretrained(args.checkpoint_dir)
            print('Best validation loss, model saved.')
        else:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))

        if epochs_since_improvement >= 5:
            print(f"\nEarly stopping triggered! No improvement for {epochs_since_improvement} epochs.")
            print("Stopping training and proceeding to inference...")
            break
    print('End training.')
