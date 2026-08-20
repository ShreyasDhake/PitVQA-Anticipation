import os
import torch
import argparse
import random
import warnings
import evaluate
from tqdm import tqdm

from transformers import XCLIPProcessor

from torch.utils.data import DataLoader
from dataloader import VideoQADataset, collate_qa_clipwise
from model import (
    LANGUAGE_MODEL_CONFIGS,
    PitVQAGen,
    add_artifact_prefix,
    create_lora_config,
)

warnings.filterwarnings('ignore')

print('Inferencing....')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_nlp_mettics(references, hypotheses):
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load('meteor')

    results_bleu = bleu.compute(predictions=hypotheses, references=references)
    results_rouge = rouge.compute(predictions=hypotheses, references=references)
    results_meteor = meteor.compute(predictions=hypotheses, references=references)

    return results_bleu, results_rouge, results_meteor

def batch_greedy_search(images, questions, model, tokenizer, max_length, device):
    answers = []
    batch_size = len(questions)

    model.eval()
    with torch.no_grad():
        system_message = (
            "You are a surgical assistant AI for endonasal pituitary surgery. "
            "Rely on visual and textual input to deliver accurate, clinically relevant answers. "
            "Use proper surgical terminology. There are 3 phases, 15 steps, 18 instruments and 14 surgical activities. "
            "Time is measured in minutes. Only short sentence answers."
        )
        prompt_texts = [f"{system_message}\nQuestion: {q}\nAnswer:" for q in questions]

        prompt_inputs = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding='longest',
            add_special_tokens=False
        )

        padded_input_ids = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
        padded_attention_mask = torch.zeros((batch_size, max_length), device=device)

        orig_length = prompt_inputs['input_ids'].size(1)
        padded_input_ids[:, :orig_length] = prompt_inputs['input_ids']
        padded_attention_mask[:, :orig_length] = prompt_inputs['attention_mask']

        only_answer_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        valid_lengths = padded_attention_mask.sum(dim=1).long()
        batch_indices = torch.arange(batch_size, device=device)

        for _ in range(max_length):
            max_valid_lengths = max(valid_lengths).item()
            if max_valid_lengths >= max_length:
                break

            logits = model(image=images, qa_inputs_ids=padded_input_ids[:, :max_valid_lengths], qa_att_mask=padded_attention_mask[:, :max_valid_lengths])

            last_valid_logits = logits[batch_indices, valid_lengths - 1, :]

            next_token_ids = torch.argmax(last_valid_logits, dim=-1)

            is_eos = (next_token_ids == tokenizer.eos_token_id)
            finished = finished | is_eos

            padded_input_ids[batch_indices, valid_lengths] = next_token_ids
            padded_attention_mask[batch_indices, valid_lengths] = 1
            valid_lengths += 1

            only_answer_ids = torch.cat([only_answer_ids, next_token_ids.unsqueeze(1)], dim=1)

            if finished.all():
                break

        generated_ids_cpu = only_answer_ids.cpu().tolist()
        for i in range(batch_size):
            try:
                eos_index = generated_ids_cpu[i].index(tokenizer.eos_token_id)
                answer_ids = generated_ids_cpu[i][:eos_index]
            except ValueError:
                answer_ids = generated_ids_cpu[i]

            answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            answers.append(answer)

    return answers

def inference(args, val_loader, model, tokenizer, device):
    references = []
    hypotheses = []

    model.eval()
    with torch.no_grad():
        for i, (images, questions, answers) in enumerate(tqdm(val_loader), 0):
            images = images.to(device)
            generated_answers = batch_greedy_search(
                images,
                questions,
                model,
                tokenizer,
                max_length=args.seq_length,
                device=device
            )

            references.extend(answers)
            hypotheses.extend(generated_answers)

    return references, hypotheses

def get_arg():
    parser = argparse.ArgumentParser(description='VisualQuestionAnswerGeneration-Inference')
    parser.add_argument('--seq_length',     type=int,   default=120,   help='sequence length for decoding')
    parser.add_argument('--seq_frames',     type=int,   default=8,     help='number of contiguous video frames')
    parser.add_argument('--batch_size',     type=int,   default=2,   help='batch size (will be doubled for inference loader)')
    parser.add_argument('--workers',        type=int,   default=8,    help='for data-loading')
    parser.add_argument('--checkpoint_dir', default='Checkpoints/',  help='path to checkpoint')
    parser.add_argument('--best_ckpt_name', default='SurgAnt.pth', help='checkpoint filename')
    parser.add_argument(
        '--language_model', '--lm',
        choices=LANGUAGE_MODEL_CONFIGS.keys(),
        default='gpt2',
        help='causal language-model backbone',
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arg()

    lora_config = create_lora_config(args.language_model)

    model = PitVQAGen(
        language_model=args.language_model,
        peft_config=lora_config,
    )
    checkpoint_name = add_artifact_prefix(
        args.best_ckpt_name, args.language_model
    )
    save_dir = os.path.join(args.checkpoint_dir, checkpoint_name)
    model.load_state_dict(torch.load(save_dir, map_location=device))
    model.to(device)
    model.eval()

    tokenizer = model.tokenizer

    processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32")

    # Same dataset splits/paths as training script
    train_seq = ['01', '03', '04', '05', '07', '08', '09', '10', '11', '14','15', '16', '17', '18', '19', '20', '21', '22', '23', '25']
    val_seq = ['02', '06', '12','13', '24']  
    image_root = '.../PitVQA_Anticipation/images'
    qa_root = '.../PitVQA_Anticipation/QA_Anticipation'

    val_dataset = VideoQADataset(
        image_root=image_root,
        qa_root=qa_root,
        split='val',
        train_seq=train_seq,
        val_seq=val_seq,
        processor=processor,
        sequence_length=args.seq_frames
    )
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size*2,
                                shuffle=False, num_workers=args.workers, pin_memory=True,
                                collate_fn=collate_qa_clipwise)

    references, hypotheses  = inference(args, val_loader=val_dataloader, model=model, tokenizer=tokenizer, device=device)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ref_name = add_artifact_prefix('SurgAnt_ref.txt', args.language_model)
    hyp_name = add_artifact_prefix('SurgAnt_hyp.txt', args.language_model)
    with open(os.path.join(args.checkpoint_dir, ref_name), 'w') as f:
        for ref in references:
            f.write(ref + '\n')

    with open(os.path.join(args.checkpoint_dir, hyp_name), 'w') as f:
        for hyp in hypotheses:
            f.write(hyp + '\n')

    print(f"References and hypotheses saved to {args.checkpoint_dir}")

    results_bleu, results_rouge, results_meteor = get_nlp_mettics(references, hypotheses)

    print(f"BLEU-1: {results_bleu['precisions'][0]:.6f}, "
          f"BLEU-2: {results_bleu['precisions'][1]:.6f}, "
          f"BLEU-3: {results_bleu['precisions'][2]:.6f}, "
          f"BLEU-4: {results_bleu['precisions'][3]:.6f}, "
          f"Overall BLEU: {results_bleu['bleu']:.6f}")

    print(f"Rouge1: {results_rouge['rouge1']:.6f}")
    print(f"RougeL: {results_rouge['rougeL']:.6f}")
    print(f"Meteor: {results_meteor['meteor']:.6f}")

    print('\n' + '='*80)
    print('7 RANDOM SAMPLES:')
    print('='*80)

    random.seed(42)
    total_samples = len(references)
    random_indices = random.sample(range(total_samples), min(7, total_samples))

    for i, idx in enumerate(random_indices):
        print(f'\nSAMPLE {i+1} (Index: {idx}):')
        print('-' * 50)
        print(f'Reference: {references[idx]}')
        print(f'Hypothesis: {hypotheses[idx]}')
        print('-' * 50)

    print('First 5 Labels:')
    print(references[:5])

    print('First 5 Prediction:')
    print(hypotheses[:5])

    print("Inference completed.")
