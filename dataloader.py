import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class VideoQADataset(Dataset):
    def __init__(self, image_root, qa_root, split='train', train_seq=None, val_seq=None, processor=None, sequence_length=8):
        self.image_root = image_root
        self.qa_root = qa_root
        self.sequence_length = sequence_length
        self.processor = processor
        self.sequences = train_seq if split == 'train' else val_seq
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        for seq in self.sequences:
            folder = f'video_{seq.zfill(2)}'
            image_folder = os.path.join(self.image_root, folder)
            qa_folder = os.path.join(self.qa_root, folder)

            if not os.path.isdir(image_folder) or not os.path.isdir(qa_folder):
                continue

            frame_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
            frame_ids = sorted([int(f.split('.')[0]) for f in frame_files])

            for i in range(len(frame_ids) - self.sequence_length + 1):
                chunk = frame_ids[i:i+self.sequence_length]
                expected = list(range(chunk[0], chunk[0]+self.sequence_length))
                if chunk != expected:
                    continue  # Skip non-contiguous chunks

                frame_paths = [
                    os.path.join(folder, f"{fid:05d}.png") for fid in chunk
                ]

                last_frame_id = chunk[-1]
                qa_path = os.path.join(folder, f"{last_frame_id:05d}.txt")
                full_qa_path = os.path.join(self.qa_root, qa_path)

                if not os.path.isfile(full_qa_path):
                    continue

                samples.append((frame_paths, qa_path))
                
        print(f"Total video samples: {len(samples)}")
        return samples

    def _load_qa(self, qa_path):
        qa_pairs = []
        full_path = os.path.join(self.qa_root, qa_path)
        with open(full_path, 'r') as f:
            for line in f:
                if "|" in line:
                    q, a = line.strip().split("|", 1)
                    qa_pairs.append((q.strip(), a.strip()))
        return qa_pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, qa_path = self.samples[idx]
        frames = [Image.open(os.path.join(self.image_root, p)).convert('RGB') for p in frame_paths]
        # Use processor to process the batch of frames (videos expects [List[Image]])
        processed = self.processor(videos=[frames], return_tensors="pt")
        video_tensor = processed["pixel_values"].squeeze(0)  # [T, 3, 224, 224]
        qa_pairs = self._load_qa(qa_path)
        questions = [q for q, a in qa_pairs]
        answers = [a for q, a in qa_pairs]
        return video_tensor, questions, answers

def collate_qa_clipwise(batch):
    """Custom collate function to handle multiple QA pairs per video"""
    vids, flat_q, flat_a = [], [], []
    for video, qs, as_ in batch:
        for q, a in zip(qs, as_):
            vids.append(video)
            flat_q.append(q)
            flat_a.append(a)
    
    if len(vids) == 0:
        return torch.empty(0), [], []
        
    videos = torch.stack(vids, dim=0)   # (N, T, 3, 224, 224)
    return videos, flat_q, flat_a
