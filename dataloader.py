import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class VideoQADataset(Dataset):
    """Iterates over contiguous 16‑frame video clips and the QA pairs attached to the clip’s last frame."""

    def __init__(
        self,
        image_root: str,
        qa_root: str,
        split: str = "train",
        train_seq=None,
        val_seq=None,
        processor=None,
        sequence_length: int = 16,
    ):
        self.image_root = image_root
        self.qa_root = qa_root
        self.sequence_length = sequence_length
        self.processor = processor
        self.sequences = train_seq if split == "train" else val_seq
        self.samples = self._build_samples()

    # internal helpers
    def _build_samples(self):
        """Scan folders and build a list of (frame_paths, qa_path) tuples."""
        samples = []
        for seq in self.sequences:
            folder = f"video_{seq.zfill(2)}"
            image_folder = os.path.join(self.image_root, folder)
            qa_folder = os.path.join(self.qa_root, folder)

            if not (os.path.isdir(image_folder) and os.path.isdir(qa_folder)):
                continue

            frame_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]
            frame_ids = sorted(int(f.split(".")[0]) for f in frame_files)

            for i in range(len(frame_ids) - self.sequence_length + 1):
                chunk = frame_ids[i : i + self.sequence_length]
                if chunk != list(range(chunk[0], chunk[0] + self.sequence_length)):
                    # skip non‑contiguous windows
                    continue

                frame_paths = [os.path.join(folder, f"{fid:05d}.png") for fid in chunk]
                qa_path = os.path.join(folder, f"{chunk[-1]:05d}.txt")

                if not os.path.isfile(os.path.join(self.qa_root, qa_path)):
                    continue

                samples.append((frame_paths, qa_path))

        print(f"Total video samples: {len(samples)}")
        return samples

    def _load_qa(self, qa_path):
        qa_pairs = []
        with open(os.path.join(self.qa_root, qa_path), "r") as f:
            for line in f:
                if "|" in line:
                    q, a = line.strip().split("|", 1)
                    qa_pairs.append((q.strip(), a.strip()))
        return qa_pairs

    # Dataset interface
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, qa_path = self.samples[idx]

        # load frames
        frames = [
            Image.open(os.path.join(self.image_root, fp)).convert("RGB")
            for fp in frame_paths
        ]
        processed = self.processor(videos=[frames], return_tensors="pt")
        video_tensor = processed["pixel_values"].squeeze(0)  # [T, 3, 224, 224]

        # load QA
        qa_pairs = self._load_qa(qa_path)
        questions = [q for q, _ in qa_pairs]
        answers = [a for _, a in qa_pairs]

        return video_tensor, questions, answers


# Collate util
def collate_qa_clipwise(batch):
    """Flattens the variable‑length QA pairs so each <video, Q, A> becomes its own sample."""
    vids, flat_q, flat_a = [], [], []

    for video, qs, as_ in batch:
        for q, a in zip(qs, as_):
            vids.append(video)
            flat_q.append(q)
            flat_a.append(a)

    if len(vids) == 0:
        return torch.empty(0), [], []

    videos = torch.stack(vids, dim=0)  # (N, T, 3, 224, 224)
    return videos, flat_q, flat_a
