import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from decord import VideoReader, cpu
from open_clip import tokenize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Subset

class MAFWDataset(Dataset):
    def __init__(self, video_dir, caption_csv, num_frames=8, image_size=224, tokenizer=None):
        self.video_dir = video_dir
        self.caption_df = pd.read_csv(caption_csv)
        self.num_frames = num_frames
        self.tokenizer = tokenizer or tokenize

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

    def __len__(self):
        return len(self.caption_df)

    def __getitem__(self, idx):
        row = self.caption_df.iloc[idx]
        video_id = row['video']
        video_path = os.path.join(self.video_dir, video_id)
        video_tensor = self._load_video_frames(video_path)
        captions = [row[f'caption_{i}'] for i in range(1, 6)]
        caption_tokens = self.tokenizer(captions)  # [5, seq_len]
        return {
            "video": video_tensor,          # [8, 3, 224, 224]
            "captions": caption_tokens,     # [5, seq_len]
            "video_id": video_id
        }

    def _load_video_frames(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        indices = torch.linspace(0, total_frames - 1, steps=self.num_frames).long().tolist()
        frames = [self.transform(Image.fromarray(vr[i].asnumpy())) for i in indices]
        return torch.stack(frames)  # [8, 3, 224, 224]


def collate_fn_mafw(batch):
    videos = torch.stack([item["video"] for item in batch])        # [B, 8, 3, 224, 224]
    video_ids = [item["video_id"] for item in batch]
    caption_lists = [item["captions"] for item in batch]           # list of [5, seq_len]
    flat_captions = [cap.clone().detach() for caps in caption_lists for cap in caps]
    padded = pad_sequence(flat_captions, batch_first=True, padding_value=0)  # [B*5, max_len]
    B = len(batch)
    captions = padded.view(B, 5, -1)                                # [B, 5, max_len]
    return {
        "video": videos,
        "captions": captions,
        "video_id": video_ids
    }


def get_dataloaders(video_dir, train_csv, val_csv, batch_size=4, num_workers=2, debug=False):
    train_dataset = MAFWDataset(video_dir, train_csv)
    val_dataset = MAFWDataset(video_dir, val_csv)

    if debug:
        print("Debug mode ON")
        train_dataset = Subset(train_dataset, range(min(1000, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(250, len(val_dataset))))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn_mafw)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_mafw)

    return train_loader, val_loader


