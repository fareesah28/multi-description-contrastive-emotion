import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from decord import VideoReader, cpu
from open_clip import tokenize  

class MAFWDataset(Dataset):
    def __init__(self, video_dir, caption_csv, num_frames=8, image_size=224, tokenizer=None):
        self.video_dir = video_dir
        self.caption_df = pd.read_csv(caption_csv)  
        self.num_frames = num_frames
        self.tokenizer = tokenizer or tokenize

        # preprocessing for CLIP (ViT-B/32)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # resize frame to 224x224
            transforms.ToTensor(),  # convert PIL to Tensor, [3, 224, 224]
            transforms.Normalize(  # normalise using CLIP stats
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

        video_tensor = self._load_video_frames(video_path) # [8, 3, 224, 224]  (8 sampled and transformed RGB frames)

        captions = [row[f'caption_{i}'] for i in range(1, 6)]  # list of 5 caption strings
        caption_tokens = self.tokenizer(captions) # [5, seq_len] (tokenised using OpenCLIP)

        return {
            "video": video_tensor,          
            "captions": caption_tokens,     
            "video_id": video_id            
        }

    def _load_video_frames(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0))  # load video using decord
        total_frames = len(vr)  

        indices = torch.linspace(0, total_frames - 1, steps=self.num_frames).long().tolist() # 8 evenly spaced frame indices

        frames = [self.transform(Image.fromarray(vr[i].asnumpy())) for i in indices] # [3, 224, 224]

        return torch.stack(frames)  # [8, 3, 224, 224]
