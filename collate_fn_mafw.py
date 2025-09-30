import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from dataset_mafw import MAFWDataset

def collate_fn_mafw(batch):

    videos = torch.stack([item["video"] for item in batch])  # [B, 8, 3, H, W]
    video_ids = [item["video_id"] for item in batch]

    # pad captions to max sequence length across batch
    # for captions: list of [5, seq_len] to list of 5 tensors per item
    caption_lists = [item["captions"] for item in batch]  # list of [5, seq_len]

    # flatten to list of [seq_len]
    flat_captions = [torch.tensor(cap) for caps in caption_lists for cap in caps]

    # pad to same length
    padded = pad_sequence(flat_captions, batch_first=True, padding_value=0)  # [B*5, max_len]

    # reshape back to [B, 5, max_len]
    B = len(batch)
    captions = padded.view(B, 5, -1)

    return {
        "video": videos,           # [B, 8, 3, H, W]
        "captions": captions,      # [B, 5, max_len]
        "video_id": video_ids
    }




