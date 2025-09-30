import torch
import torch.nn.functional as F

def multi_positive_contrastive_loss(video_embeds, caption_embeds, temperature=None, return_debug=False):

    B, N, D = caption_embeds.shape
    v = F.normalize(video_embeds, dim=-1)
    c = F.normalize(caption_embeds, dim=-1).view(B * N, D)

    if temperature is None:
        scale = torch.tensor(1.0 / 0.07, device=v.device) # CLIP default
    elif torch.is_tensor(temperature):
        # temperature is logit_scale, use exp() and clamp like CLIP
        scale = temperature.exp().clamp(max=100.0)
    else:
        # temperature is a float T, convert to 1/T scale
        scale = torch.tensor(1.0 / float(temperature), device=v.device)

    logits = (v @ c.T) * scale                     

    targets = torch.zeros(B, B * N, device=v.device)
    for i in range(B):
        targets[i, i*N:(i+1)*N] = 1.0               # multi-hot positives

    loss = F.binary_cross_entropy_with_logits(logits, targets)

    if return_debug:
        probs = torch.sigmoid(logits)
        return loss, logits, targets, probs
    return loss
