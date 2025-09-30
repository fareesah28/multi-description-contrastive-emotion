import torch
import torch.nn as nn
import open_clip

def _any_trainable(module: nn.Module) -> bool:
    return any(p.requires_grad for p in module.parameters())

class VideoTextEncoderV2(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        caption_pool: str = "mean_token",   
        temporal_layers: int = 1,           # 1 or 2
        tune_policy: str = "light",         
    ):
        super().__init__()
        assert caption_pool in {"mean_token", "eot_token"}
        assert tune_policy in {"light", "strict", "vision_unfrozen", "text_unfrozen", "all_unfrozen"}
        self.caption_pool = caption_pool
        self.tune_policy = tune_policy

        # load CLIP (OpenCLIP) 
        model, _, _ = open_clip.create_model_and_transforms(clip_model_name, pretrained=pretrained)
        self.clip_model = model
        self.image_encoder = model.visual
        self.text_projection = model.text_projection
        self.ln_final = model.ln_final
        self.context_length = model.context_length
        self.embed_dim = self.text_projection.shape[1]

        # freeze everything first 
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        for p in self.clip_model.transformer.parameters():
            p.requires_grad = False
        if isinstance(self.ln_final, nn.LayerNorm):
            for p in self.ln_final.parameters():
                p.requires_grad = False
        self.text_projection.requires_grad = False

        # apply tune policy 
        if self.tune_policy == "light":
            if isinstance(self.ln_final, nn.LayerNorm):
                for p in self.ln_final.parameters():
                    p.requires_grad = True
            self.text_projection.requires_grad = True
            for _, p in self.clip_model.transformer.resblocks[0].named_parameters():
                p.requires_grad = True

        elif self.tune_policy == "strict":
            pass

        elif self.tune_policy == "vision_unfrozen":
            for p in self.image_encoder.parameters():
                p.requires_grad = True

        elif self.tune_policy == "text_unfrozen":
            for p in self.clip_model.transformer.parameters():
                p.requires_grad = True
            if isinstance(self.ln_final, nn.LayerNorm):
                for p in self.ln_final.parameters():
                    p.requires_grad = True
            self.text_projection.requires_grad = True

        elif self.tune_policy == "all_unfrozen":
            for p in self.image_encoder.parameters():
                p.requires_grad = True
            for p in self.clip_model.transformer.parameters():
                p.requires_grad = True
            if isinstance(self.ln_final, nn.LayerNorm):
                for p in self.ln_final.parameters():
                    p.requires_grad = True
            self.text_projection.requires_grad = True

        # heads/projections 
        self.caption_projection = nn.Linear(self.embed_dim, self.embed_dim)
        self.video_projection   = nn.Linear(self.embed_dim, self.embed_dim)

        # temporal encoder (on frame features) 
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, nhead=4, dim_feedforward=512, dropout=0.1, batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers=temporal_layers)

        # learnable temperature (CLIP-style logit_scale) 
        # init so that exp(logit_scale) ~= 1/0.07
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07)))

    # encoders

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video.shape
        vision_trainable = _any_trainable(self.image_encoder)

        vid = video.view(B * T, C, H, W)
        ctx = torch.enable_grad() if vision_trainable else torch.no_grad()
        with ctx:
            frame_features = self.image_encoder(vid)  # [B*T, D]
        frame_features = frame_features.view(B, T, -1)   # [B, T, D]
        encoded = self.temporal_encoder(frame_features)  # [B, T, D]
        v = encoded.mean(dim=1)                          # [B, D]
        return self.video_projection(v)                  # [B, D]

    def encode_captions(self, captions_token_ids: torch.Tensor) -> torch.Tensor:
        assert isinstance(captions_token_ids, torch.Tensor), \
            "Expected token IDs tensor of shape [B, 5, L] from collate_fn"
        B, N, L = captions_token_ids.shape
        device = next(self.parameters()).device
        toks = captions_token_ids.to(device).view(B * N, L)  # [B*N, L]

        text_fully_frozen = (
            not _any_trainable(self.clip_model.transformer)
            and not _any_trainable(self.ln_final if isinstance(self.ln_final, nn.LayerNorm) else nn.Module())
            and not self.text_projection.requires_grad
        )
        ctx = torch.no_grad() if text_fully_frozen else torch.enable_grad()
        with ctx:
            x = self.clip_model.encode_text(toks)            # [B*N, D]

        x = self.caption_projection(x)                       # [B*N, D]
        return x.view(B, N, -1)                              # [B, 5, D]

    def forward(self, video, captions_token_ids):
        v = self.encode_video(video)                 # [B, D]
        c = self.encode_captions(captions_token_ids) # [B, 5, D]
        return v, c
