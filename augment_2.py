import os, copy, warnings
import pandas as pd

os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"
os.environ["PYTHONBREAKPOINT"] = "0"

if "SLURM_JOB_ID" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("Not in a GPU job: forcing CPU-only mode")

import torch, numpy as np
from decord import VideoReader, cpu
from transformers import AutoConfig
from llava.model.builder  import load_pretrained_model
from llava.mm_utils        import tokenizer_image_token
from llava.constants       import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation    import conv_templates

warnings.filterwarnings("ignore")

CLIPS_DIR    = "/mnt/scratch/users/adhg808/MAFW/data/clips"
LABELS_XLSX  = "/mnt/scratch/users/adhg808/MAFW/Labels/descriptive_text.xlsx"
OUTPUT_CSV   = "/mnt/scratch/users/adhg808/MAFW/Labels/generated_captions_more.csv"
NUM_FRAMES   = 8
NUM_CAPTIONS = 5

MODEL_NAME = "lmms-lab/LLaVA-Video-7B-Qwen2"
MODEL_TYPE = "llava_qwen"

# setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
if DEVICE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["ATTN_IMPLEMENTATION"] = "eager"
    print("CPU mode: forcing eager attention")

# load frames
def load_video(video_path, max_frames):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total = len(vr); fps = vr.get_avg_fps()
    idxs = np.linspace(0, total - 1, max_frames, dtype=int)
    return vr.get_batch(idxs).asnumpy()

# load model once
print("Loading model…")
tokenizer, model, image_processor, _ = load_pretrained_model(
    MODEL_NAME, None, MODEL_TYPE,
    torch_dtype=torch.bfloat16 if DEVICE=="cuda" else torch.float32,
    device_map=DEVICE,
    trust_remote_code=True,
    attn_implementation="eager",
)
model.eval()
print("Model loaded.")

# load csv and get existing video names
df_human = pd.read_excel(LABELS_XLSX, header=None, engine='openpyxl')
existing_videos = set(str(row[0]) for _, row in df_human.iterrows())

# loop over clips not in csv
results = []
video_list = sorted([f for f in os.listdir(CLIPS_DIR) if f.endswith(".mp4")])
print(f" Found {len(video_list)} videos in total.")
print(f" {len(existing_videos)} videos already have human captions.")
print(f" Generating captions for {len(video_list) - len(existing_videos)} new videos.")

for video_file in video_list:
    if video_file in existing_videos:
        continue  # skip if already in csv

    video_path = os.path.join(CLIPS_DIR, video_file)
    print(f"\n Processing: {video_file}")
    frames_np = load_video(video_path, NUM_FRAMES)
    pv = image_processor.preprocess(frames_np, return_tensors="pt")["pixel_values"].to(DEVICE)
    video_batch = [pv]

    # prompt
    prompt_text = (
        f"{DEFAULT_IMAGE_TOKEN}\n"
        "Within 60 tokens, describe the person's **facial expression**, with brief mention of relevant **gestures** and **background context**. Use concise, physical descriptions only. "
        "Avoid emotion words (e.g., 'happy', 'neutral', 'angry', 'surprised', 'concerned') and avoid any interpretive language (e.g., 'suggesting', 'implying', 'indicates')."
    )

    captions = []
    for i in range(NUM_CAPTIONS):
        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors="pt").unsqueeze(0).to(DEVICE)

        outputs = model.generate(
            input_ids,
            images=video_batch,
            modalities=["video"],
            do_sample=True,
            max_new_tokens=60,
            temperature=1.0,
            top_p=0.95,
        )

        text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        caption = text.split(conv.roles[1])[-1].strip()
        print(f"Caption {i+1}: {caption}")
        captions.append(caption)

    # append row without human caption
    results.append([video_file] + captions)

# save results 
df_out = pd.DataFrame(results, columns=[
    "video", "llava_caption_1", "llava_caption_2", "llava_caption_3", "llava_caption_4", "llava_caption_5"
])
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\nCaptions saved to: {OUTPUT_CSV}")
