from transformers import CLIPTokenizer

# load the tokenizer used by CLIP
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

caption = "The person stands in front of a shelf, wearing a dark blazer with white piping and a high-necked purple garment. The expression is neutral with a slight furrow to the brow. Behind them, shelves hold various decorative objects including a large brown vase, an orange ceramic cat figure, and a"

# tokenise the caption
tokens = tokenizer(caption, return_tensors="pt")
num_tokens = len(tokens.input_ids[0])

# print result
print(f"Token count: {num_tokens}")
if num_tokens > 77:
    print("Too long for CLIP (77-token limit)")
else:
    print("OK for CLIP")

