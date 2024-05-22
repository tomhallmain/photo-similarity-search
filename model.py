from config import config

mlx_imported = False

try:
    import mlx_clip
    mlx_imported = True
except Exception:
    print("MLX clip failed to import, will fall back to OpenAI's python CLIP implementation.")
    from PIL import Image
    import torch
    import clip

if mlx_imported:
    #Instantiate MLX Clip model
    model = mlx_clip.mlx_clip("mlx_model", hf_repo=config.CLIP_MODEL)
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(config.CLIP_MODEL, device=device)

def generate_embedding_for_frame(frame):
    if mlx_imported:
        return model.image_encoder(frame)
    else:
        image = preprocess(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            return model.encode_image(image)[0]

def image_embeddings(image_path):
    if image_path.endswith(".gif") or image_path.endswith(".GIF"):
        gif = Image.open(image_path)
        embeddings = []
        for frame_index in range(gif.n_frames):
            gif.seek(frame_index)
            frame_embedding = generate_embedding_for_frame(gif)
            embeddings.append(frame_embedding)
        embedding = torch.cat(embeddings, dim=0)
    else:
        embedding = generate_embedding_for_frame(Image.open(image_path))
    return embedding.tolist()


def text_embeddings(text):
    if mlx_imported:
        return model.text_encoder(text)
    else:
        tokens = clip.tokenize([text]).to(device)
        with torch.no_grad():
            return model.encode_text(tokens)[0].tolist()
