import torch
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
bert_model = SentenceTransformer("all-mpnet-base-v2").to(device)
import pickle


def save_embeddings(embeddings, filename="embeddings.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(embeddings, f)


def load_embeddings(filename="embeddings.pkl"):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}


def encode_captions(captions, cache_file="embeddings.pkl"):
    encoded_captions = load_embeddings(cache_file)

    for image_id, caption in captions.items():
        if image_id not in encoded_captions:  # Chỉ encode nếu chưa có
            encoded_captions[image_id] = {
                "embed": bert_model.encode(caption).tolist(),
                "text": caption
            }

    save_embeddings(encoded_captions, cache_file)
    return encoded_captions
