import torch
from sentence_transformers import SentenceTransformer
device = "cuda" if torch.cuda.is_available() else "cpu"
bert_model = SentenceTransformer("all-mpnet-base-v2").to(device)

def encode_captions(captions):
    encoded_captions = {}
    for image_id, caption in captions.items():
        encoded_captions[image_id] = {
            "embed": torch.Tensor(bert_model.encode(caption)),
            "text": caption
        }
    return encoded_captions