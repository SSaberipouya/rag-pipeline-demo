from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_data(path):
    with open(path, "r") as f:
        text = f.read()
    return text.split("\n")

def create_embeddings(chunks):
    return model.encode(chunks)