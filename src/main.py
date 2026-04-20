from embed import load_data, create_embeddings, model
from retrieve import VectorStore
from generate import generate_answer

# Load data
chunks = load_data("../data/sample.txt")

# Embed
embeddings = create_embeddings(chunks)

# Store
vector_store = VectorStore(embeddings)

# Query
query = "What is hallucination?"
query_embedding = model.encode([query])

# Retrieve
indices = vector_store.search(query_embedding)
retrieved = [chunks[i] for i in indices]

# Generate
context = "\n".join(retrieved)
answer = generate_answer(context, query)

print("Answer:", answer)