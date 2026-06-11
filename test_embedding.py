from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cuda')
emb = model.encode('test')
print(f'Embedding size: {emb.shape}, Device: {emb.device}')