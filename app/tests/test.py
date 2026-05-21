from app.core.pipeline import embed_document

text = """
AI is smart. Students use AI. Learning improves. 
Deep learning is powerful. Transformers understand context.
"""

result = embed_document(text)

print("Chunks:")
print(result["chunks"])

print("\nFirst embedding preview:")
print(result["embeddings"][0][:5])