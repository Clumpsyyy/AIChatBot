import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("modules.json", "r", encoding="utf-8") as f:
    modules = json.load(f)

embeddings_data = [] 

for m in modules:
    depends_list = m.get("depends") or [] 
    text = f"""
    Module: {m.get('name')}
    Summary: {m.get('summary')}
    Depends: {', '.join(depends_list)}
    """

    embedding = model.encode(text)

    embeddings_data.append({
        "module": m.get("name"),
        "text": text,
        "embedding": embedding.tolist()  
    })

    print(m.get('name'), len(embedding))

with open("module_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(embeddings_data, f, indent=4)