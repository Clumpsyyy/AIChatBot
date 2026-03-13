import json
import chromadb
from sentence_transformers import SentenceTransformer
# import google.generativeai as genai
import os
import ollama

# ollama = (api_key="30f9d1cc57d84e2db83c5f8d76d85383.JKJZCH3nS9uKDZ2kPPnA9FVw")
# from google import genai

# # client = genai.Client()
# gemini_client = genai.Client(api_key="AIzaSyBnU8VYnjXlVBT_6KajJRmypLk4tBIM5jU")
# print(dir(client))
client = chromadb.Client()
collection_name = "odoo_modules"

# Create or get collection
if collection_name in [c.name for c in client.list_collections()]:
    collection = client.get_collection(collection_name)
else:
    collection = client.create_collection(name=collection_name)


if len(collection.get(include=["documents"])["documents"]) == 0:
    with open("module_embeddings.json", "r", encoding="utf-8") as f:
        embeddings_data = json.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    for module in embeddings_data:
        collection.add(
            documents=[module["text"]],
            embeddings=[module["embedding"]],
            ids=[module["module"]]
        )

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Interactive loop
print("Type your inquiries here (type 'exit' to quit)")

while True:
    query = input("\nAsk a question: ")
    if query.lower() in ["exit", "quit"]:
        break


    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
        )

    context_text = "\n\n".join(results["documents"][0])

    # prompt = f"""
    # You are an expert Odoo consultant. Use the following module information to answer the user's question in clear language.

    # Module data:
    # {context_text}

    # User question: {query}

    # Answer:
    # """

    # response = gemini_client.models.generate_content(
    # model="gemini-2.0-flash",
    # contents=prompt
    # )
    # answer = response.text

    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "system",
                 "content": 
                (
                "You are a professional Odoo ERP consultant. "
                "Answer questions clearly and professionally. "
                "Your name is OdooChan. "
                "Keep responses concise (maximum 120 words). "
                "Provide practical advice and examples when relevant. "
                "Do not repeat module text."
                "Do not answer if you are unsure, clarify to the user"
                "Do not repeat your name if you already say that"
                )
            },
            {"role": "user",
            "content": f"""
    Use the following module information to answer the question professionally:

    Module data:
    {context_text}

    User question: {query}
    """}
        ]
    )

    answer = response["message"]["content"]
    print(answer)
