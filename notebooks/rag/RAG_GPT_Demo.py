#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Auto-generated from RAG_GPT_Demo.ipynb

# ===== Cell 1 (markdown) =====
# Preprocessing

# ===== Cell 3 (code) =====
pdf_folder_path = "HIV_guidelines" #clinical document location

# ===== Cell 4 (code) =====
from langchain_community.document_loaders import PyPDFDirectoryLoader
import json

loader = PyPDFDirectoryLoader(pdf_folder_path)
dataset = loader.load()

#from langchain_community.document_loaders import PyPDFLoader
#import json

#pdf_path = "guidelines-12.0.pdf"
#loader = PyPDFLoader(pdf_path)
#dataset = loader.load()

# ===== Cell 5 (code) =====
data = []
for doc in dataset:
    data.append({
        "reference": doc.metadata.get("source", ""),
        "page": doc.metadata.get("page", None),
        "text": doc.page_content,
    })

# ===== Cell 6 (code) =====
import tiktoken

tokenizer = tiktoken.get_encoding('cl100k_base')

# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

# ===== Cell 7 (code) =====
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)

# ===== Cell 8 (code) =====
from uuid import uuid4
from tqdm.auto import tqdm

chunks = []
global_chunk = 0

for record in tqdm(data):
    texts = text_splitter.split_text(record["text"])
    for i, t in enumerate(texts):
        chunks.append({
            "id": str(uuid4()),
            "text": t,
            "reference": record["reference"],
            "page": record.get("page"),
            "chunk_in_page": i,
            "chunk_global": global_chunk,
        })
        global_chunk += 1

# --- Inspect chunking (run after `chunks` is created) ---

import random
from collections import Counter

print(f"Loaded docs: {len(dataset)}")
print(f"Total chunks: {len(chunks)}")

# Token length stats
token_lengths = [tiktoken_len(c["text"]) for c in chunks]
print("Token length stats:")
print(f"  min: {min(token_lengths)}")
print(f"  p50: {sorted(token_lengths)[len(token_lengths)//2]}")
print(f"  p90: {sorted(token_lengths)[int(len(token_lengths)*0.90)]}")
print(f"  max: {max(token_lengths)}")

# Chunk count per source/reference
by_ref = Counter([c["reference"] for c in chunks])
print("\nTop references by chunk count:")
for ref, n in by_ref.most_common(5):
    print(f"  {n:>5}  {ref}")

def show_chunk(i: int, preview_chars: int = 800):
    c = chunks[i]
    print("\n" + "="*100)
    print(f"Chunk idx: {i}")
    print(f"id: {c['id']}")
    print(f"reference: {c['reference']}")
    print(f"chunk_global: {c['chunk_global']} | chunk_in_page: {c['chunk_in_page']} | page: {c.get('page')}")
    print(f"chars: {len(c['text'])} | tokens: {tiktoken_len(c['text'])}")
    print("-"*100)
    print(c["text"][:preview_chars].replace("\n", "\\n"))
    if len(c["text"]) > preview_chars:
        print("...")

# Show first few chunks
for i in range(min(3, len(chunks))):
    show_chunk(i)

# Show a few random chunks
for i in random.sample(range(len(chunks)), k=min(5, len(chunks))):
    show_chunk(i)

# Optional: check overlap continuity for a random document/reference
ref = random.choice(list(by_ref.keys()))
ref_chunks = [c for c in chunks if c["reference"] == ref]
ref_chunks_sorted = sorted(ref_chunks, key=lambda x: x["chunk_global"])

print("\n" + "="*100)
print(f"Overlap sanity check for reference:\n{ref}")
if len(ref_chunks_sorted) >= 2:
    a = ref_chunks_sorted[0]["text"]
    b = ref_chunks_sorted[1]["text"]
    # show last 200 chars of chunk 0 and first 200 chars of chunk 1
    print("\n[Chunk 0 tail]")
    print(a[-200:].replace("\n", "\\n"))
    print("\n[Chunk 1 head]")
    print(b[:200].replace("\n", "\\n"))
else:
    print("Not enough chunks for overlap check.")

# Optional: export chunks to a JSONL file for easy inspection
out_path = "chunks_preview.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for c in chunks[:200]:  # first 200 chunks (adjust)
        f.write(json.dumps({
            "id": c["id"],
            "reference": c["reference"],
            "page": c.get("page"),
            "chunk_in_page": c.get("chunk_in_page"),
            "chunk_global": c.get("chunk_global"),
            "tokens": tiktoken_len(c["text"]),
            "text": c["text"]
        }, ensure_ascii=False) + "\n")


print(f"\nWrote preview file: {out_path} (first 200 chunks)")

assert len(chunks) > 0
assert all("text" in c and len(c["text"].strip()) > 0 for c in chunks)
assert all("reference" in c for c in chunks)

print("Chunks ready:", len(chunks))


# ===== Cell 9 (markdown) =====
# Embedding Model

# ===== Cell 10 (code) =====
import openai

#openai.api_key = ""  #OpenAI API Key

embed_model = "text-embedding-3-small"

# ===== Cell 11 (markdown) =====
# Vector Storage

# ===== Cell 12 (code) =====

import os
from pinecone import Pinecone, ServerlessSpec

proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") or os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")

pc = Pinecone(
    proxy_url=proxy,
    api_key=os.environ["PINECONE_API_KEY"]
)

index_name = "hiv-guidelines-emb3small"

# List indexes (new API)
existing = pc.list_indexes().names()

# Create index if missing (serverless spec required)
if index_name not in existing:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",   # free tier region
        )
    )

index = pc.Index(index_name)

# Optional: stats
print(index.describe_index_stats())


# ===== Cell 13 (code) =====
from tqdm.auto import tqdm
import datetime
from time import sleep
from openai import OpenAI
client = OpenAI()

batch_size = 100

for i in tqdm(range(0, len(chunks), batch_size)):
    i_end = min(len(chunks), i+batch_size)
    meta_batch = chunks[i:i_end]
    ids_batch = [x['id'] for x in meta_batch]
    texts = [x['text'] for x in meta_batch]
    try:
        res = client.embeddings.create(input=texts, model=embed_model)
    except:
        done = False
        while not done:
            sleep(5)
            try:
                res = client.embeddings.create(input=texts, model=embed_model)
                done = True
            except:
                pass
    embeds = [record.embedding for record in res.data]
    meta_batch = [{
        "text": x["text"],
        "reference": x["reference"],
        "page": x.get("page"),
        "chunk_in_page": x.get("chunk_in_page"),
        "chunk_global": x.get("chunk_global"),
    } for x in meta_batch]
    to_upsert = list(zip(ids_batch, embeds, meta_batch))
    index.upsert(vectors=to_upsert)


# ===== Cell 14 (markdown) =====
# Retrieval Agent

# ===== Cell 15 (code) =====
# Reuse pc + index
index = pc.Index(index_name)
print(index.describe_index_stats())

# ===== Cell 16 (code) =====
from openai import OpenAI
client = OpenAI()

query = str("A 45-year-old man comes to the emergency department because of chills and numerous skin lesions for 1 week. He has also had watery diarrhea, nausea, and abdominal pain for the past 2 weeks. The skin lesions are nonpruritic and painless. He was diagnosed with HIV infection approximately 20 years ago. He has not taken any medications for over 5 years. He sleeps in homeless shelters and parks. Vital signs are within normal limits. Examination shows several bright red, friable nodules on his face, trunk, extremities. The liver is palpated 3 cm below the right costal margin. His CD4+ T-lymphocyte count is 180/mm3 (N ≥ 500). A rapid plasma reagin test is negative. Abdominal ultrasonography shows hepatomegaly and a single intrahepatic 1.0 x 1.2-cm hypodense lesion. Biopsy of a skin lesion shows vascular proliferation and abundant neutrophils. What is the most likely causal organism?") #clinical query

res = client.embeddings.create(
    input=[query],
    model=embed_model
)

xq = res.data[0].embedding
res = index.query(vector=xq, top_k=4, include_metadata=True)

# ===== Cell 17 (markdown) =====
# Response Generation

# ===== Cell 18 (code) =====
contexts = [item['metadata']['text'] for item in res['matches']]
augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+query

# ===== Cell 19 (code) =====
print(augmented_query)

# ===== Cell 20 (markdown) =====
# LLM Integration (GPT 4)

# ===== Cell 21 (code) =====
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
  model="gpt-4",
  messages=[
    {"role": "system", "content": ""}, #System Prompt
    {"role": "user", "content": augmented_query},
  ]
)

# ===== Cell 22 (code) =====
print(response.choices[0].message.content)
