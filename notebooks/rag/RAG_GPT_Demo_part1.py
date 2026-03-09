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
        'reference': doc.metadata['source'].replace('rtdocs/', 'https://'),
        'text': doc.page_content
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
    chunk_size=1200,
    chunk_overlap=150,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)

# ===== Cell 8 (code) =====
from uuid import uuid4
from tqdm.auto import tqdm

chunks = []

for idx, record in enumerate(tqdm(data)):
    texts = text_splitter.split_text(record['text'])
    chunks.extend([{
        'id': str(uuid4()),
        'text': texts[i],
        'chunk': i,
        'reference': record['reference']
    } for i in range(len(texts))])

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
    print(f"chunk #: {c['chunk']}")
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
ref_chunks_sorted = sorted(ref_chunks, key=lambda x: x["chunk"])

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
            "chunk": c["chunk"],
            "tokens": tiktoken_len(c["text"]),
            "text": c["text"]
        }, ensure_ascii=False) + "\n")

print(f"\nWrote preview file: {out_path} (first 200 chunks)")

assert len(chunks) > 0
assert all("text" in c and len(c["text"].strip()) > 0 for c in chunks)
assert all("reference" in c for c in chunks)

print("Chunks ready:", len(chunks))
