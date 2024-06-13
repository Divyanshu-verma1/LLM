## Required Libraries
from sentence_transformers import SentenceTransformer
import faiss
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
import requests
from bs4 import BeautifulSoup
import re
import torch

# Function to download and parse lecture notes
def get_lecture_notes(urls):
    lectures = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.find_all('p')
        text = "\n".join([p.text for p in content])
        lectures.append(text)
    return lectures

# Function to get model architectures
def get_model_architectures(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('tbody')
    rows = table.find_all('tr')
    architectures = []
    for row in rows:  # skip header row
        cols = row.find_all('td')
        architectures.append(" ".join([col.text.strip() for col in cols]))
    return architectures

# URLs for lecture notes and model architectures
lecture_urls = [
        'https://stanford-cs324.github.io/winter2022/lectures/introduction',
        'https://stanford-cs324.github.io/winter2022/lectures/capabilities',
        'https://stanford-cs324.github.io/winter2022/lectures/harms-1',
        'https://stanford-cs324.github.io/winter2022/lectures/harms-2',
        'https://stanford-cs324.github.io/winter2022/lectures/data',
        'https://stanford-cs324.github.io/winter2022/lectures/security',
        'https://stanford-cs324.github.io/winter2022/lectures/legality',
        'https://stanford-cs324.github.io/winter2022/lectures/modeling',
        'https://stanford-cs324.github.io/winter2022/lectures/training'
    # Add other lecture URLs as needed
]
model_architectures_url = 'https://github.com/Hannibal046/Awesome-LLM/blob/main/README.md'

# Downloading data
lectures = get_lecture_notes(lecture_urls)
model_architectures = get_model_architectures(model_architectures_url)

# Combine and prepare data
documents = lectures + model_architectures

# Embedding and Indexing
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

## Query Handling
def generate_response(query, k=3, max_tokens=1000, chunk_size=512):
    query_embedding = model.encode([query])[0]
    _, indices = index.search(query_embedding.reshape(1, -1), k)
    relevant_docs = [documents[idx] for idx in indices[0]]

    combined_text = "\n\n".join(relevant_docs)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

    tokens = tokenizer.encode(combined_text, return_tensors='pt')
    chunked_texts = [tokens[0, i:i + chunk_size] for i in range(0, tokens.size(1), chunk_size)]

    generated_text = ""
    text_generation_pipeline = pipeline('text-generation', model=gpt2_model, tokenizer=tokenizer)

    for chunk in chunked_texts:
        input_text = tokenizer.decode(chunk, skip_special_tokens=True)
        response = text_generation_pipeline(input_text, max_length=len(input_text.split()) + chunk_size, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, truncation=True)
        new_text = response[0]['generated_text'].strip()
        generated_text += " " + new_text

        if len(generated_text.split()) >= max_tokens:
            break

    return generated_text.strip()

# Example Query
query = "What are some milestone model architectures and papers in the last few years?"
response = generate_response(query, max_tokens=1000)
print(response)