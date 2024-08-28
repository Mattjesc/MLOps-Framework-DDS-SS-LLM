import streamlit as st
import openai
import pandas as pd
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

# Set your OpenAI API key
openai.api_key = "YOUR_API_KEY_HERE"

# Mapping of keywords to dataset paths
DATASET_MAPPING = {
    "keyword1": "dataset1",
    "keyword2": "dataset1",
    "keyword3": "dataset1",
    "keyword4": "dataset1",
    "keyword5": "dataset1",
    "keyword6": "dataset1",
    "keyword7": "dataset1",
    "keyword8": "dataset1",
    "keyword9": "dataset1",
    
    "keyword10": "dataset2",
    "keyword11": "dataset2",
    "keyword12": "dataset2",
    "keyword13": "dataset2",
    "keyword14": "dataset2",
    "keyword15": "dataset2",
    "keyword16": "dataset2",
    
    "keyword17": "dataset3",
    "keyword18": "dataset3",
    "keyword19": "dataset3",
    "keyword20": "dataset3",
    "keyword21": "dataset3"
}

# Load dataset based on keyword
def load_dataset(keyword):
    dataset_path = DATASET_MAPPING.get(keyword)
    if not dataset_path:
        raise ValueError(f"No dataset found for keyword: {keyword}")
    return pd.read_csv(dataset_path)

# Semantic search function (example using transformers)
def semantic_search(query, dataset, top_k=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings_model = pipeline("feature-extraction", model="distilbert-base-uncased", device=device)
    
    # Generate embeddings for the query
    query_embedding = embeddings_model(query, truncation=True)
    query_embedding = np.array(query_embedding).mean(axis=1)  # Average over tokens to get a fixed-size embedding
    query_embedding = query_embedding.reshape(1, -1)  # Reshape to 2D array
    
    results = []
    for index, row in dataset.iterrows():
        text = " ".join(str(value) for value in row.values)
        text_embedding = embeddings_model(text, truncation=True)
        text_embedding = np.array(text_embedding).mean(axis=1)  # Average over tokens to get a fixed-size embedding
        text_embedding = text_embedding.reshape(1, -1)  # Reshape to 2D array
        
        similarity = cosine_similarity(query_embedding, text_embedding)[0][0]
        results.append((similarity, text))
    
    results.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in results[:top_k]]

# Augment prompt with retrieved documents
def augment_prompt(query, retrieved_documents):
    augmented_prompt = f"Query: {query}\n\nRelevant Information:\n"
    for doc in retrieved_documents:
        augmented_prompt += f"{doc}\n"
    return augmented_prompt

# Run RAG pipeline
def run_rag_pipeline(query):
    # Extract keywords from query
    query_lower = query.lower()
    matched_keyword = None
    
    for keyword in DATASET_MAPPING.keys():
        if keyword.lower() in query_lower:
            matched_keyword = keyword
            break
    
    if matched_keyword:
        dataset = load_dataset(matched_keyword)
        retrieved_documents = semantic_search(query, dataset)
        augmented_prompt = augment_prompt(query, retrieved_documents)
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": augmented_prompt}
            ]
        )
        
        # Extract and print the assistant's reply
        assistant_reply = response.choices[0].message.content
        st.session_state.chat_history.append({"user": query, "assistant": assistant_reply})
    else:
        # Respond even if no keywords are found
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ]
        )
        assistant_reply = response.choices[0].message.content
        st.session_state.chat_history.append({"user": query, "assistant": assistant_reply})

# Streamlit app
def main():
    st.title("Interactive RAG Pipeline")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    query = st.text_input("Enter your query:")
    if st.button("Submit"):
        if query:
            run_rag_pipeline(query)
        else:
            st.write("Please enter a query.")
    
    for chat in st.session_state.chat_history:
        st.write(f"User: {chat['user']}")
        st.write(f"Assistant: {chat['assistant']}")

if __name__ == "__main__":
    main()