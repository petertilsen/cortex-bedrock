import json
import os
import time
from collections import defaultdict

import numpy as np
import tiktoken
from dotenv import load_dotenv
from jinja2 import Template
import boto3
from tqdm import tqdm

load_dotenv()

PROMPT = """
# Question: 
{{QUESTION}}

# Context: 
{{CONTEXT}}

# Short answer:
"""


class RAGManager:
    def __init__(self, data_path="dataset/locomo10_rag.json", chunk_size=500, k=1):
        self.model = os.getenv("MODEL", "anthropic.claude-4-sonnet-20241022-v2:0")
        self.client = boto3.client('bedrock-runtime', region_name='us-east-1')
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.k = k

    def generate_response(self, question, context):
        template = Template(PROMPT)
        prompt = template.render(
            CONTEXT=context,
            QUESTION=question
        )

        max_retries = 3
        retries = 0

        while retries <= max_retries:
            try:
                t1 = time.time()
                
                messages = [{
                    "role": "user", 
                    "content": "You are a helpful assistant that can answer questions based on the provided context. "
                              "If the question involves timing, use the conversation date for reference. "
                              "Provide the shortest possible answer. "
                              "Use words directly from the conversation when possible. "
                              f"Avoid using subjects in your answer.\n\n{prompt}"
                }]
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "temperature": 0,
                    "messages": messages
                }
                
                response = self.client.invoke_model(
                    modelId=self.model,
                    body=json.dumps(body)
                )
                
                response_body = json.loads(response['body'].read())
                content = response_body['content'][0]['text']
                t2 = time.time()
                return content.strip(), t2-t1
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    raise e
                time.sleep(1)  # Wait before retrying

    def clean_chat_history(self, chat_history):
        cleaned_chat_history = ""
        for c in chat_history:
            cleaned_chat_history += (f"{c['timestamp']} | {c['speaker']}: "
                                     f"{c['text']}\n")

        return cleaned_chat_history

    def calculate_embedding(self, document):
        body = {"inputText": document}
        response = self.client.invoke_model(
            modelId=os.getenv("EMBEDDING_MODEL", "amazon.titan-embed-text-v1"),
            body=json.dumps(body)
        )
        response_body = json.loads(response['body'].read())
        return response_body['embedding']

    def calculate_similarity(self, embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def search(self, query, chunks, embeddings, k=1):
        """
        Search for the top-k most similar chunks to the query.

        Args:
            query: The query string
            chunks: List of text chunks
            embeddings: List of embeddings for each chunk
            k: Number of top chunks to return (default: 1)

        Returns:
            combined_chunks: The combined text of the top-k chunks
            search_time: Time taken for the search
        """
        t1 = time.time()
        query_embedding = self.calculate_embedding(query)
        similarities = [
            self.calculate_similarity(query_embedding, embedding) 
            for embedding in embeddings
        ]

        # Get indices of top-k most similar chunks
        if k == 1:
            # Original behavior - just get the most similar chunk
            top_indices = [np.argmax(similarities)]
        else:
            # Get indices of top-k chunks
            top_indices = np.argsort(similarities)[-k:][::-1]

        # Combine the top-k chunks
        combined_chunks = "\n<->\n".join([chunks[i] for i in top_indices])

        t2 = time.time()
        return combined_chunks, t2-t1

    def create_chunks(self, chat_history, chunk_size=500):
        """
        Create chunks using tiktoken for accurate token counting
        """
        # Use cl100k_base encoding (compatible with most modern models)
        encoding = tiktoken.get_encoding("cl100k_base")

        documents = self.clean_chat_history(chat_history)

        if chunk_size == -1:
            return [documents], []

        chunks = []

        # Encode the document
        tokens = encoding.encode(documents)

        # Split into chunks based on token count
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i+chunk_size]
            chunk = encoding.decode(chunk_tokens)
            chunks.append(chunk)

        embeddings = []
        for chunk in chunks:
            embedding = self.calculate_embedding(chunk)
            embeddings.append(embedding)

        return chunks, embeddings

    def process_all_conversations(self, output_file_path):
        with open(self.data_path, "r") as f:
            data = json.load(f)

        FINAL_RESULTS = defaultdict(list)
        for key, value in tqdm(data.items(), desc="Processing conversations"):
            chat_history = value["conversation"]
            questions = value["question"]

            chunks, embeddings = self.create_chunks(
                chat_history, self.chunk_size
            )

            for item in tqdm(
                questions, desc="Answering questions", leave=False
            ):
                question = item["question"]
                answer = item.get("answer", "")
                category = item["category"]

                if self.chunk_size == -1:
                    context = chunks[0]
                    search_time = 0
                else:
                    context, search_time = self.search(
                        question, chunks, embeddings, k=self.k
                    )
                response, response_time = self.generate_response(
                    question, context
                )

                FINAL_RESULTS[key].append({
                    "question": question,
                    "answer": answer,
                    "category": category,
                    "context": context,
                    "response": response,
                    "search_time": search_time,
                    "response_time": response_time,
                })
                with open(output_file_path, "w+") as f:
                    json.dump(FINAL_RESULTS, f, indent=4)

        # Save results
        with open(output_file_path, "w+") as f:
            json.dump(FINAL_RESULTS, f, indent=4)
