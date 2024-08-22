import embedding
from groq import Groq
import os

with open('resources/paragraphs.txt', 'r', encoding='utf-8') as file:
    paragraphs = file.readlines()


groq_api_key = os.environ.get('GROQ_API_KEY')
query = "What are the sequences of the error pathway?"

scores, indices = embedding.similarity_search(query=query, embeddings_file_path='resources/embeddings.pt', top_k=5)
context = embedding.get_context(indices=indices, paragraphs=paragraphs)

client = Groq(
    api_key=groq_api_key,
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": f"""
                  You are a novel reader who've completed the novel and now conversing with a fellow novel reader casually and answering some queries. You are provided with some context which is enough to answer the queries.
                    Query : {query}
                    Context : {context}
            Answer with as many details as possible and as relevant data as possible, with concise description. Also gather everything from the context and present it as a if you've read the whole novel rather than based on the scene. Also dont mention the characters appearing in the context, just try to integrate their stories into your answer rather than directly repeating the scene."""
        }
    ],
    model="llama3-8b-8192",
)

print(chat_completion.choices[0].message.content)