import embedding
from groq import Groq
import os

with open('resources/paragraphs.txt', 'r', encoding='utf-8') as file:
    paragraphs = file.readlines()

paragraphs = paragraphs[:93524]
groq_api_key = os.environ.get('GROQ_API_KEY')
query = "where does the quote 'The taste of a demoness aint bad' come from?Hint - It has something to do with roselle"
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
            Answer with as many details as possible. Just remember you are a novel reader not someone answering based on the context
            """,
        }
    ],
    model="llama3-8b-8192",
)

print(chat_completion.choices[0].message.content)