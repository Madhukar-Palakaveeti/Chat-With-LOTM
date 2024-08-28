import cohere
import os
import embedding

cohere_api_key = os.environ.get('COHERE_API_KEY ')

co = cohere.Client(cohere_api_key)

def get_context(query):
    with open('resources/paragraphs.txt', 'r', encoding='utf-8') as file:
        paragraphs = file.readlines()
    scores, indices = embedding.similarity_search(query=query, embeddings_file_path='resources/embeddings.pt', top_k=10)
    relevant_docs = embedding.get_relevant_docs(indices=indices, paragraphs=paragraphs)

    response = co.rerank(
        model = "rerank-english-v3.0",
        query = query,
        documents=relevant_docs,
        top_n=3,
        return_documents=True
    )

    context = "\n".join(i.document.text for i in response.results)

    return context