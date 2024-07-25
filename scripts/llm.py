import embedding

with open('resources/paragraphs.txt', 'r', encoding='utf-8') as file:
    paragraphs = file.readlines()

paragraphs = paragraphs[:93524]

embeddings = embedding.get_embeddings(paragraphs)
embedding.save_embeddings(embeddings=embeddings, embeddings_path='resources/embeddings.pt')

query = "Who/What is susie?"
scores, indices = embedding.similarity_search(query=query, embeddings_file_path='resources/embeddings.pt', top_k=5)
context = embedding.get_context(indices=indices, paragraphs=paragraphs)

print(context)