import embedding

with open('resources/new.txt', 'r', encoding='utf-8') as file:
    paragraphs = file.readlines()
paragraphs = paragraphs[:93524]


embeddings = embedding.get_embeddings(paragraphs)
embedding.save_embeddings(embeddings=embeddings, embeddings_path='resources/embeddings.pt')