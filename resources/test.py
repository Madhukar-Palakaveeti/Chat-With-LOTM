import torch
embeddings = torch.load('resources/error_embeddings.pt',map_location=torch.device('cpu'))
print(len(embeddings[0]))