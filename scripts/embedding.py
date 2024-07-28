from sentence_transformers import SentenceTransformer
import torch

MODEL_NAME = "nq-distilbert-base-v1"
bi_encoder = SentenceTransformer(MODEL_NAME)

def get_embeddings(paragraphs):
    ''' Get the embeddings for paragraphs or chunks. The embeddings are a list of tensors.'''
    embeddings = bi_encoder.encode(paragraphs, convert_to_tensor=True, show_progress_bar=True)
    return embeddings

def save_embeddings(embeddings, embeddings_path):
    ''' Save the embeddings to a file'''
    with open(embeddings_path, 'wb') as f:
            torch.save(embeddings, f)
def similarity_search(query : str, embeddings_file_path, top_k):
    '''Converts the query to an embedding in the same vector space and does a cosine similarity search.
        Returns similarity scores and indices of the top-k similar items.
    '''
    query_embed = bi_encoder.encode(query, convert_to_tensor=True)
    embeddings = torch.load(embeddings_file_path,map_location=torch.device('cpu'))
    
    similarity_scores = bi_encoder.similarity(query_embed, embeddings)[0]
    return torch.topk(similarity_scores, k=top_k)

def get_context(indices, paragraphs):
    return "\n".join(paragraphs[i] for i in indices)