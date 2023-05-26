import openai
import numpy as np
import dotenv
import os

# Load env vars
dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_cosine_similarity(word1, word2):
    # Create embeddings
    response1 = openai.Embedding.create(
        input=word1,
        engine="text-embedding-ada-002"
    )

    response2 = openai.Embedding.create(
        input=word2,
        engine="text-embedding-ada-002"
    )

    # Extract embeddings
    embedding1 = response1['data'][0]['embedding']
    embedding2 = response2['data'][0]['embedding']

    # Calculate cosine similarity; embedding1 and embedding2 are already normalized
    # no need to divide by the norm of the embeddings
    similarity_score = np.dot(embedding1, embedding2)

    # convert embedding1 and embedding2 to numpy arrays
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)

    # Calculate euclidian distance
    euclidian_distance = np.linalg.norm(embedding1 - embedding2)    


    return similarity_score, euclidian_distance

# Example usage
word1 = "Man"
word2 = "Big Man"
similarity_score, euclidian_distance = get_cosine_similarity(word1, word2)
print(f'The cosine similarity between "{word1}" and "{word2}" is: {similarity_score}')
print(f'The euclidian distance between "{word1}" and "{word2}" is: {euclidian_distance}')