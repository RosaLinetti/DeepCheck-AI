from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(embedding1, embedding2):

    similarity = cosine_similarity(
        [embedding1],
        [embedding2]
    )

    return float(similarity[0][0])