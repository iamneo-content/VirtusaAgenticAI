"""
Semantic search utility for evaluating answer similarity.
Uses SentenceTransformers for semantic embeddings and cosine similarity.
"""

from sentence_transformers import SentenceTransformer, util

# Initialize SBERT model
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_semantic_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity between two texts.

    Args:
        text1: First text (reference/correct answer)
        text2: Second text (student/submitted answer)

    Returns:
        Similarity score between 0 and 1
    """
    embeddings = sbert_model.encode([text1, text2])
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return similarity


def evaluate_semantic_similarity(reference_answer: str, student_answer: str) -> dict:
    """
    Evaluate similarity between reference and student answers.

    Args:
        reference_answer: The correct/reference answer
        student_answer: The student's submitted answer

    Returns:
        Dictionary with score (0-10) and feedback
    """
    similarity = get_semantic_similarity(reference_answer, student_answer)
    score = round(similarity * 10, 2)
    feedback = f"Semantic similarity: {similarity:.2f}"

    return {
        "score": score,
        "feedback": feedback,
        "similarity": similarity
    }
