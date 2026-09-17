
import json
from pathlib import Path

from retriever import retrieve


# Paths
PROJECT_DIR = Path(__file__).parent.parent
EVALUATION_FILE = PROJECT_DIR / "rag_data" / "evaluation_questions.json"


# Load evaluation questions
with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
    evaluation_data = json.load(file)


def precision_at_k(retrieved_documents, relevant_documents, k):


    # Take only the top K retrieved documents
    top_k_documents = retrieved_documents[:k]

    # Find documents that are both:
    # 1. retrieved
    # 2. actually relevant
    relevant_retrieved = set(top_k_documents) & set(relevant_documents)

    # Calculate precision
    precision = len(relevant_retrieved) / k

    return precision


def recall_at_k(retrieved_documents, relevant_documents, k):
    """
    Calculate Recall@K.

    Recall@K =
    Number of relevant documents retrieved in top K
    ------------------------------------------------
              Total number of relevant documents
    """

    # Take only the top K retrieved documents
    top_k_documents = retrieved_documents[:k]

    # Find documents that are both retrieved and relevant
    relevant_retrieved = set(top_k_documents) & set(relevant_documents)

    # Calculate recall
    recall = len(relevant_retrieved) / len(relevant_documents)

    return recall


def reciprocal_rank(retrieved_documents, relevant_documents):
    """
    Calculate Reciprocal Rank (RR).

    RR = 1 / rank of the first relevant document
    """


    for rank, document in enumerate(retrieved_documents, start=1):

        if document in relevant_documents:
            return 1 / rank

    # No relevant document was retrieved
    return 0.0

def mean_reciprocal_rank(all_reciprocal_ranks):
    """
    Calculate Mean Reciprocal Rank (MRR).
    """

    if not all_reciprocal_ranks:
        return 0.0

    return sum(all_reciprocal_ranks) / len(all_reciprocal_ranks)
import math


def ndcg_at_k(retrieved_documents, relevant_documents, k):
    # Take only the top K retrieved documents
    top_k_documents = retrieved_documents[:k]

    # Assign relevance scores
    # 1 = relevant
    # 0 = not relevant
    relevance_scores = [
        1 if document in relevant_documents else 0
        for document in top_k_documents
    ]

    # Calculate DCG
    dcg = 0.0

    for rank, relevance in enumerate(relevance_scores, start=1):
        dcg += relevance / math.log2(rank + 1)

    # Calculate ideal relevance scores
    ideal_scores = sorted(relevance_scores, reverse=True)

    # Calculate IDCG
    idcg = 0.0

    for rank, relevance in enumerate(ideal_scores, start=1):
        idcg += relevance / math.log2(rank + 1)

    # Avoid division by zero
    if idcg == 0:
        return 0.0

    # Calculate NDCG
    return dcg / idcg


if __name__ == "__main__":

    K = 3
    reciprocal_ranks = []

    for item in evaluation_data:

        question = item["question"]
        relevant_documents = item["relevant_documents"]

        # Retrieve documents
        results = retrieve(question, k=K)

        # Extract document names
        retrieved_documents = [
            result["document"]
            for result in results
        ]

        # Precision
        precision = precision_at_k(
            retrieved_documents,
            relevant_documents,
            K
        )

        # Recall
        recall = recall_at_k(
            retrieved_documents,
            relevant_documents,
            K
        )
        rr = reciprocal_rank(
            retrieved_documents,
            relevant_documents
        )
        ndcg = ndcg_at_k(
            retrieved_documents,
            relevant_documents,
            K
        )

        reciprocal_ranks.append(rr)

        print("\nQuestion:", question)
        print(f"RR: {rr:.4f}")

        print("Retrieved:")
        for rank, document in enumerate(retrieved_documents, start=1):
            print(f"  {rank}. {document}")

        print("Relevant:", relevant_documents)

        print(f"Precision@{K}: {precision:.4f}")
        print(f"Recall@{K}: {recall:.4f}")
        print(f"NDCG@{K}: {ndcg:.4f}")
    mrr = mean_reciprocal_rank(reciprocal_ranks)

    print(f"\nMRR: {mrr:.4f}")