from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

def retrieve(query: str, k: int = 4):
    """
    Returns a list of (Document, score) tuples.
    Lower score = more similar (ChromaDB uses distance, not similarity).
    """
    results = db.similarity_search_with_score(query, k=k)
    return results


if __name__ == "__main__":
    test_query = "How do I reset my password?"
    results = retrieve(test_query)

    print(f"Query: {test_query}\n")
    for i, (doc, score) in enumerate(results, 1):
        print(f"Result {i} (score: {score:.4f})")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content: {doc.page_content[:150]}...")
        print("-" * 50)