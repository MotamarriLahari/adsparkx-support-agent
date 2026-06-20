ESCALATION_KEYWORDS = [
    "billing dispute", "legal", "lawsuit", "refund denied",
    "account hacked", "fraud", "speak to a manager", "supervisor",
    "cancel my account", "sue", "compromised"
]

LOW_CONFIDENCE_THRESHOLD = 1.2  # ChromaDB distance score; higher = less relevant

def should_escalate(user_message: str, retrieved_results: list, turn_count: int = 1, unresolved: bool = False):
    """
    Returns: (should_escalate: bool, reason: str)
    """
    msg_lower = user_message.lower()

    # Trigger 1: Sensitive/legal/billing keywords
    for kw in ESCALATION_KEYWORDS:
        if kw in msg_lower:
            return True, f"Sensitive topic detected: contains '{kw}'"

    # Trigger 2: Low retrieval confidence (no good match found)
    if retrieved_results:
        best_score = min(score for _, score in retrieved_results)
        if best_score > LOW_CONFIDENCE_THRESHOLD:
            return True, f"Low retrieval confidence (best match score: {best_score:.2f})"
    else:
        return True, "No relevant information found in knowledge base"

    # Trigger 3: User remains unresolved after multiple turns
    if turn_count >= 3 and unresolved:
        return True, "User remains unsatisfied after multiple interactions"

    return False, ""


if __name__ == "__main__":
    from src.rag_pipeline import retrieve

    test_cases = [
        "I want to file a legal complaint about my account being hacked",
        "How do I reset my password?",
        "asdkjaslkdj random gibberish query xyz123"
    ]

    for msg in test_cases:
        results = retrieve(msg)
        escalate, reason = should_escalate(msg, results)
        print(f"Message: {msg}")
        print(f"Escalate: {escalate} | Reason: {reason}")
        print("-" * 50)