import json
from datetime import datetime

def generate_handoff_summary(persona: str, issue: str, conversation_history: list,
                               documents_used: list, attempted_steps: list, escalation_reason: str):
    summary = {
        "timestamp": datetime.now().isoformat(),
        "persona": persona,
        "issue": issue,
        "escalation_reason": escalation_reason,
        "conversation_history": conversation_history,
        "documents_used": documents_used,
        "attempted_steps": attempted_steps,
        "recommendation": get_recommendation(persona, issue, escalation_reason)
    }
    return json.dumps(summary, indent=2)


def get_recommendation(persona: str, issue: str, escalation_reason: str) -> str:
    issue_lower = issue.lower()

    if "hack" in issue_lower or "fraud" in issue_lower or "compromised" in issue_lower:
        return "Immediate security team review required. Verify identity before any account changes."
    if "billing" in issue_lower or "refund" in issue_lower or "legal" in issue_lower:
        return "Transfer to billing/legal specialist for manual review."
    if "low retrieval confidence" in escalation_reason.lower() or "no relevant information" in escalation_reason.lower():
        return "Knowledge base does not cover this issue. Escalate to Tier-2 support for manual handling and consider adding documentation."
    if persona == "Frustrated User":
        return "Prioritize quick response. Customer has had a negative experience — consider goodwill gesture."
    return "Escalate to Tier-2 support with full conversation context."


if __name__ == "__main__":
    summary = generate_handoff_summary(
        persona="Frustrated User",
        issue="Unable to reset password after multiple attempts",
        conversation_history=["I can't log in", "I tried resetting password but nothing works"],
        documents_used=["password_reset.txt", "troubleshooting_login.txt"],
        attempted_steps=["Password reset", "Browser cache clear"],
        escalation_reason="User remains unsatisfied after multiple interactions"
    )
    print(summary)