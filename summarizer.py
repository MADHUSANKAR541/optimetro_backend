import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # Load GROQ_API_KEY

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def llm_summarize(reasons: list, train_id: str = None, decision: str = None) -> str:
    if not reasons:
        return "No reasoning provided."

    prompt = (
        f"Train ID: {train_id or 'Unknown'}\n"
        f"Decision: {decision or 'Unknown'}\n"
        f"Reasons:\n- " + "\n- ".join(reasons) + "\n\n"
        "Please summarize the reasons provided in a clear, concise, and human-readable explanation. Act as an explainable AI and add relevant context to your summary. Respond only in plain text, without using markdown, tables, bullet points, or any escape characters such as \n. Present each category as key: value pairs, separated by commas or semicolons, using comprehensive but not too long explanations (max 5 lines)."
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-20b",
            stream=False,
        )
        summary = chat_completion.choices[0].message.content
        return summary
    except Exception as e:
        # Return mock summary if LLM fails
        return "Summary unavailable, fallback: " + ", ".join(reasons[:3]) + "..."
