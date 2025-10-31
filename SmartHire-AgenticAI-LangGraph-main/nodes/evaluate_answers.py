"""
Answer Evaluation Node
Evaluates student answers using semantic search and Gemini AI.
"""

import os
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv
from state import CandidateState
from utils.semantic_search import evaluate_semantic_similarity

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY_4"))
gemini_model = genai.GenerativeModel("gemini-2.0-flash")


def clean_json_response(text: str) -> str:
    """Clean Gemini JSON response"""
    text = text.strip()
    text = re.sub(r"```json|```", "", text)
    text = text.replace("'", '"')
    match = re.search(r"{.*}", text, re.DOTALL)
    return match.group(0) if match else '{"score": 0, "feedback": "Evaluation failed"}'


def evaluate_answer(question: dict, student_answer: str) -> dict:
    """Evaluate a single answer"""

    if question["type"] == "concept":
        # Use semantic search for concept questions
        evaluation = evaluate_semantic_similarity(
            question["reference_answer"],
            student_answer
        )
        return {
            "score": evaluation["score"],
            "feedback": evaluation["feedback"]
        }

    else:  # code questions
        # Use Gemini for code evaluation
        prompt = f"""
Evaluate this coding answer. Return JSON only:
{{"score": int (0-10), "feedback": "brief feedback"}}

Question: {question["question"]}
Reference: {question["reference_answer"]}
Student Answer: {student_answer}
"""
        try:
            response = gemini_model.generate_content(prompt)
            result = json.loads(clean_json_response(response.text))
            score = float(result.get("score", 0))
            feedback = result.get("feedback", "No feedback")
        except:
            score, feedback = 0.0, "Evaluation failed"

    return {"score": score, "feedback": feedback}


def evaluate_answers(state: CandidateState) -> CandidateState:
    """Evaluate all submitted answers"""

    try:
        total_score = 0
        evaluated_answers = []

        for i, answer_data in enumerate(state["answers"]):
            if i < len(state["questions"]):
                question = state["questions"][i]
                evaluation = evaluate_answer(question, answer_data["answer"])

                evaluated_answers.append({
                    "question_id": question["id"],
                    "answer": answer_data["answer"],
                    "score": evaluation["score"],
                    "feedback": evaluation["feedback"]
                })

                total_score += evaluation["score"]

        state["final_score"] = round(total_score / len(evaluated_answers), 2) if evaluated_answers else 0
        state["answers"] = evaluated_answers
        state["current_step"] = "completed"

    except Exception as e:
        state["errors"].append(f"Answer evaluation failed: {str(e)}")
        state["final_score"] = 0.0

    return state
