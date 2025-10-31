"""
Job Fit Analysis Node
Analyzes candidate-job compatibility using Gemini AI.
"""

import os
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv
from state import CandidateState
import time

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY_2"))

# Configure model with timeout settings
generation_config = {
    "temperature": 0.1,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 1024,
}

model = genai.GenerativeModel("gemini-2.0-flash", generation_config=generation_config)


def extract_json_from_text(text):
    """Extract JSON from text using simple string operations"""
    text = text.strip()

    # Remove markdown code blocks
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)

    # Find the JSON object
    start = text.find('{')
    if start == -1:
        return None

    # Find matching closing brace
    brace_count = 0
    end = start
    for i, char in enumerate(text[start:], start):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end = i
                break

    if brace_count != 0:
        return None

    json_str = text[start:end+1]
    return json_str


def analyze_job_fit(state: CandidateState) -> CandidateState:
    """Analyze candidate-job compatibility using AI"""

    print("🔄 Starting job fit analysis...")

    # Very simple prompt to avoid JSON issues
    prompt = f"""
Analyze if this candidate fits the job. Respond with valid JSON only:

{{"job_fit": "Fit", "fit_score": 7.5, "reason": "explanation here"}}

Candidate: {state["resume_features"].get("skills", [])[:3]}
Job: {state["job_title"]}

Use "Fit" or "Not Fit" for job_fit. Use score 0-10. Keep reason short.
"""

    try:
        print("📤 Sending job fit request to Gemini...")
        start_time = time.time()

        response = model.generate_content(prompt, request_options={"timeout": 20})

        elapsed_time = time.time() - start_time
        print(f"📥 Received job fit response in {elapsed_time:.2f} seconds")

        # Debug: print raw response
        raw_text = response.text.strip()
        print(f"🔍 Raw response: {raw_text[:150]}...")

        # Extract JSON
        json_str = extract_json_from_text(raw_text)
        if not json_str:
            raise ValueError("No valid JSON found in response")

        print(f"🔍 Extracted JSON: {json_str}")

        # Parse JSON
        job_fit = json.loads(json_str)

        # Validate and fix fields
        if "job_fit" not in job_fit or job_fit["job_fit"] not in ["Fit", "Not Fit"]:
            job_fit["job_fit"] = "Fit"

        if "fit_score" not in job_fit:
            job_fit["fit_score"] = 7.0
        else:
            try:
                job_fit["fit_score"] = float(job_fit["fit_score"])
            except:
                job_fit["fit_score"] = 7.0

        if "reason" not in job_fit:
            job_fit["reason"] = "AI analysis completed"

        state["job_fit"] = job_fit
        state["current_step"] = "job_fit_analyzed"

        print(f"✅ Job fit analysis completed: {job_fit['job_fit']} (Score: {job_fit['fit_score']})")

    except Exception as e:
        print(f"❌ Job fit analysis failed: {str(e)}")
        state["errors"].append(f"Job fit analysis failed: {str(e)}")

        # Fallback job fit analysis
        features = state["resume_features"]
        candidate_skills = [skill.lower() for skill in features.get("skills", [])]
        job_desc_lower = state["job_description"].lower()
        job_title_lower = state["job_title"].lower()

        # Simple keyword matching
        skill_matches = 0
        for skill in candidate_skills:
            if skill in job_desc_lower or skill in job_title_lower:
                skill_matches += 1

        total_skills = len(candidate_skills)

        if total_skills > 0:
            match_ratio = skill_matches / total_skills
            fit_score = min(10, max(1, match_ratio * 8 + 2))  # Score between 2-10
            fit_result = "Fit" if fit_score >= 5 else "Not Fit"
        else:
            fit_score = 5.0
            fit_result = "Fit"

        fallback_job_fit = {
            "job_fit": fit_result,
            "fit_score": round(fit_score, 1),
            "reason": f"Skill matching: {skill_matches}/{total_skills} skills match job requirements"
        }

        state["job_fit"] = fallback_job_fit
        state["current_step"] = "job_fit_analyzed"

        print(f"⚠️ Using fallback job fit analysis: {fit_result} (Score: {fit_score})")

    return state
