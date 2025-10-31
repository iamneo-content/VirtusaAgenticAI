"""
Resume Parsing Node
Extracts structured features from resume text using Gemini AI.
"""

import os
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv
from state import CandidateState
import time

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY_1"))

# Configure model with timeout settings
generation_config = {
    "temperature": 0.1,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 2048,
}

model = genai.GenerativeModel("gemini-2.0-flash", generation_config=generation_config)


def clean_json_block(text):
    """Clean Gemini response to valid JSON"""
    text = text.strip()
    text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE)
    text = text.replace("'", '"')
    text = re.sub(r"//.*", "", text)
    text = re.sub(r",(\s*[}\]])", r"\1", text)

    if not text.strip().endswith("}"):
        text += "}"
    return text


def parse_resume(state: CandidateState) -> CandidateState:
    """Parse resume and extract structured features"""

    print("🔄 Starting resume parsing...")

    prompt = f"""
Extract structured data from this resume. Return valid JSON only:

{{
  "full_name": str,
  "email": str,
  "phone": str,
  "education": {{"degree": str, "major": str, "university": str}},
  "total_experience_years": float,
  "skills": [str],
  "projects": [str],
  "certifications": [str],
  "leadership_experience": int,
  "has_research_work": int
}}

Resume: {state["resume_text"][:3000]}
"""

    try:
        print("📤 Sending request to Gemini...")
        start_time = time.time()

        # Set a timeout for the API call
        response = model.generate_content(prompt, request_options={"timeout": 30})

        elapsed_time = time.time() - start_time
        print(f"📥 Received response in {elapsed_time:.2f} seconds")

        cleaned = clean_json_block(response.text)
        features = json.loads(cleaned)

        state["resume_features"] = features
        state["candidate_name"] = features.get("full_name", "Unknown")
        state["current_step"] = "parsed"

        print(f"✅ Resume parsed successfully for {state['candidate_name']}")

    except Exception as e:
        print(f"❌ Resume parsing failed: {str(e)}")
        state["errors"].append(f"Resume parsing failed: {str(e)}")

        # Fallback: create basic features from resume text
        resume_text = state["resume_text"].lower()

        # Extract basic info using simple text processing
        lines = state["resume_text"].split('\n')
        name = lines[0] if lines else "Unknown Candidate"

        # Basic skill extraction
        common_skills = ["python", "java", "javascript", "sql", "react", "node", "aws", "docker", "git"]
        found_skills = [skill for skill in common_skills if skill in resume_text]

        # Estimate experience from text
        exp_keywords = ["year", "experience", "worked"]
        exp_years = 2.0  # default

        fallback_features = {
            "full_name": name,
            "email": "not_extracted@example.com",
            "phone": "not_extracted",
            "education": {"degree": "Not extracted", "major": "Not extracted", "university": "Not extracted"},
            "total_experience_years": exp_years,
            "skills": found_skills,
            "projects": ["Project details not extracted"],
            "certifications": [],
            "leadership_experience": 0,
            "has_research_work": 0
        }

        state["resume_features"] = fallback_features
        state["candidate_name"] = name
        state["current_step"] = "parsed"

        print("⚠️ Using fallback feature extraction")

    return state
