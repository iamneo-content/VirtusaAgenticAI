import os
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv
from state import CandidateState
import time

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY_3"))

# Configure model with timeout settings
generation_config = {
    "temperature": 0.3,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 2048,
}

model = genai.GenerativeModel("gemini-2.0-flash", generation_config=generation_config)

def extract_json_array_from_text(text):
    """Extract JSON array from text using simple string operations"""
    text = text.strip()
    
    # Remove markdown code blocks
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)
    
    # Find the JSON array
    start = text.find('[')
    if start == -1:
        return None
    
    # Find matching closing bracket
    bracket_count = 0
    end = start
    for i, char in enumerate(text[start:], start):
        if char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
            if bracket_count == 0:
                end = i
                break
    
    if bracket_count != 0:
        return None
    
    json_str = text[start:end+1]
    return json_str

def generate_questions(state: CandidateState) -> CandidateState:
    """Generate personalized interview questions"""
    
    print("🔄 Starting question generation...")
    
    # Simple prompt with clear JSON structure
    prompt = f"""
Create 3 interview questions for {state["job_title"]} position. Return only this JSON array:

[
  {{"id": "Q1", "type": "concept", "question": "Tell me about your experience with {state['job_title']}", "reference_answer": "Should describe relevant experience"}},
  {{"id": "Q2", "type": "concept", "question": "What are your strongest technical skills", "reference_answer": "Should mention key skills"}},
  {{"id": "Q3", "type": "code", "question": "Write a simple function or algorithm", "reference_answer": "Basic implementation"}}
]

Experience level: {state["experience_level"]}
Return only the JSON array.
"""
    
    try:
        print("📤 Sending question generation request to Gemini...")
        start_time = time.time()
        
        response = model.generate_content(prompt, request_options={"timeout": 30})
        
        elapsed_time = time.time() - start_time
        print(f"📥 Received questions in {elapsed_time:.2f} seconds")
        
        # Debug: print raw response
        raw_text = response.text.strip()
        print(f"🔍 Raw response: {raw_text[:200]}...")
        
        # Extract JSON array
        json_str = extract_json_array_from_text(raw_text)
        if not json_str:
            raise ValueError("No valid JSON array found in response")
        
        print(f"🔍 Extracted JSON: {json_str[:150]}...")
        
        # Parse JSON
        questions = json.loads(json_str)
        
        if not isinstance(questions, list):
            raise ValueError("Response is not a list")
        
        # Validate and fix question structure
        for i, q in enumerate(questions):
            if not isinstance(q, dict):
                continue
                
            if "id" not in q:
                q["id"] = f"Q{i+1}"
            if "type" not in q:
                q["type"] = "concept"
            if "question" not in q:
                q["question"] = f"Tell me about your experience with {state['job_title']}"
            if "reference_answer" not in q:
                q["reference_answer"] = "Should provide relevant details"
        
        # Filter out invalid questions
        valid_questions = [q for q in questions if isinstance(q, dict) and "question" in q]
        
        if len(valid_questions) == 0:
            raise ValueError("No valid questions found")
        
        state["questions"] = valid_questions
        state["current_step"] = "questions_generated"
        print(f"✅ Generated {len(valid_questions)} questions successfully")
            
    except Exception as e:
        print(f"❌ Question generation failed: {str(e)}")
        state["errors"].append(f"Question generation failed: {str(e)}")
        
        # Fallback questions based on job title and experience level
        job_title = state["job_title"].lower()
        exp_level = state["experience_level"].lower()
        
        if "engineer" in job_title or "developer" in job_title:
            fallback_questions = [
                {
                    "id": "Q1",
                    "type": "concept",
                    "question": f"Tell me about your experience with {state['job_title']} and what interests you about this position?",
                    "reference_answer": "Should demonstrate understanding of the role and genuine interest"
                },
                {
                    "id": "Q2", 
                    "type": "concept",
                    "question": "What programming languages and technologies are you most comfortable with?",
                    "reference_answer": "Should mention relevant technologies for the role"
                },
                {
                    "id": "Q3",
                    "type": "code",
                    "question": "Write a simple function to solve a basic programming problem.",
                    "reference_answer": "Should show basic programming skills"
                }
            ]
        else:
            # Generic questions for non-technical roles
            fallback_questions = [
                {
                    "id": "Q1",
                    "type": "concept",
                    "question": f"Why are you interested in the {state['job_title']} position?",
                    "reference_answer": "Should show understanding of role and company fit"
                },
                {
                    "id": "Q2",
                    "type": "concept", 
                    "question": "What are your key strengths that make you suitable for this role?",
                    "reference_answer": "Should align strengths with job requirements"
                },
                {
                    "id": "Q3",
                    "type": "concept",
                    "question": "Describe a challenging situation you faced and how you handled it.",
                    "reference_answer": "Should demonstrate problem-solving and resilience"
                }
            ]
        
        state["questions"] = fallback_questions
        state["current_step"] = "questions_generated"
        
        print(f"⚠️ Using {len(fallback_questions)} fallback questions")
    
    return state