import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from state import CodeCrafterState, get_file_extension

load_dotenv()

def codegen_agent(state: CodeCrafterState) -> CodeCrafterState:
    """
    Code generation agent node that creates microservice code.
    """
    print("Running code generation agent...")
    
    # Initialize the Gemini model
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GEMINI_API_KEY_2"),
        temperature=0.1
    )
    
    service_files = {}
    
    features = state.get("features", [])
    services = state.get("services", [])
    
    for feature, service in zip(features, services):
        prompt = f"""You are a senior {state['language']} backend developer.

Generate production-ready microservice code for the following:
- Feature: {feature}
- Service Name: {service}
- Architecture: {state['architecture_config'].get('architecture', 'REST')}
- Database: {state['architecture_config'].get('database', 'PostgreSQL')}
- Messaging: {state['architecture_config'].get('messaging', 'None')}
- Cache: {state['architecture_config'].get('cache', 'None')}

Generate:
1. Controller or Route handler
2. Service class/method (business logic stub)
3. Model/Schema class

Only return JSON in this format:
{{
  "controller_filename": "...",
  "controller_code": "...",
  "service_filename": "...",
  "service_code": "...",
  "model_filename": "...",
  "model_code": "..."
}}
"""
        try:
            # Create a HumanMessage with the prompt
            message = HumanMessage(content=prompt)

            # Generate the response
            response = model.invoke([message])

            # Extract the text from the response
            raw_text = response.content.strip()

            # Clean the response - remove markdown code blocks
            raw_text = raw_text.strip("```json").strip("```").strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1].strip()

            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                raw_text = json_match.group(0)

            # Parse JSON with better error handling
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, create a minimal valid structure
                parsed = {
                    "controller_filename": f"{service}_controller.py",
                    "controller_code": "# Controller code generation failed",
                    "service_filename": f"{service}_service.py",
                    "service_code": "# Service code generation failed",
                    "model_filename": f"{service}_model.py",
                    "model_code": "# Model code generation failed"
                }

            # Add language info for file extension determination
            parsed["language"] = state["language"]

            service_files[service] = parsed

        except Exception as e:
            error_msg = f"[CodeGen Error for {service}] {e}"
            print(error_msg)
            # Still add a minimal structure so workflow continues
            service_files[service] = {
                "controller_filename": f"{service}_controller.py",
                "controller_code": "# Error in code generation",
                "service_filename": f"{service}_service.py",
                "service_code": "# Error in code generation",
                "model_filename": f"{service}_model.py",
                "model_code": "# Error in code generation",
                "language": state["language"]
            }
            continue

    updated_state = {
        "service_outputs": service_files,
        "codegen_complete": True,
        "codegen_error": ""
    }
    
    print("Code generation agent completed successfully")
    return {**state, **updated_state}