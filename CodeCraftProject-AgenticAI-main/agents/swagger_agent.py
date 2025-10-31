import os
import json
import re
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from state import CodeCrafterState, get_file_extension

load_dotenv()

def swagger_agent(state: CodeCrafterState) -> CodeCrafterState:
    """
    Swagger documentation agent node that generates API documentation.
    """
    print("Running swagger agent...")
    
    # Initialize the Gemini model
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GEMINI_API_KEY_3"),
        temperature=0.1
    )
    
    swagger_docs = {}
    
    service_outputs = state.get("service_outputs", {})
    
    for service_name, files in service_outputs.items():
        controller_code = files.get("controller_code", "")
        controller_code_escaped = controller_code.replace('"', '\\"').replace('\n', '\\n')
        
        prompt = f"""You are an expert in API documentation.

Given this {state['language']} controller code, generate a Swagger (OpenAPI) specification for its endpoints.

Controller Code:
"{controller_code_escaped}"

Respond ONLY in this JSON format:
{{
  "filename": "swagger.yaml",
  "content": "---\\nopenapi: 3.0.0\\ninfo:\\n  title: {service_name} API\\n..."
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
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                raw_text = json_match.group(0)

            # Clean up control characters
            raw_text = raw_text.replace("\\r", "").replace("\\t", "    ").replace("\\x00", "")
            raw_text = re.sub(r"[\x01-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", raw_text)

            # Parse JSON with better error handling
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, create a minimal valid structure
                parsed = {
                    "filename": f"{service_name}_swagger.yaml",
                    "content": f"openapi: 3.0.0\ninfo:\n  title: {service_name} API\npaths: {{}}"
                }

            swagger_docs[service_name] = parsed

        except Exception as e:
            error_msg = f"[Swagger Error for {service_name}] {e}"
            print(error_msg)
            # Still add a minimal structure so workflow continues
            swagger_docs[service_name] = {
                "filename": f"{service_name}_swagger.yaml",
                "content": f"openapi: 3.0.0\ninfo:\n  title: {service_name} API\npaths: {{}}"
            }
            continue

    updated_state = {
        "swagger_outputs": swagger_docs,
        "swagger_complete": True,
        "swagger_error": ""
    }
    
    print("Swagger agent completed successfully")
    return {**state, **updated_state}