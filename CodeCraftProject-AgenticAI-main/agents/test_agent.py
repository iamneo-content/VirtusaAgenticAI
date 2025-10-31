import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from state import CodeCrafterState, get_file_extension

load_dotenv()

def generate_tests(state: CodeCrafterState) -> CodeCrafterState:
    """
    Test generation agent node that creates unit tests.
    """
    print("Running test agent...")
    
    # Initialize the Gemini model
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GEMINI_API_KEY_4"),
        temperature=0.1
    )
    
    test_files = {}
    
    service_outputs = state.get("service_outputs", {})
    
    for service_name, files in service_outputs.items():
        controller_code = files.get("controller_code", "")
        service_code = files.get("service_code", "")
        
        controller_code_escaped = controller_code.replace('"', '\\"').replace('\n', '\\n')
        service_code_escaped = service_code.replace('"', '\\"').replace('\n', '\\n')
        
        prompt = f"""You are a senior {state['language']} test automation engineer.

Write unit test code for the following controller and service.

Controller Code:
"{controller_code_escaped}"

Service Code:
"{service_code_escaped}"

Respond ONLY in this JSON format:
{{
  "filename": "UserServiceTest.{get_file_extension(state['language'])}",
  "content": "..."
}}
"""
        
        try:
            # Create a HumanMessage with the prompt
            message = HumanMessage(content=prompt)
            
            # Generate the response
            response = model.invoke([message])
            
            # Extract the text from the response
            raw = response.content.strip()
            
            # Clean the response
            raw = raw.strip("```json").strip("```").strip()
            
            # Parse JSON
            parsed = json.loads(raw)
            
            test_files[service_name] = parsed
            
        except Exception as e:
            error_msg = f"[TestAgent Error for {service_name}] {e}"
            print(error_msg)
            continue

    updated_state = {
        "test_outputs": test_files,
        "tests_complete": True,
        "test_error": ""
    }
    
    print("Test agent completed successfully")
    return {**state, **updated_state}