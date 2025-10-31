import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from state import CodeCrafterState, get_file_extension

load_dotenv()

def planning_agent(state: CodeCrafterState) -> CodeCrafterState:
    """
    Planning agent node that analyzes user story and generates architecture plan.
    """
    print("Running planning agent...")
    
    # Initialize the Gemini model
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GEMINI_API_KEY_1"),
        temperature=0.1
    )
    
    user_story_escaped = state["user_story"].replace('"', '\\"').replace('\n', '\\n')
    prompt = f"""You are a software architect. Read the following user story and perform two tasks:

1. Extract the following:
    - A list of core features (verbs like "register", "login", "subscribe")
    - Corresponding service names based on standard microservice naming conventions
    - Optional architectural hints mentioned by the user (e.g., architecture style, DB, messaging, cache)

2. Based on the features, services, and backend language, recommend the most suitable microservice architecture and infrastructure stack. Prioritize any user-specified preferences when provided.

User Story:
"{user_story_escaped}"

Backend Language: {state["language"]}

Respond ONLY in this JSON format:
{{
  "features": [...],
  "services": [...],
  "architecture_hints": {{
    "architecture": "REST | gRPC | Event-driven | None",
    "database": "PostgreSQL | MongoDB | DynamoDB | None",
    "messaging": "Kafka | SQS | None",
    "cache": "Redis | None",
    "api_gateway": "Spring Cloud Gateway | Express Gateway | Ocelot | None",
    "service_discovery": "Eureka | Consul | None"
  }},
  "architecture_config": {{
    "architecture": "...",
    "database": "...",
    "messaging": "...",
    "cache": "...",
    "api_gateway": "...",
    "service_discovery": "..."
  }}
}}"""
    
    try:
        # Create a HumanMessage with the prompt
        message = HumanMessage(content=prompt)
        
        # Generate the response
        response = model.invoke([message])
        
        # Extract the text from the response
        raw_text = response.content.strip()
        
        # Clean the response
        raw_text = raw_text.strip("```json").strip("```").strip()
        
        # Parse JSON
        json_output = json.loads(raw_text)
        
        # Update state with the results
        updated_state = {
            "features": json_output.get("features", []),
            "services": json_output.get("services", []),
            "architecture_hints": json_output.get("architecture_hints", {}),
            "architecture_config": json_output.get("architecture_config", {}),
            "planning_complete": True,
            "planning_error": ""
        }
        
        print("Planning agent completed successfully")
        return {**state, **updated_state}
        
    except Exception as e:
        error_msg = f"[PlanningAgent Error] {e}"
        print(error_msg)
        return {**state, "planning_error": error_msg, "planning_complete": False, "error_occurred": True}