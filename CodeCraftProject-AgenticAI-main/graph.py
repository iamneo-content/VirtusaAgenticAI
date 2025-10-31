from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing import Dict
import os
from state import CodeCrafterState, slugify, write_all_outputs

# Import the agent functions
from agents.planning_agent import planning_agent
from agents.codegen_agent import codegen_agent
from agents.swagger_agent import swagger_agent
from agents.test_agent import generate_tests


def create_codecrafter_graph():
    """
    Create the CodeCrafter LangGraph workflow.
    """
    # Create a state graph with the defined state
    workflow = StateGraph(CodeCrafterState)
    
    # Add nodes for each agent
    workflow.add_node("planning_agent", planning_agent)
    workflow.add_node("codegen_agent", codegen_agent)
    workflow.add_node("swagger_agent", swagger_agent)
    workflow.add_node("test_agent", generate_tests)
    
    # Define the flow of execution
    # Start with planning
    workflow.set_entry_point("planning_agent")
    
    # After planning, proceed to code generation
    workflow.add_edge("planning_agent", "codegen_agent")
    
    # After code generation, proceed to swagger documentation
    workflow.add_edge("codegen_agent", "swagger_agent")
    
    # After swagger, proceed to test generation
    workflow.add_edge("swagger_agent", "test_agent")
    
    # Set the final node
    workflow.set_finish_point("test_agent")
    
    # Compile the graph
    app = workflow.compile()
    
    return app


def run_all_agents(user_story: str, language: str) -> Dict:
    """
    Function to run the LangGraph workflow and return results.
    """
    # Create the initial state
    initial_state = CodeCrafterState(
        user_story=user_story,
        language=language,
        features=[],
        services=[],
        architecture_hints={},
        architecture_config={},
        service_outputs={},
        swagger_outputs={},
        test_outputs={},
        output_base="output",
        output_dir="",
        planning_error="",
        codegen_error="",
        swagger_error="",
        test_error="",
        planning_complete=False,
        codegen_complete=False,
        swagger_complete=False,
        tests_complete=False,
        error_occurred=False
    )
    
    # Create the graph
    app = create_codecrafter_graph()
    
    # Run the graph with the initial state
    result = app.invoke(initial_state)
    
    # Set output directory based on the user story
    folder_name = slugify(user_story)
    output_dir = os.path.join(result["output_base"], folder_name)
    result["output_dir"] = output_dir
    
    # Write all outputs to the file system
    write_all_outputs(
        output_dir,
        result.get("service_outputs", {}),
        result.get("swagger_outputs", {}),
        result.get("test_outputs", {})
    )
    
    # For backward compatibility with UI - create aliases for the old key names
    result["code"] = result.get("service_outputs", {})
    result["swagger"] = result.get("swagger_outputs", {})
    result["tests"] = result.get("test_outputs", {})
    result["arch"] = result.get("architecture_config", {})
    
    return result