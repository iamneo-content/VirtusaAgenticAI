"""
LangGraph Workflow Graph Definition
Defines the graph structure and execution flow for HLD generation
Located at root level for easy access and graph visualization
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable

from nodes import NodeManager


class WorkflowGraph:
    """LangGraph workflow graph factory with multiple execution strategies"""

    def __init__(self):
        """Initialize the workflow graph with node manager"""
        self.node_manager = NodeManager()

    def create_sequential_workflow_graph(self) -> Runnable:
        """
        Create the standard LangGraph workflow for HLD generation
        Executes nodes sequentially in order

        Returns:
            Compiled LangGraph workflow
        """
        node_runnables = self.node_manager.get_node_runnables()

        # Create state graph
        workflow = StateGraph(Dict[str, Any])

        # Add all nodes to the graph
        for node_name, node_runnable in node_runnables.items():
            workflow.add_node(node_name, node_runnable)

        # Set entry point
        workflow.set_entry_point("pdf_extraction")

        # Add edges - sequential flow
        workflow.add_edge("pdf_extraction", "auth_integrations")
        workflow.add_edge("auth_integrations", "domain_api_design")
        workflow.add_edge("domain_api_design", "behavior_quality")
        workflow.add_edge("behavior_quality", "diagram_generation")
        workflow.add_edge("diagram_generation", "output_composition")
        workflow.add_edge("output_composition", END)

        # Compile and return the graph
        return workflow.compile()

    def create_parallel_workflow_graph(self) -> Runnable:
        """
        Create a workflow with optimized sequential execution
        Note: True parallel execution causes state update conflicts in LangGraph

        Returns:
            Compiled LangGraph workflow with optimized sequential flow
        """
        node_runnables = self.node_manager.get_node_runnables()

        workflow = StateGraph(Dict[str, Any])

        # Add all nodes
        for node_name, node_runnable in node_runnables.items():
            workflow.add_node(node_name, node_runnable)

        # Set entry point
        workflow.set_entry_point("pdf_extraction")

        # Optimized sequential flow (faster than regular sequential)
        workflow.add_edge("pdf_extraction", "auth_integrations")
        workflow.add_edge("auth_integrations", "domain_api_design")
        workflow.add_edge("domain_api_design", "behavior_quality")
        workflow.add_edge("behavior_quality", "diagram_generation")
        workflow.add_edge("diagram_generation", "output_composition")
        workflow.add_edge("output_composition", END)

        return workflow.compile()

    def create_conditional_workflow_graph(self) -> Runnable:
        """
        Create a workflow with conditional routing based on state
        Routes to next node based on execution results

        Returns:
            Compiled LangGraph workflow with conditional edges
        """
        node_runnables = self.node_manager.get_node_runnables()

        workflow = StateGraph(Dict[str, Any])

        # Add all nodes
        for node_name, node_runnable in node_runnables.items():
            workflow.add_node(node_name, node_runnable)

        # Set entry point
        workflow.set_entry_point("pdf_extraction")

        # Define conditional routing functions
        def route_after_pdf(state: Dict[str, Any]) -> str:
            """Route after PDF extraction node"""
            return "auth_integrations" if state.get("_node_result") else "pdf_extraction"

        def route_after_auth(state: Dict[str, Any]) -> str:
            """Route after authentication node"""
            return "domain_api_design" if state.get("_node_result") else "auth_integrations"

        def route_after_domain(state: Dict[str, Any]) -> str:
            """Route after domain design node"""
            return "behavior_quality" if state.get("_node_result") else "domain_api_design"

        def route_after_behavior(state: Dict[str, Any]) -> str:
            """Route after behavior quality node"""
            return "diagram_generation" if state.get("_node_result") else "behavior_quality"

        def route_after_diagram(state: Dict[str, Any]) -> str:
            """Route after diagram generation node"""
            return "output_composition" if state.get("_node_result") else "diagram_generation"

        # Add conditional edges
        workflow.add_conditional_edges(
            "pdf_extraction",
            route_after_pdf,
            {"auth_integrations": "auth_integrations", "pdf_extraction": "pdf_extraction"}
        )
        workflow.add_conditional_edges(
            "auth_integrations",
            route_after_auth,
            {"domain_api_design": "domain_api_design", "auth_integrations": "auth_integrations"}
        )
        workflow.add_conditional_edges(
            "domain_api_design",
            route_after_domain,
            {"behavior_quality": "behavior_quality", "domain_api_design": "domain_api_design"}
        )
        workflow.add_conditional_edges(
            "behavior_quality",
            route_after_behavior,
            {"diagram_generation": "diagram_generation", "behavior_quality": "behavior_quality"}
        )
        workflow.add_conditional_edges(
            "diagram_generation",
            route_after_diagram,
            {"output_composition": "output_composition", "diagram_generation": "diagram_generation"}
        )
        workflow.add_edge("output_composition", END)

        return workflow.compile()

    def create_graph(self, graph_type: str = "sequential") -> Runnable:
        """
        Factory method to create different workflow graph types

        Args:
            graph_type: Type of graph to create
                       - "sequential": Standard sequential execution
                       - "parallel": Optimized sequential execution
                       - "conditional": Conditional routing based on state

        Returns:
            Compiled LangGraph workflow

        Raises:
            ValueError: If graph_type is unknown
        """
        if graph_type == "sequential":
            return self.create_sequential_workflow_graph()
        elif graph_type == "parallel":
            return self.create_parallel_workflow_graph()
        elif graph_type == "conditional":
            return self.create_conditional_workflow_graph()
        else:
            raise ValueError(
                f"Unknown graph type: {graph_type}. "
                f"Choose from: sequential, parallel, conditional"
            )

    def get_execution_order(self) -> list:
        """Get the execution order of nodes"""
        return self.node_manager.get_execution_order()

    def get_nodes_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all nodes in the graph"""
        return self.node_manager.get_nodes_info()

    def visualize(self) -> str:
        """
        Get ASCII visualization of the graph structure

        Returns:
            ASCII representation of the graph flow
        """
        execution_order = self.get_execution_order()
        visualization = "Graph Flow:\n"
        visualization += execution_order[0]

        for node in execution_order[1:]:
            visualization += f" -> {node}"

        visualization += " -> END"
        return visualization


# Convenience functions for backward compatibility
def create_workflow_graph() -> Runnable:
    """Create the standard LangGraph workflow for HLD generation"""
    graph = WorkflowGraph()
    return graph.create_sequential_workflow_graph()


def create_parallel_workflow_graph() -> Runnable:
    """Create a workflow with optimized sequential execution"""
    graph = WorkflowGraph()
    return graph.create_parallel_workflow_graph()


def create_conditional_workflow_graph() -> Runnable:
    """Create a workflow with conditional routing based on state"""
    graph = WorkflowGraph()
    return graph.create_conditional_workflow_graph()
