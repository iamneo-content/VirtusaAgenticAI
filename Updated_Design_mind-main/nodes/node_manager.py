"""
Node Manager
Manages all workflow nodes and provides utilities for node operations
"""

from typing import Dict, Any, List
from langchain_core.runnables import RunnableLambda

from .pdf_extraction_node import PDFExtractionNode
from .auth_integrations_node import AuthIntegrationsNode
from .domain_api_node import DomainAPINode
from .behavior_quality_node import BehaviorQualityNode
from .diagram_generation_node import DiagramGenerationNode
from .output_composition_node import OutputCompositionNode
from state.models import HLDState


class NodeManager:
    """Manages all workflow nodes"""

    # Node definitions with metadata
    NODE_DEFINITIONS = {
        "pdf_extraction": {
            "description": "Extract and parse content from PDF documents",
            "dependencies": [],
            "critical": True
        },
        "auth_integrations": {
            "description": "Analyze authentication mechanisms and integrations",
            "dependencies": ["pdf_extraction"],
            "critical": False
        },
        "domain_api_design": {
            "description": "Design domain models and API interfaces",
            "dependencies": ["auth_integrations"],
            "critical": False
        },
        "behavior_quality": {
            "description": "Analyze behavior patterns and quality metrics",
            "dependencies": ["domain_api_design"],
            "critical": False
        },
        "diagram_generation": {
            "description": "Generate visual diagrams and representations",
            "dependencies": ["behavior_quality"],
            "critical": False
        },
        "output_composition": {
            "description": "Compose and format final output",
            "dependencies": ["diagram_generation"],
            "critical": False
        }
    }

    # Workflow execution order
    EXECUTION_ORDER = [
        "pdf_extraction",
        "auth_integrations",
        "domain_api_design",
        "behavior_quality",
        "diagram_generation",
        "output_composition"
    ]

    def __init__(self):
        """Initialize all nodes"""
        self.nodes = {
            "pdf_extraction": PDFExtractionNode(),
            "auth_integrations": AuthIntegrationsNode(),
            "domain_api_design": DomainAPINode(),
            "behavior_quality": BehaviorQualityNode(),
            "diagram_generation": DiagramGenerationNode(),
            "output_composition": OutputCompositionNode()
        }

    def get_node(self, node_name: str):
        """Get a node by name"""
        if node_name not in self.nodes:
            raise ValueError(f"Unknown node: {node_name}")
        return self.nodes[node_name]

    def get_all_nodes(self) -> Dict[str, Any]:
        """Get all nodes"""
        return self.nodes

    def get_node_runnables(self) -> Dict[str, RunnableLambda]:
        """Get all nodes as RunnableLambda objects for LangGraph"""
        return {
            node_name: node.get_runnable()
            for node_name, node in self.nodes.items()
        }

    def get_nodes_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all nodes"""
        return {
            node_name: {
                "description": self.NODE_DEFINITIONS[node_name]["description"],
                "dependencies": self.NODE_DEFINITIONS[node_name]["dependencies"],
                "critical": self.NODE_DEFINITIONS[node_name]["critical"],
                "status": node.status
            }
            for node_name in self.EXECUTION_ORDER
        }

    def get_execution_order(self) -> List[str]:
        """Get the execution order of nodes"""
        return self.EXECUTION_ORDER.copy()

    def get_node_dependencies(self, node_name: str) -> List[str]:
        """Get dependencies for a specific node"""
        if node_name not in self.NODE_DEFINITIONS:
            raise ValueError(f"Unknown node: {node_name}")
        return self.NODE_DEFINITIONS[node_name]["dependencies"]

    def is_node_critical(self, node_name: str) -> bool:
        """Check if a node is critical"""
        if node_name not in self.NODE_DEFINITIONS:
            raise ValueError(f"Unknown node: {node_name}")
        return self.NODE_DEFINITIONS[node_name]["critical"]

    def should_continue(self, state: Dict[str, Any]) -> str:
        """Conditional routing based on state"""
        hld_state = HLDState(**state)

        # Check for critical errors that should stop the workflow
        if hld_state.has_errors():
            critical_stages = ["pdf_extraction"]
            for stage in critical_stages:
                if (stage in hld_state.status and
                    hld_state.status[stage].status == "failed"):
                    return "END"

        # Continue to next stage based on completion
        for node_name in self.EXECUTION_ORDER:
            if not hld_state.is_stage_completed(node_name):
                return node_name

        return "END"

    def reset_all_nodes(self):
        """Reset the status of all nodes"""
        for node in self.nodes.values():
            node.status = "pending"
            node.last_error = None
