"""
PDF Extraction Node
Handles PDF document extraction and parsing
"""

from typing import Any
from state.models import HLDState
from agent import PDFExtractionAgent
from .base_node import BaseNode


class PDFExtractionNode(BaseNode):
    """Node for PDF extraction and parsing"""

    def __init__(self):
        """Initialize PDF extraction node"""
        super().__init__(
            name="pdf_extraction",
            description="Extract and parse content from PDF documents",
            critical=True
        )
        self.agent = PDFExtractionAgent()

    def execute_logic(self, hld_state: HLDState) -> Any:
        """
        Execute PDF extraction logic

        Args:
            hld_state: Current workflow state

        Returns:
            Extracted PDF content and metadata
        """
        return self.agent.process(hld_state)
