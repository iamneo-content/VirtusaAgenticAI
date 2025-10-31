"""Base agent class for all LangGraph agents."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
from loguru import logger

from langchain_core.language_models import BaseLanguageModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import get_settings
from .state_models import WorkflowState, AgentExecution


class BaseAgent(ABC):
    """Base class for all LangGraph agents."""
    
    def __init__(
        self,
        name: str,
        description: str,
        llm: Optional[BaseLanguageModel] = None,
        temperature: float = 0.1,
        max_tokens: int = 4000
    ):
        self.name = name
        self.description = description
        self.settings = get_settings()
        self.logger = logger.bind(agent=name)
        
        # Initialize LLM
        if llm is None:
            api_key = self.settings.gemini_api_key
            if not api_key:
                self.logger.error(f"Settings object: {self.settings}")
                self.logger.error(f"Gemini API Key: {api_key}")
                raise ValueError(
                    "GEMINI_API_KEY_1 not found in environment. "
                    "Please add it to your .env file."
                )
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            self.llm = llm
        
        # Agent metadata
        self.created_at = datetime.now()
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.success_count = 0
        self.error_count = 0
        
        self.logger.info(f"Initialized agent: {self.name}")
    
    @abstractmethod
    async def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute the agent's main functionality."""
        pass
    
    @abstractmethod
    def get_prompt_template(self) -> ChatPromptTemplate:
        """Get the agent's prompt template."""
        pass
    
    async def run(self, state: WorkflowState) -> WorkflowState:
        """Run the agent with error handling and monitoring."""
        execution = state.add_agent_execution(self.name)
        
        try:
            self.logger.info(f"Starting execution for agent: {self.name}")
            
            # Pre-execution validation
            if not self.validate_input(state):
                raise ValueError(f"Input validation failed for agent: {self.name}")
            
            # Execute the agent
            result_state = await self.execute(state)
            
            # Post-execution validation
            if not self.validate_output(result_state):
                raise ValueError(f"Output validation failed for agent: {self.name}")
            
            # Update execution tracking
            state.complete_agent_execution(self.name, success=True)
            self.success_count += 1
            
            self.logger.info(f"Successfully completed execution for agent: {self.name}")
            return result_state
            
        except Exception as e:
            error_msg = f"Error in agent {self.name}: {str(e)}"
            self.logger.error(error_msg)
            
            # Update execution tracking
            state.complete_agent_execution(self.name, success=False, error=error_msg)
            self.error_count += 1
            
            # Add error to state
            if hasattr(state, 'errors'):
                state.errors.append(error_msg)
            
            # Re-raise or handle gracefully based on agent configuration
            if self.is_critical():
                raise
            else:
                return state
        
        finally:
            self.execution_count += 1
            if execution.execution_time:
                self.total_execution_time += execution.execution_time
    
    def validate_input(self, state: WorkflowState) -> bool:
        """Validate input state before execution."""
        # Basic validation - can be overridden by specific agents
        return state is not None
    
    def validate_output(self, state: WorkflowState) -> bool:
        """Validate output state after execution."""
        # Basic validation - can be overridden by specific agents
        return state is not None
    
    def is_critical(self) -> bool:
        """Determine if this agent is critical for the workflow."""
        # Override in specific agents if they are critical
        return False
    
    async def generate_response(
        self,
        prompt_template: ChatPromptTemplate,
        input_variables: Dict[str, Any]
    ) -> str:
        """Generate response using the LLM."""
        try:
            chain = prompt_template | self.llm | StrOutputParser()
            response = await chain.ainvoke(input_variables)
            return response.strip()
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        return f"""You are {self.name}, a specialized AI agent in a financial advisory system.
        
        Your role: {self.description}
        
        Guidelines:
        - Provide accurate, professional financial advice
        - Consider regulatory compliance and risk factors
        - Be clear and concise in your responses
        - Use data-driven insights when available
        - Maintain client confidentiality and professionalism
        
        Always respond in a structured format that can be easily processed by other agents in the system."""
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this agent."""
        avg_execution_time = (
            self.total_execution_time / self.execution_count 
            if self.execution_count > 0 else 0
        )
        
        success_rate = (
            self.success_count / self.execution_count 
            if self.execution_count > 0 else 0
        )
        
        return {
            "agent_name": self.name,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": avg_execution_time,
            "created_at": self.created_at.isoformat()
        }
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.success_count = 0
        self.error_count = 0
        self.logger.info(f"Reset metrics for agent: {self.name}")
    
    def __str__(self) -> str:
        return f"Agent({self.name})"
    
    def __repr__(self) -> str:
        return f"Agent(name='{self.name}', description='{self.description}')"


class CriticalAgent(BaseAgent):
    """Base class for critical agents that must succeed."""
    
    def is_critical(self) -> bool:
        return True


class OptionalAgent(BaseAgent):
    """Base class for optional agents that can fail gracefully."""
    
    def is_critical(self) -> bool:
        return False