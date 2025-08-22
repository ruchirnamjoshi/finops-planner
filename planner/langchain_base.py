"""
LangChain-based base agent for consistent LLM interactions.
"""
from __future__ import annotations
import os
import logging
from typing import Dict, Any, List, Optional, Type
from abc import ABC, abstractmethod

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed")
except Exception as e:
    print(f"Warning: Failed to load .env file: {e}")

logger = logging.getLogger(__name__)

class LangChainBaseAgent(ABC):
    """
    Base agent class using LangChain for consistent LLM interactions.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.3):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LangChain LLM client."""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                openai_api_key=api_key
            )
            logger.info(f"✅ LangChain LLM initialized successfully with {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LangChain LLM: {e}")
            self.llm = None
    
    def _create_structured_chain(self, 
                               prompt_template: str, 
                               output_schema: Type[BaseModel],
                               system_message: str = "") -> Any:
        """
        Create a LangChain chain with structured output parsing.
        
        Args:
            prompt_template: The prompt template string
            output_schema: Pydantic model for output structure
            system_message: Optional system message
        
        Returns:
            LangChain chain with structured output parsing
        """
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        
        # Create the prompt template
        if system_message:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("human", prompt_template)
            ])
        else:
            prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Create output parser
        parser = JsonOutputParser(pydantic_object=output_schema)
        
        # Create the chain
        chain = prompt | self.llm | parser
        
        return chain
    
    def _invoke_chain_safely(self, 
                            chain: Any, 
                            inputs: Dict[str, Any],
                            fallback: Any = None) -> Any:
        """
        Safely invoke a LangChain chain with error handling and fallback.
        
        Args:
            chain: The LangChain chain to invoke
            inputs: Input data for the chain
            fallback: Fallback value if chain fails
        
        Returns:
            Chain output or fallback value
        """
        try:
            result = chain.invoke(inputs)
            logger.info("✅ LangChain chain executed successfully")
            return result
        except Exception as e:
            logger.error(f"LangChain chain failed: {e}")
            if fallback is not None:
                logger.info("Using fallback value")
                return fallback
            raise
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Main processing method to be implemented by subclasses."""
        pass

# Example output schemas for different agent types
class CostInsightsSchema(BaseModel):
    """Schema for cost analysis insights."""
    cost_breakdown_analysis: Dict[str, Any] = Field(description="Cost breakdown analysis")
    optimization_opportunities: List[Dict[str, Any]] = Field(description="List of optimization opportunities")
    pricing_insights: List[Dict[str, Any]] = Field(description="Pricing insights and observations")
    cost_forecast: Dict[str, Any] = Field(description="Cost forecasting and trends")

class OptimizationRecommendationsSchema(BaseModel):
    """Schema for optimization recommendations."""
    recommendations: List[Dict[str, Any]] = Field(description="List of optimization recommendations")

class RiskAnalysisSchema(BaseModel):
    """Schema for risk analysis."""
    risks: List[Dict[str, Any]] = Field(description="List of identified risks")

class VisualizationInsightsSchema(BaseModel):
    """Schema for visualization insights."""
    insights: List[str] = Field(description="Key insights from the data")
    recommendations: List[str] = Field(description="Recommendations based on analysis")
