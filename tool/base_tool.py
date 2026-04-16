from abc import ABC, abstractmethod
from typing import Any
import logging

 
logger = logging.getLogger(__name__)


class BaseTool(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_tool_definitions(self):
        """give the definition of all the tools available for that specific agent tools"""
        
    @abstractmethod
    def _execute_tool(self,tool_name:str, tool_args:dict) ->dict:
        """implementation of mapping the actual backend function with the tool based on tool name"""

    def execute_tool(self, tool_name: str, tool_args: dict) -> dict:
        """Public execution entrypoint used by ToolExecuter."""
        return self._execute_tool(tool_name=tool_name, tool_args=tool_args)