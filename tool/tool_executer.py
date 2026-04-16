
from agents.agent_names import Agents
from tool.base_tool import BaseTool
import logging
from typing import Dict


logger = logging.getLogger(__name__)
class ToolExecuter:

    _registry: Dict[str, type[BaseTool]] = {}

    @classmethod
    def register(cls,agent_type:str,tool_class:type[BaseTool]):
        cls._registry[agent_type] = tool_class
        logger.info(f"Registered {tool_class.__name__} for {agent_type}")

    def __init__(self,agent_type:str):
        if agent_type not in self._registry:
            raise ValueError(f"No tools registered for {agent_type}. "
                           f"Did you forget to call ToolExecutor.register()?")
        self._tools = self._registry[agent_type]()
        self._agent_type = agent_type
        

    def get_tool_definitions(self) -> list[dict]:
        return self._tools.get_tool_definitions()
    
    
    def execute_tool(self,tool_name:str,tool_args:dict):
        try:
            return self._tools.execute_tool(tool_name, tool_args)
        except Exception as e:
            logger.exception(f"Tool execution failed: {tool_name}")
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
                "data": None
            }
    def __repr__(self):
        return f"ToolExecutor({self._agent_type})"

