
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
            logger.info("TOOL_EXECUTE agent=%s tool=%s args=%s", self._agent_type, tool_name, self._safe_json(self._sanitize_for_log(tool_args)))
            result = self._tools.execute_tool(tool_name, tool_args)
            logger.info("TOOL_RESULT agent=%s tool=%s payload=%s", self._agent_type, tool_name, self._safe_json(self._sanitize_for_log(result)))
            return result
        except Exception as e:
            logger.exception(f"Tool execution failed: {tool_name}")
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
                "data": None
            }

    @staticmethod
    def _sanitize_for_log(value):
        if isinstance(value, dict):
            sanitized = {}
            for k, v in value.items():
                key_l = str(k).lower()
                if any(secret_key in key_l for secret_key in ("token", "key", "api_key", "authorization")):
                    sanitized[k] = "***"
                else:
                    sanitized[k] = ToolExecuter._sanitize_for_log(v)
            return sanitized
        if isinstance(value, list):
            return [ToolExecuter._sanitize_for_log(v) for v in value]
        return value

    @staticmethod
    def _safe_json(value) -> str:
        try:
            import json
            return json.dumps(value, ensure_ascii=True)
        except Exception:
            return str(value)
    def __repr__(self):
        return f"ToolExecutor({self._agent_type})"

