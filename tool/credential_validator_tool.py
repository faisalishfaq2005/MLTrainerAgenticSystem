from tool_methods.credential_validator_methods import CredentialValidator
from tool.base_tool import BaseTool


class CredentialValidatorTools(BaseTool):
    def __init__(self):
        super().__init__()
    
    # Add to intake_manager_agent.py

    def get_tool_definitions(self):
        """Return tool definitions in OpenAI/LiteLLM format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "validate_hf_token",
                    "description": "Validate a HuggingFace token by calling the HF API. Returns (is_valid: bool, error_message: str | None). Use this immediately when user provides an HF token to verify it's correct before storing it.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "token": {
                                "type": "string",
                                "description": "The HuggingFace token to validate. Must start with 'hf_'."
                            }
                        },
                        "required": ["token"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "validate_kaggle_credentials",
                    "description": "Validate Kaggle username and API key by calling Kaggle API. Returns (is_valid: bool, error_message: str | None). Use this immediately when user provides Kaggle credentials to verify they're correct.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "username": {
                                "type": "string",
                                "description": "Kaggle username"
                            },
                            "key": {
                                "type": "string",
                                "description": "Kaggle API key"
                            }
                        },
                        "required": ["username", "key"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "validate_llm_api_key",
                    "description": "Validate an LLM provider API key (anthropic, openai, google). Returns (is_valid: bool, error_message: str | None). Use this when user provides an LLM API key to verify it works.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "provider": {
                                "type": "string",
                                "enum": ["anthropic", "openai", "google", "ollama"],
                                "description": "The LLM provider name"
                            },
                            "api_key": {
                                "type": "string",
                                "description": "The API key to validate"
                            }
                        },
                        "required": ["provider", "api_key"]
                    }
                }
            }
        ]
    
    def _execute_tool(self,tool_name:str, tool_args:dict) ->dict:
        self._validator=CredentialValidator()
        if tool_name=="validate_hf_token":
            token=tool_args.get("token","")
            is_valid,error=self._validator.validate_hf_token(token)

            username=None
            if is_valid:
                username=self._validator.get_hf_username(token)

            return {
                "is_valid":is_valid,
                "error":error,
                "username":username
            }
        
        elif tool_name == "validate_kaggle_credentials":
            username = tool_args.get("username", "")
            key = tool_args.get("key", "")
            is_valid, error = self._validator.validate_kaggle_credentials(username, key)
            
            return {
                "is_valid": is_valid,
                "error": error
            }
        
        elif tool_name == "validate_llm_api_key":
            provider = tool_args.get("provider", "")
            api_key = tool_args.get("api_key", "")
            is_valid, error = self._validator.validate_llm_api_key(provider, api_key)
            
            return {
                "is_valid": is_valid,
                "error": error
            }
        
        else:
            return {
                "is_valid": False,
                "error": f"Unknown tool: {tool_name}"
            }
            
        




