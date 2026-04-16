
import re
import logging
from dataclasses import dataclass, field
import json
from typing import Optional, Any
from llm.router import LLMRouter
from orchestrator.job_context import JobContext
from agents.base_agent import BaseAgent, AgentError
from tool.credential_validator_tool import CredentialValidatorTools
from llm.prompts.intake_manager_agent_prompt import INTAKE_MANAGER_AGENT_SYSTEM_PROMPT
from tool.tool_executer import ToolExecuter
from agents.agent_names import Agents
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal state — lives only inside this agent, never exposed to context
# ---------------------------------------------------------------------------
ToolExecuter.register(Agents.INTAKE_MANAGER_AGENT, CredentialValidatorTools)


@dataclass
class _IntakeState:
    raw_prompt: Optional[str] = None

    # Dataset
    dataset_url: Optional[str] = None
    dataset_source: Optional[str] = None   # "huggingface"|"kaggle"|"url"|"upload"

    # Runtime
    runtime: Optional[str] = None          # "kaggle"|"modal"

    # HuggingFace
    hf_token: Optional[str] = None
    hf_username: Optional[str] = None      # fetched from HF API
    hf_repo_name: Optional[str] = None
    hf_org: Optional[str] = None

    # Kaggle (conditional)
    kaggle_username: Optional[str] = None
    kaggle_key: Optional[str] = None

    # LLM for the agent system
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None

    # Validation flags
    hf_token_valid: bool = False
    kaggle_creds_valid: bool = False
    llm_key_valid: bool = False

    # Error messages to surface to user on retry
    _hf_token_error: Optional[str] = None
    _kaggle_error: Optional[str] = None
    _llm_key_error: Optional[str] = None

    # Full conversation for context.conversation_history
    messages: list = field(default_factory=list)








class IntakeManagerAgent(BaseAgent):

    def __init__(self,llm_router):
        super().__init__(name=Agents.INTAKE_MANAGER_AGENT, llm_router=llm_router)
        self._state=_IntakeState()
        self._validator_tools=ToolExecuter(Agents.INTAKE_MANAGER_AGENT)
        

    #orhestrator will fisrt call get_opening_message and then converse function turn by turn until the converse function returns ready:True
    # the loop of calling converse till ready:True, will be in orchestrator , not here, because orchestrator is interacting wih frontend api and needs to send llm response to frontend and user message to agent
    #orchestrator will call finalize when ready==True
    def _execute(self, context):
        """
        Not used directly — intake is interactive.
        The orchestrator uses run_turn() + finalise() instead.
        This method exists to satisfy BaseAgent's abstract contract and is
        called only if someone mistakenly calls agent.run(context) directly.
        """
        raise AgentError(
            agent_name=self.name,
            stage=Agents.INTAKE_MANAGER_AGENT,
            reason=(
                "IntakeManagerAgent is interactive and cannot be run via run(context). "
                "Use run_turn(user_message) in a loop, then finalise(context)."
            ),
        )

    def get_opening_message(self) -> str:
        opening_message= """👋 Welcome to the ML Training Agent!\n\n
            I'll help you train and deploy a model to HuggingFace — automatically.\n\n
            Tell me what you want to build. For example:\n
              • \"Train a spam email classifier on this dataset: [link]\"\n
              • \"Fine-tune Mistral-7B on my instruction dataset with QLoRA\"\n
              • \"Image classifier for 5 dog breeds, EfficientNet backbone\"\n\n
            You can give as much or as little detail as you like — 
            I'll ask for anything I need."""
        self._state.messages.append({"role":"assistant","content":opening_message})
        return opening_message
       
   

#make a tool olso that the llm can call to see if all required info is collected 
    def converse(self, user_message:str) -> dict:
        try:
            self.populate_raw_prompt(user_message=user_message)

            raw_response=self.llm_router.complete(
                system_prompt=INTAKE_MANAGER_AGENT_SYSTEM_PROMPT,
                user_message=user_message,
                message_history=self._state.messages,
                tools=self._validator_tools.get_tool_definitions(),
                tool_choice="auto"
            )

            if not raw_response:
                raise AgentError(agent_name=self.name,stage=Agents.INTAKE_MANAGER_AGENT,reason="LLM returned empty response")

            self._state.messages.append({"role":"user","content":user_message})

            response, data_dict, tool_locked_fields = self._resolve_response_with_tools(raw_response=raw_response)

            self.populate_intake_state(data_dict, tool_locked_fields=tool_locked_fields)
            ready = self.determine_ready_state()

            self._state.messages.append({"role":"assistant","content":response})

            return {"message":response,"ready":ready}
        except Exception as e:
            raise AgentError(agent_name=Agents.INTAKE_MANAGER_AGENT,stage=Agents.INTAKE_MANAGER_AGENT,reason=str(e)) from e 
        
    

    def populate_raw_prompt(self,user_message:str):
        if self._state.raw_prompt is None and len(user_message.strip()) > 10:
            self._state.raw_prompt = user_message.strip()

    def _resolve_response_with_tools(self, raw_response: Any) -> tuple[str, dict, set[str]]:
        current_response = raw_response
        max_tool_rounds = 3
        tool_locked_fields: set[str] = set()

        for _ in range(max_tool_rounds):
            if isinstance(current_response, dict) and current_response.get("tool_calls"):
                self._append_assistant_tool_call_message(current_response)

                for tool in current_response["tool_calls"]:
                    tool_name = tool.get("name", "")
                    tool_args = tool.get("arguments", {})
                    tool_result = self._validator_tools.execute_tool(tool_name=tool_name, tool_args=tool_args)
                    self._apply_tool_result(tool_name=tool_name, tool_args=tool_args, tool_result=tool_result)
                    tool_locked_fields.update(self._locked_fields_for_tool(tool_name))

                    self._state.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool.get("id", ""),
                            "name": tool_name,
                            "content": json.dumps(tool_result),
                        }
                    )

                current_response = self.llm_router.complete(
                    system_prompt=INTAKE_MANAGER_AGENT_SYSTEM_PROMPT,
                    user_message=None,
                    message_history=self._state.messages,
                    tools=self._validator_tools.get_tool_definitions(),
                    tool_choice="auto",
                )
                continue

            response, data_dict = self.parse_response(raw_response=current_response)
            return response, data_dict, tool_locked_fields

        raise AgentError(
            agent_name=Agents.INTAKE_MANAGER_AGENT,
            stage=Agents.INTAKE_MANAGER_AGENT,
            reason="Too many sequential tool-calling rounds without final response",
        )

    def _append_assistant_tool_call_message(self, llm_response_with_tools: dict) -> None:
        tool_calls_for_history = []
        for tc in llm_response_with_tools.get("tool_calls", []):
            tool_calls_for_history.append(
                {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": json.dumps(tc.get("arguments", {})),
                    },
                }
            )

        self._state.messages.append(
            {
                "role": "assistant",
                "content": llm_response_with_tools.get("content") or "",
                "tool_calls": tool_calls_for_history,
            }
        )

    def _apply_tool_result(self, tool_name: str, tool_args: dict, tool_result: dict) -> None:
        s = self._state
        is_valid = bool(tool_result.get("is_valid"))
        error = tool_result.get("error")

        if tool_name == "validate_hf_token":
            s.hf_token = tool_args.get("token") or s.hf_token
            s.hf_token_valid = is_valid
            s._hf_token_error = error
            if tool_result.get("username"):
                s.hf_username = tool_result["username"]

        elif tool_name == "validate_kaggle_credentials":
            s.kaggle_username = tool_args.get("username") or s.kaggle_username
            s.kaggle_key = tool_args.get("key") or s.kaggle_key
            s.kaggle_creds_valid = is_valid
            s._kaggle_error = error

        elif tool_name == "validate_llm_api_key":
            s.llm_provider = tool_args.get("provider") or s.llm_provider
            s.llm_api_key = tool_args.get("api_key") or s.llm_api_key
            s.llm_key_valid = is_valid
            s._llm_key_error = error

    def _locked_fields_for_tool(self, tool_name: str) -> set[str]:
        if tool_name == "validate_hf_token":
            return {"hf_token", "hf_username"}
        if tool_name == "validate_kaggle_credentials":
            return {"kaggle_username", "kaggle_key"}
        if tool_name == "validate_llm_api_key":
            return {"llm_provider", "llm_api_key"}
        return set()



    def parse_response(self,raw_response):
        try:
            if isinstance(raw_response, dict):
                raise AgentError(
                    agent_name=Agents.INTAKE_MANAGER_AGENT,
                    stage=Agents.INTAKE_MANAGER_AGENT,
                    reason="Expected final JSON string response, received unresolved tool-call payload",
                )

            parsed = json.loads(raw_response)

            response = parsed["response"]
            intake_data = parsed["intake_data"]

            # Be tolerant if model accidentally serializes intake_data as a JSON string.
            if isinstance(intake_data, str):
                intake_data = json.loads(intake_data)

            if not isinstance(response, str) or not isinstance(intake_data, dict):
                raise AgentError(
                    agent_name=Agents.INTAKE_MANAGER_AGENT,
                    stage=Agents.INTAKE_MANAGER_AGENT,
                    reason="Invalid response or intake_data type"
                )
                

            return response , intake_data
        except Exception as e:
            raise AgentError(
                agent_name=Agents.INTAKE_MANAGER_AGENT,
                stage=Agents.INTAKE_MANAGER_AGENT,
                reason=f"LLM returned incorrect format of required response: {e}"
            )

    def populate_intake_state(self, data_dict: dict, tool_locked_fields: Optional[set[str]] = None):
        s = self._state
        locked = tool_locked_fields or set()

        def _is_new_value(field: str) -> bool:
            incoming = data_dict.get(field)
            return incoming is not None and incoming != getattr(s, field)

        s.dataset_url = data_dict.get("dataset_url") or s.dataset_url
        s.dataset_source = data_dict.get("dataset_source") or s.dataset_source
        s.runtime = data_dict.get("runtime") or s.runtime

        # Credential fields are tool-authoritative when that tool ran this turn.
        # If LLM changes credential values without tool validation, invalidate status.
        if "hf_token" not in locked and _is_new_value("hf_token"):
            s.hf_token = data_dict.get("hf_token")
            s.hf_token_valid = False
            s._hf_token_error = "HuggingFace token changed; needs validation."

        if "hf_username" not in locked:
            s.hf_username = data_dict.get("hf_username") or s.hf_username

        s.hf_repo_name = data_dict.get("hf_repo_name") or s.hf_repo_name
        s.hf_org = data_dict.get("hf_org") or s.hf_org

        kaggle_changed = False
        if "kaggle_username" not in locked and _is_new_value("kaggle_username"):
            s.kaggle_username = data_dict.get("kaggle_username")
            kaggle_changed = True
        if "kaggle_key" not in locked and _is_new_value("kaggle_key"):
            s.kaggle_key = data_dict.get("kaggle_key")
            kaggle_changed = True
        if kaggle_changed:
            s.kaggle_creds_valid = False
            s._kaggle_error = "Kaggle credentials changed; needs validation."

        # Optional LLM selection fields
        if "llm_provider" not in locked:
            s.llm_provider = data_dict.get("llm_provider") or s.llm_provider
        s.llm_model = data_dict.get("llm_model") or s.llm_model

        if "llm_api_key" not in locked and _is_new_value("llm_api_key"):
            s.llm_api_key = data_dict.get("llm_api_key")
            if s.llm_provider in {"anthropic", "openai", "google"}:
                s.llm_key_valid = False
                s._llm_key_error = "LLM API key changed; needs validation."

    def determine_ready_state(self)-> bool:
        # Mandatory fields
        if self._state.raw_prompt is None:
            return False

        if not self._state.dataset_url:
            return False
        if self._state.dataset_source not in {"huggingface", "kaggle", "url", "upload"}:
            return False
        if self._state.runtime not in {"kaggle", "modal"}:
            return False
        if not self._state.hf_token:
            return False
        if not self._state.hf_repo_name:
            return False
        if not self._state.hf_token_valid:
            return False

        # Kaggle creds are conditional
        requires_kaggle = (
            self._state.runtime == "kaggle"
            or self._state.dataset_source == "kaggle"
        )
        if requires_kaggle:
            if not self._state.kaggle_username or not self._state.kaggle_key:
                return False
            if not self._state.kaggle_creds_valid:
                return False

        # LLM settings are optional globally, but once a paid provider key is given,
        # ensure that key is validated before readiness.
        if self._state.llm_provider in {"anthropic", "openai", "google"}:
            if not self._state.llm_api_key:
                return False
            if not self._state.llm_key_valid:
                return False

        # llm_provider / llm_model / llm_api_key are optional by design
        return True

    def finalize(self,context)-> None:
        """
        Write the three intake outputs into JobContext.
        Call this once run_turn() returns ready=True.

        Writes:
            context.raw_prompt           str
            context.conversation_history list
            context.collected_info       dict
        """
        s = self._state
        context.raw_prompt = s.raw_prompt
        context.conversation_history = list(s.messages)
        context.collected_info = {
            "dataset_url":      s.dataset_url,
            "dataset_source":   s.dataset_source,
            "runtime":          s.runtime,
            "hf_token":         s.hf_token,
            "hf_username":      s.hf_username,
            "hf_repo_name":     s.hf_repo_name,
            "hf_org":           s.hf_org,
            "kaggle_username":  s.kaggle_username,
            "kaggle_key":       s.kaggle_key,
            "llm_provider":     s.llm_provider,
            "llm_model":        s.llm_model,
            "llm_api_key":      s.llm_api_key,
        }