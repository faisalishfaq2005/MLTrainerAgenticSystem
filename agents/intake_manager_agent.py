
import re
import logging
from dataclasses import dataclass, field
import json
from typing import Optional
from llm.router import LLMRouter
from orchestrator.job_context import JobContext
from agents.base_agent import BaseAgent, AgentError
from agent_support.credentials_validator import CredentialValidator
from llm.prompts.intake_manager_agent_prompt import INTAKE_MANAGER_AGENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal state — lives only inside this agent, never exposed to context
# ---------------------------------------------------------------------------


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
        super().__init__(name="intake_manager", llm_router=llm_router)
        self._state=_IntakeState()
        self._validator=CredentialValidator()

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
            stage="intake",
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
            )

            if not raw_response:
                raise AgentError(agent_name=self.name,stage="intake_manager",reason="LLM returned empty response")
            
            response,data_dict=self.parse_response(raw_response=raw_response)
            self.populate_intake_state(data_dict)
            ready = self.determine_ready_state(data_dict)

            self._state.messages.append({"role":"user","content":user_message})
            self._state.messages.append({"role":"assistant","content":response})

            return {"message":response,"ready":ready}
        except Exception as e:
            raise AgentError(agent_name="intake_manager",stage="intake_manager",reason=str(e)) from e 

    def populate_raw_prompt(self,user_message:str):
        if self._state.raw_prompt is None and len(user_message.strip()) > 10:
            self._state.raw_prompt = user_message.strip()


    def parse_response(self,raw_response):
        try:
            parsed = json.loads(raw_response)

            response = parsed["response"]
            intake_data = parsed["intake_data"]

            # Be tolerant if model accidentally serializes intake_data as a JSON string.
            if isinstance(intake_data, str):
                intake_data = json.loads(intake_data)

            if not isinstance(response, str) or not isinstance(intake_data, dict):
                raise AgentError(
                    agent_name="intake_manager",
                    stage="intake_manager",
                    reason="Invalid response or intake_data type"
                )
                

            return response , intake_data
        except Exception as e:
            raise AgentError(
                agent_name="intake_manager",
                stage="intake_manager",
                reason=f"LLM returned incorrect format of required response: {e}"
            )

    def populate_intake_state(self,data_dict:dict):
        s = self._state

        s.dataset_url = data_dict.get("dataset_url")
        s.dataset_source = data_dict.get("dataset_source")
        s.runtime = data_dict.get("runtime")

        s.hf_token = data_dict.get("hf_token")
        s.hf_username = data_dict.get("hf_username")
        s.hf_repo_name = data_dict.get("hf_repo_name")
        s.hf_org = data_dict.get("hf_org")

        s.kaggle_username = data_dict.get("kaggle_username")
        s.kaggle_key = data_dict.get("kaggle_key")

        # Optional LLM selection fields
        s.llm_provider = data_dict.get("llm_provider")
        s.llm_model = data_dict.get("llm_model")
        s.llm_api_key = data_dict.get("llm_api_key")

    def determine_ready_state(self,data_dict:dict)-> bool:
        # Mandatory fields
        if self._state.raw_prompt is None:
            return False

        if not data_dict.get("dataset_url"):
            return False
        if data_dict.get("dataset_source") not in {"huggingface", "kaggle", "url", "upload"}:
            return False
        if data_dict.get("runtime") not in {"kaggle", "modal"}:
            return False
        if not data_dict.get("hf_token"):
            return False
        if not data_dict.get("hf_repo_name"):
            return False

        # Kaggle creds are conditional
        requires_kaggle = (
            data_dict.get("runtime") == "kaggle"
            or data_dict.get("dataset_source") == "kaggle"
        )
        if requires_kaggle:
            if not data_dict.get("kaggle_username") or not data_dict.get("kaggle_key"):
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