"""
demo_intake_flow.py
--------------------
Demonstrates the correct two-stage flow:

  Stage 1: IntakeManagerAgent   — conversational, writes to context
  Stage 2: IntentParserAgent    — LLM call, writes parsed_intent to context

Run:
    python demo_intake_flow.py
"""

import json
from agents.b import IntakeManagerAgent
from agents.intent_parser_agent import IntentParserAgent
from orchestrator.job_context import JobContext


# ---------------------------------------------------------------------------
# Mock LLM router (replace with real llm/router.py in production)
# ---------------------------------------------------------------------------

class MockLLMRouter:
    """
    Simulates llm/router.py for demo purposes.
    In production: wraps LiteLLM → Claude / GPT-4o / Gemini.
    """
    def complete(self, system_prompt: str, user_message: str) -> str:
        lower = user_message.lower()

        # Detect task type
        if "mistral" in lower or "llama" in lower or "qlora" in lower or "fine-tune" in lower:
            task = "llm_finetuning"
        elif "efficientnet" in lower or "image classif" in lower or "dog breed" in lower:
            task = "image_classification"
        elif "house price" in lower or "regression" in lower:
            task = "tabular_regression"
        else:
            task = "text_classification"

        expertise = "expert" if any(
            s in lower for s in ["lr=", "lora", "qlora", "fp16", "bf16", "batch_size", "r=16"]
        ) else "novice"

        # Extract collected_info values the LLM should embed from the verified section
        import re
        def extract(pattern, text, default=None):
            m = re.search(pattern, text)
            return m.group(1) if m else default

        hf_token     = extract(r'hf_token:\s*(\S+)',     user_message)
        hf_username  = extract(r'hf_username:\s*(\S+)',  user_message)
        hf_repo      = extract(r'hf_repo_name:\s*(\S+)', user_message)
        hf_org       = extract(r'hf_org:\s*(\S+)',       user_message)
        dataset_url  = extract(r'dataset_url:\s*(\S+)',  user_message)
        ds_source    = extract(r'dataset_source:\s*(\S+)', user_message)
        runtime      = extract(r'runtime:\s*(\S+)',      user_message)
        llm_provider = extract(r'llm_provider:\s*(\S+)', user_message)
        llm_model    = extract(r'llm_model:\s*(\S+)',    user_message)
        llm_key      = extract(r'llm_api_key:\s*(\S+)',  user_message)
        kaggle_user  = extract(r'kaggle_username:\s*(\S+)', user_message)
        kaggle_key   = extract(r'kaggle_key:\s*(\S+)',   user_message)
        backbone     = "mistralai/Mistral-7B-v0.1" if "mistral" in lower else (
                       "efficientnet_b3" if "efficientnet" in lower else None)

        return json.dumps({
            "task_type": task,
            "task_description": "Extracted from user's training request.",
            "user_expertise_level": expertise,
            "dataset": {
                "url": dataset_url, "source_type": ds_source,
                "format": None, "text_column": None, "label_column": None,
                "image_column": None, "train_split": None, "val_split": None,
                "test_split": None, "extra_columns": None,
            },
            "hyperparameters": {
                "learning_rate": 0.0002 if "lr=2e-4" in lower else None,
                "epochs": 3 if "3 epoch" in lower else None,
                "batch_size": 4 if "batch=4" in lower else None,
                "optimizer": None, "lr_scheduler": None, "warmup_ratio": None,
                "weight_decay": None,
                "gradient_accumulation_steps": 8 if "grad_accum=8" in lower else None,
                "max_grad_norm": None, "early_stopping_patience": None, "seed": None,
            },
            "peft": {
                "use_lora": True if "lora" in lower else None,
                "use_qlora": True if "qlora" in lower else None,
                "lora_r": 16 if "r=16" in lower else None,
                "lora_alpha": 32 if "alpha=32" in lower else None,
                "lora_dropout": None, "target_modules": None,
            },
            "precision": {
                "fp16": True if "fp16" in lower else None,
                "bf16": True if "bf16" in lower else None,
                "use_gradient_checkpointing": None,
            },
            "architecture": {
                "backbone": backbone, "pretrained": True if backbone else None,
                "custom_components": ["self_attention"] if "attention" in lower else None,
                "head_type": None, "freeze_backbone": None, "num_classes": None,
            },
            "deploy": {
                "hf_username": hf_username, "hf_org": hf_org,
                "hf_repo_name": hf_repo, "hf_token": hf_token,
                "private_repo": None, "create_space": None,
                "model_card_description": None,
            },
            "credentials": {
                "hf_token": hf_token,
                "kaggle_username": kaggle_user, "kaggle_key": kaggle_key,
                "llm_provider": llm_provider, "llm_model": llm_model,
                "llm_api_key": llm_key,
            },
            "runtime": runtime,
        })


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def simulate(label: str, conversation: list[tuple[str, str]]) -> JobContext | None:
    """
    Runs a scripted conversation through IntakeManagerAgent,
    then passes the result to IntentParserAgent.
    Returns the final JobContext, or None if intake didn't complete.
    """
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}\n")

    # Stage 1: IntakeManagerAgent
    agent = IntakeManagerAgent(bypass_validation=True)
    print(f"INTAKE AGENT: {agent.get_opening_message()}\n")

    context = JobContext()
    intake_done = False

    for user_msg, desc in conversation:
        print(f"  [{desc}]")
        print(f"  USER:  {user_msg}")
        result = agent.run_turn(user_msg)
        print(f"  AGENT: {result['message'][:200]}")
        print(f"         stage={result['stage']}, ready={result['ready']}\n")

        if result["ready"]:
            # Write raw_prompt, conversation_history, collected_info into context
            agent.finalise(context)
            intake_done = True
            break

    if not intake_done:
        print("  [Intake did not complete within the scripted conversation]")
        return None

    # Verify context fields are clean
    print(f"\n  context.raw_prompt     = {context.raw_prompt[:60]}...")
    print(f"  context.collected_info = {list(context.collected_info.keys())}")
    print(f"  context.parsed_intent  = {context.parsed_intent}  (not yet — next stage)")

    # Stage 2: IntentParserAgent
    print(f"\n{'─'*70}")
    print(f"  INTENT PARSER STAGE")
    print(f"{'─'*70}\n")
    llm_router=MockLLMRouter()
    parser = IntentParserAgent(llm_router=llm_router)
    result = parser.run(context)

    # The orchestrator would normally merge result into context like this:
    context.parsed_intent = result["parsed_intent"]

    print("  parsed_intent written to context.parsed_intent")
    print(f"\n  FINAL parsed_intent JSON:")
    print(json.dumps(context.parsed_intent, indent=2))

    # Summary
    pi = context.parsed_intent
    print(f"\n  task_type:       {pi['task_type']}")
    print(f"  expertise:       {pi['user_expertise_level']}")
    print(f"  backbone:        {pi['architecture']['backbone']}")
    print(f"  learning_rate:   {pi['hyperparameters']['learning_rate']}")
    print(f"  use_qlora:       {pi['peft']['use_qlora']}")
    print(f"  runtime:         {pi['runtime']}")
    print(f"  hf_token:        {(pi['credentials']['hf_token'] or '')[:20]}...")
    print(f"  dataset.url:     {pi['dataset']['url']}")

    return context


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ── Scenario 1: Novice user (spam classifier, Kaggle runtime) ────────────
    simulate("NOVICE — Spam detector (Kaggle)", [
        ("I want to train a model to detect spam SMS messages",
         "Initial task"),
        ("https://huggingface.co/datasets/sms_spam",
         "Dataset URL"),
        ("kaggle",
         "Runtime"),
        ("hf_FakeTok1abcdefgh",
         "HF token"),
        ("spam-detector",
         "Repo name"),
        ("username: john_doe\nkey: abcdef1234567890abcdef1234567890",
         "Kaggle credentials"),
        ("anthropic",
         "LLM provider"),
        ("claude-sonnet-4-6",
         "LLM model"),
        ("sk-ant-FakeAnthropicKey",
         "LLM API key"),
    ])

    # ── Scenario 2: Expert user (QLoRA, Modal runtime) ───────────────────────
    simulate("EXPERT — Mistral-7B QLoRA fine-tune (Modal)", [
        ("Fine-tune mistralai/Mistral-7B-v0.1 on my instruction dataset using QLoRA. "
         "r=16, alpha=32, lr=2e-4, 3 epochs, batch=4, grad_accum=8, bf16.",
         "Full expert prompt"),
        ("https://huggingface.co/datasets/my-org/instructions",
         "Dataset URL"),
        ("modal",
         "Runtime"),
        ("hf_FakeTok2expertxyz",
         "HF token"),
        ("mistral-finetuned",
         "Repo name"),
        ("anthropic",
         "LLM provider"),
        ("claude-sonnet-4-6",
         "LLM model"),
        ("sk-ant-FakeAnthropicKey",
         "LLM API key"),
    ])

    # ── Scenario 3: Custom architecture (EfficientNet + attention) ───────────
    simulate("INTERMEDIATE — EfficientNet + attention (Modal)", [
        ("Train image classifier for 10 product categories. "
         "EfficientNet-B3 backbone with self-attention mechanism.",
         "Task + architecture"),
        ("https://drive.google.com/file/d/abc123/view",
         "Dataset URL"),
        ("modal",
         "Runtime"),
        ("hf_FakeTok3visionabc",
         "HF token"),
        ("product-classifier",
         "Repo name"),
        ("openai",
         "LLM provider"),
        ("gpt-4o",
         "LLM model"),
        ("sk-FakeOpenAIKey",
         "LLM API key"),
    ])

    print(f"\n{'='*70}")
    print("  All simulations complete.")
    print(f"{'='*70}\n")