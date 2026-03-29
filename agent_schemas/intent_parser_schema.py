"""
intent_schema.py
----------------
Defines the ONE universal JSON schema that the intent parser always produces.

Design principle:
  - Fields user provided  → populated with their exact values (never overridden later)
  - Fields user omitted   → null (config_agent / architecture_agent fills them in)
  - This is the single source of truth for the entire pipeline.

The schema is represented as a Python TypedDict so it's importable
by all agents for type checking, and also rendered as a JSON template
for the LLM prompt.
"""

from typing import Optional, Literal, TypedDict


# ── Sub-schemas ───────────────────────────────────────────────────────────────

class DatasetSchema(TypedDict):
    url: Optional[str]              # HF / Kaggle / GDrive / direct URL
    source_type: Optional[str]      # "huggingface" | "kaggle" | "gdrive" | "url" | "upload"
    format: Optional[str]           # "csv" | "json" | "jsonl" | "image_folder" | "parquet" | "text"
    text_column: Optional[str]      # for NLP tasks
    label_column: Optional[str]     # target column name
    image_column: Optional[str]     # for multimodal
    train_split: Optional[float]    # e.g. 0.8 — null means auto (80/10/10)
    val_split: Optional[float]
    test_split: Optional[float]
    extra_columns: Optional[list]   # columns to keep beyond text/label


class HyperparametersSchema(TypedDict):
    # All fields nullable — null means "config_agent will decide"
    learning_rate: Optional[float]
    epochs: Optional[int]
    batch_size: Optional[int]
    optimizer: Optional[str]        # "adamw" | "adam" | "sgd" | "adafactor"
    lr_scheduler: Optional[str]     # "cosine" | "linear" | "constant" | "reduce_on_plateau"
    warmup_ratio: Optional[float]   # fraction of steps for warmup
    weight_decay: Optional[float]
    gradient_accumulation_steps: Optional[int]
    max_grad_norm: Optional[float]
    early_stopping_patience: Optional[int]
    seed: Optional[int]


class PeftSchema(TypedDict):
    use_lora: Optional[bool]
    use_qlora: Optional[bool]       # 4-bit quantisation + LoRA
    lora_r: Optional[int]
    lora_alpha: Optional[int]
    lora_dropout: Optional[float]
    target_modules: Optional[list]  # which layers to apply LoRA to


class PrecisionSchema(TypedDict):
    fp16: Optional[bool]
    bf16: Optional[bool]
    use_gradient_checkpointing: Optional[bool]


class ArchitectureSchema(TypedDict):
    # If user said "use EfficientNet with attention" → backbone="efficientnet_b3", custom_components=["attention"]
    # If user said nothing                          → all null, architecture_agent decides
    backbone: Optional[str]         # timm model name or HF model id
    pretrained: Optional[bool]
    custom_components: Optional[list]  # ["attention", "residual", "custom_head", ...]
    head_type: Optional[str]        # "linear" | "mlp" | "custom"
    freeze_backbone: Optional[bool]
    num_classes: Optional[int]      # overrides auto-detect from data


class DeploySchema(TypedDict):
    hf_username: Optional[str]
    hf_org: Optional[str]           # if pushing to an org instead of personal
    hf_repo_name: Optional[str]     # final repo = hf_username/hf_repo_name
    hf_token: Optional[str]         # validated before pipeline starts
    private_repo: Optional[bool]    # default False
    create_space: Optional[bool]    # auto Gradio demo
    model_card_description: Optional[str]  # user-provided blurb for the model card


class CredentialsSchema(TypedDict):
    hf_token: Optional[str]
    kaggle_username: Optional[str]
    kaggle_key: Optional[str]
    llm_provider: Optional[str]     # "anthropic" | "openai" | "google" | "ollama"
    llm_model: Optional[str]        # e.g. "claude-sonnet-4-6", "gpt-4o", "gemini-2.0-flash"
    llm_api_key: Optional[str]


# ── Master intent schema ──────────────────────────────────────────────────────

class IntentSchema(TypedDict):
    # Task identification
    task_type: Optional[str]
    # Supported V1 values:
    #   "tabular_classification" | "tabular_regression"
    #   "image_classification"   | "image_regression"
    #   "text_classification"    | "text_generation"
    #   "llm_finetuning"         | "token_classification"
    #   "summarization"          | "translation"
    #   "object_detection"       | "time_series_forecasting"
    #   "clustering"

    task_description: Optional[str]  # user's own words, preserved verbatim

    # Sub-schemas
    dataset: DatasetSchema
    hyperparameters: HyperparametersSchema
    peft: PeftSchema
    precision: PrecisionSchema
    architecture: ArchitectureSchema
    deploy: DeploySchema
    credentials: CredentialsSchema

    # Runtime
    runtime: Optional[str]           # "kaggle" | "modal" — asked during intake

    # Meta flags (set by intent parser, used by orchestrator)
    user_expertise_level: Optional[str]  # "novice" | "intermediate" | "expert"
    # novice  = almost everything null, pipeline uses all defaults
    # expert  = many fields populated, pipeline respects them strictly


# ── JSON template (used in LLM prompt) ───────────────────────────────────────

INTENT_JSON_TEMPLATE = """{
  "task_type": null,
  "task_description": null,
  "user_expertise_level": null,

  "dataset": {
    "url": null,
    "source_type": null,
    "format": null,
    "text_column": null,
    "label_column": null,
    "image_column": null,
    "train_split": null,
    "val_split": null,
    "test_split": null,
    "extra_columns": null
  },

  "hyperparameters": {
    "learning_rate": null,
    "epochs": null,
    "batch_size": null,
    "optimizer": null,
    "lr_scheduler": null,
    "warmup_ratio": null,
    "weight_decay": null,
    "gradient_accumulation_steps": null,
    "max_grad_norm": null,
    "early_stopping_patience": null,
    "seed": null
  },

  "peft": {
    "use_lora": null,
    "use_qlora": null,
    "lora_r": null,
    "lora_alpha": null,
    "lora_dropout": null,
    "target_modules": null
  },

  "precision": {
    "fp16": null,
    "bf16": null,
    "use_gradient_checkpointing": null
  },

  "architecture": {
    "backbone": null,
    "pretrained": null,
    "custom_components": null,
    "head_type": null,
    "freeze_backbone": null,
    "num_classes": null
  },

  "deploy": {
    "hf_username": null,
    "hf_org": null,
    "hf_repo_name": null,
    "hf_token": null,
    "private_repo": null,
    "create_space": null,
    "model_card_description": null
  },

  "credentials": {
    "hf_token": null,
    "kaggle_username": null,
    "kaggle_key": null,
    "llm_provider": null,
    "llm_model": null,
    "llm_api_key": null
  },

  "runtime": null
}"""