"""
The Fork — Discipline Agent Manifest Models

Typed Pydantic models for loading/validating agent manifests.
Mirrors RetailOps pattern, adapted for construction disciplines.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class Discipline(str, Enum):
    BASE = "base"
    PLANNING = "planning"
    COMMERCIAL = "commercial"
    CONTRACTS = "contracts"
    QAQC = "qaqc"
    PROCUREMENT = "procurement"
    QUANTITIES = "quantities"
    SAFETY = "safety"
    DESIGN = "design"


class ManifestKind(str, Enum):
    BASE = "base"
    HAT = "hat"


class ActivationMode(str, Enum):
    MANUAL = "manual"
    AUTO = "auto"
    HYBRID = "hybrid"


class GroundingPolicy(str, Enum):
    DOCUMENTS_OR_LEARNING_ONLY = "documents_or_learning_only"
    FORMULAS_ONLY = "formulas_only"
    DOCUMENTS_OR_FORMULAS = "documents_or_formulas"
    NONE = "none"


class MissingContextPolicy(str, Enum):
    DEPENDENCY_REQUIRED = "dependency_required"
    ASK_CLARIFYING = "ask_clarifying"
    USE_BEST_EFFORT = "use_best_effort"
    REFUSE = "refuse"


class TriggerType(str, Enum):
    KEYWORD = "keyword"
    INTENT = "intent"
    ACTION_ID = "action_id"
    FORMULA_ID = "formula_id"


class ContextSourceKind(str, Enum):
    PROJECT_DOCUMENTS = "project_documents"
    OFFICIAL_GUIDANCE = "official_guidance"
    LEARNING_PROFILES = "learning_profiles"
    INDUSTRY_BENCHMARKS = "industry_benchmarks"
    ACTION_METADATA = "action_metadata"


class FormulaTier(str, Enum):
    FOUNDATIONAL = "foundational"
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"


# -- Sub-models -----------------------------------------------------------

class ActivationTrigger(BaseModel):
    type: TriggerType
    value: str = ""
    score: float = 0.5
    case_sensitive: bool = False


class ActivationConfig(BaseModel):
    triggers: List[ActivationTrigger]
    mode: ActivationMode = ActivationMode.HYBRID
    default_enabled: bool = True
    put_off_by: Optional[str] = None


class ContextSource(BaseModel):
    kind: ContextSourceKind
    weight: float = 1.0
    filters: Optional[Dict[str, Any]] = None
    max_items: Optional[int] = None


class HandoffRule(BaseModel):
    to: str
    when: str
    carry: Optional[List[str]] = None


class MemoryConfig(BaseModel):
    save_on_turn_end: Optional[List[str]] = None
    never_persist: Optional[List[str]] = None


class Verification(BaseModel):
    grounding: GroundingPolicy = GroundingPolicy.DOCUMENTS_OR_FORMULAS
    cite_required: bool = True
    refuse_unaided_figures: bool = True
    standards_advisory: bool = True


class FailurePolicy(BaseModel):
    on_missing_context: MissingContextPolicy = MissingContextPolicy.ASK_CLARIFYING
    on_formula_error: MissingContextPolicy = MissingContextPolicy.ASK_CLARIFYING
    on_unverified_claim: str = "flag_and_continue"


class PlaybookFormula(BaseModel):
    id: str
    description: str
    implemented_by: Optional[str] = None
    domain: Optional[str] = None
    tier: FormulaTier = FormulaTier.OPERATIONAL


class PlaybookFlow(BaseModel):
    id: str
    description: str
    steps: List[str] = Field(default_factory=list)


class Playbook(BaseModel):
    formulas: List[PlaybookFormula]
    flows: Optional[List[PlaybookFlow]] = None


# -- Main Manifest Model --------------------------------------------------

class AgentManifest(BaseModel):
    manifest_version: str = "1.0"
    id: str
    name: str
    description: str
    kind: ManifestKind
    extends: Optional[str] = None
    discipline: Discipline
    system_prompt_guidance: str
    activation: ActivationConfig
    context_sources: List[ContextSource]
    allowed_actions: List[str] = Field(default_factory=list)
    #: The tools this hat's package registers (app.agents.base / hats.<hat>);
    #: tests/test_manifest_tools_match_packages.py keeps the two equal.
    available_tools: List[str] = Field(default_factory=list)
    handoffs: Optional[List[HandoffRule]] = None
    memory: Optional[MemoryConfig] = None
    verification: Verification
    failure_policy: FailurePolicy
    playbook: Playbook

    @field_validator("extends")
    @classmethod
    def validate_extends(cls, v: Optional[str], info) -> Optional[str]:
        values = info.data
        if values.get("kind") == ManifestKind.HAT and not v:
            raise ValueError("Hats must declare 'extends' pointing to base manifest")
        return v

    @property
    def is_base(self) -> bool:
        return self.kind == ManifestKind.BASE

    @property
    def is_hat(self) -> bool:
        return self.kind == ManifestKind.HAT

    @property
    def formula_ids(self) -> List[str]:
        return [f.id for f in self.playbook.formulas]

    @property
    def formula_bindings(self) -> Dict[str, Optional[str]]:
        return {f.id: f.implemented_by for f in self.playbook.formulas if f.implemented_by}

    @property
    def trigger_keywords(self) -> List[str]:
        return [t.value for t in self.activation.triggers if t.type == TriggerType.KEYWORD and t.value]

    @property
    def trigger_intents(self) -> List[str]:
        return [t.value for t in self.activation.triggers if t.type == TriggerType.INTENT and t.value]

    def score_message(self, message: str) -> float:
        """Score a user message against this manifest's triggers."""
        message_lower = message.lower()
        total_score = 0.0
        for trigger in self.activation.triggers:
            if trigger.type == TriggerType.KEYWORD and trigger.value:
                check = message if trigger.case_sensitive else message_lower
                val = trigger.value if trigger.case_sensitive else trigger.value.lower()
                if val in check:
                    total_score += trigger.score
        return total_score

    def get_binding(self, formula_id: str) -> Optional[str]:
        """Get the implemented_by path for a formula id."""
        for f in self.playbook.formulas:
            if f.id == formula_id:
                return f.implemented_by
        return None
