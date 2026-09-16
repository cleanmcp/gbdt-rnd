"""Evidence-locked StoryCards with caching and an explicit LLM budget."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .contracts import AccountSnapshot, FactSheet, IcpSpec, ProductSpec, StoryCard
from .hashing import sha256_json
from .store import ExperimentStore


class StoryNarration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    story_type: str
    phase: str
    confidence: float = Field(ge=0, le=1)
    summary: str
    implications: tuple[str, ...]
    evidence_fact_ids: tuple[str, ...]


class NarrationProvider(Protocol):
    model_id: str

    async def narrate(
        self,
        *,
        product: ProductSpec,
        icp: IcpSpec,
        snapshot: AccountSnapshot,
        fact_payload: list[dict[str, object]],
    ) -> StoryNarration: ...


class TemplateNarrator:
    """Zero-cost provider used by tests and by default in local runs."""

    model_id = "deterministic-template-v1"

    async def narrate(
        self,
        *,
        product: ProductSpec,
        icp: IcpSpec,
        snapshot: AccountSnapshot,
        fact_payload: list[dict[str, object]],
    ) -> StoryNarration:
        observed = [
            signal_type for signal_type, status in snapshot.coverage.items() if status == "observed"
        ]
        return StoryNarration(
            story_type="multi_signal_opportunity",
            phase="active",
            confidence=min(0.45 + 0.08 * len(observed), 0.9),
            summary=(
                f"{product.name} opportunity for a {icp.name} account, supported by "
                f"{', '.join(observed[:4]) or 'the available account evidence'}."
            ),
            implications=(
                "Review the cited operational signals before contact.",
                "Treat the score as an opportunity hypothesis until real outcomes calibrate it.",
            ),
            evidence_fact_ids=tuple(snapshot.evidence_fact_ids[:12]),
        )


class OpenAiNarrator:
    prompt_version = "account-story-v1"

    def __init__(self, model_id: str):
        self.model_id = model_id

    async def narrate(
        self,
        *,
        product: ProductSpec,
        icp: IcpSpec,
        snapshot: AccountSnapshot,
        fact_payload: list[dict[str, object]],
    ) -> StoryNarration:
        try:
            from openai import AsyncOpenAI
        except ImportError as error:
            raise RuntimeError("OpenAI narration requires the `llm` project extra") from error
        client = AsyncOpenAI()
        response = await client.responses.parse(
            model=self.model_id,
            max_output_tokens=800,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Create a concise B2B opportunity StoryCard. Use only supplied facts. "
                        "Every implication must be supported by evidence_fact_ids. Never state "
                        "that the account will buy; describe an evidence-based hypothesis."
                    ),
                },
                {
                    "role": "user",
                    "content": str(
                        {
                            "product": product.model_dump(mode="json"),
                            "icp": icp.model_dump(mode="json"),
                            "account_features": snapshot.features,
                            "coverage": snapshot.coverage,
                            "facts": fact_payload,
                        }
                    ),
                },
            ],
            text_format=StoryNarration,
        )
        if response.output_parsed is None:
            raise RuntimeError("OpenAI returned no parsed StoryCard")
        allowed = set(snapshot.evidence_fact_ids)
        if not set(response.output_parsed.evidence_fact_ids).issubset(allowed):
            raise RuntimeError("StoryCard referenced unsupported evidence")
        return response.output_parsed


class StoryBudget:
    def __init__(self, maximum: int):
        self.maximum = maximum
        self.used = 0

    def claim(self) -> bool:
        if self.used >= self.maximum:
            return False
        self.used += 1
        return True


def deterministic_signal_story(sheet: FactSheet) -> StoryCard:
    fact_map = {fact.name: fact.value for fact in sheet.facts}
    terminal = bool(fact_map.get("is_terminal", False))
    material_changes = int(fact_map.get("material_change_count", 0) or 0)
    phase = "resolved" if terminal else "escalating" if material_changes >= 2 else "active"
    story_type = f"{sheet.signal_type}_{phase}"
    summary = (
        f"{sheet.signal_type.replace('_', ' ').title()} is {phase}; "
        f"{int(fact_map.get('event_count', 1) or 1)} observations are visible "
        f"at {sheet.as_of.date().isoformat()}."
    )
    created_at = datetime.now(UTC)
    return StoryCard(
        story_card_id=sha256_json(
            {"scope": "signal", "subject": sheet.signal_instance_id, "hash": sheet.semantic_hash}
        ),
        scope="signal",
        subject_id=sheet.signal_instance_id,
        story_type=story_type,
        phase=phase,
        confidence=0.85,
        summary=summary,
        evidence_fact_ids=tuple(fact.fact_id for fact in sheet.facts),
        semantic_hash=sheet.semantic_hash,
        generator="deterministic",
        created_at=created_at,
    )


async def account_story(
    *,
    store: ExperimentStore,
    provider: NarrationProvider,
    budget: StoryBudget,
    product: ProductSpec,
    icp: IcpSpec,
    snapshot: AccountSnapshot,
    fact_payload: list[dict[str, object]],
) -> StoryCard | None:
    semantic_hash = sha256_json(
        {
            "snapshot_hash": snapshot.snapshot_hash,
            "product_version": product.product_version,
            "icp_version": icp.icp_version,
            "provider": provider.model_id,
        }
    )
    generator: Literal["deterministic", "llm"] = (
        "deterministic" if isinstance(provider, TemplateNarrator) else "llm"
    )
    cached = store.story_card("account", snapshot.account_id, semantic_hash, generator)
    if cached:
        return cached
    if not budget.claim():
        return None
    narration = await provider.narrate(
        product=product,
        icp=icp,
        snapshot=snapshot,
        fact_payload=fact_payload,
    )
    card = StoryCard(
        story_card_id=sha256_json(
            {"scope": "account", "subject": snapshot.account_id, "hash": semantic_hash}
        ),
        scope="account",
        subject_id=snapshot.account_id,
        story_type=narration.story_type,
        phase=narration.phase,
        confidence=narration.confidence,
        summary=narration.summary,
        implications=narration.implications,
        evidence_fact_ids=narration.evidence_fact_ids,
        semantic_hash=semantic_hash,
        generator=generator,
        model=provider.model_id,
        prompt_version=getattr(provider, "prompt_version", None),
        created_at=datetime.now(UTC),
    )
    store.save_story_card(card)
    return card
