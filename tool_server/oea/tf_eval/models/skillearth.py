"""
SkillEarth evaluator model for OpenEarthAgent (tf_eval).

This model plugs the SkillEarth *skillbank* into the OEA evaluation loop. Per
requirement (8), on each turn it:

  1. Retrieves the most relevant skill from the skillbank for the user's question
     (built by the local ``skillearth`` package).
  2. Integrates that skill (its When-to-use / Plan / tool-list) into the system
     context alongside the standard RS_AGENT_PROMPT.
  3. Produces exactly ONE tool call in the OEA JSON action format.
  4. The inferencer executes the tool and feeds the observation back; the model
     is called again to predict the next tool call, and so on, until it emits a
     ``Terminate`` action carrying the final answer.

It follows the ``tp_model`` interface (see ``abstract_model.py``) and mirrors the
conversation/generate contract of ``OpenAIModels`` (``openai.py``), so it works
unchanged in both ``e2e`` (multi-round, live tools) and ``step`` modes.

Backend:
  * ``planner="llm"`` : delegate reasoning to an OpenAI-compatible chat model
    (via ``SKILLEARTH_*`` / ``OPENAI_*`` env), with the retrieved skill injected.

Evaluation never silently falls back to replay.  Skillbank or LLM initialization
errors abort the run so a failed setup cannot look like a successful evaluation.

Usage in eval_e2e.sh:
    --model skillearth \
    --model_args pretrained=gpt-4o-mini,skillbank_dir=/abs/path/OpenEarthAgent/skillbank,planner=llm,top_k=1

Also register it in ``models/__init__.py``:
    AVAILABLE_MODELS["skillearth"] = "SkillEarth"
"""

from __future__ import annotations

import hashlib
import ast
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .abstract_model import tp_model
from ..tool_inferencer.dynamic_batch_manager import DynamicBatchItem
from ..utils.rs_agent_prompt import RS_AGENT_PROMPT
from ..utils.log_utils import get_logger

logger = get_logger("skillearth_model")

TERMINATE_RULES = "\n".join(
    [
        "Terminate answer rules:",
        "- Use Terminate only when no more tool call is needed.",
        "- Terminate.arguments.ans must be non-empty.",
        "- Build ans only from the user's question and the latest relevant OBSERVATIONs in the conversation.",
        "- If an OBSERVATION already contains the final answer, copy its required facts directly instead of re-computing or paraphrasing heavily.",
        "- Preserve required numbers, percentages, distances, counts, entity names, and artifact markers exactly as observed whenever possible.",
        "- Do not add buffer radius, GSD, bbox coordinates, layer names, travel time, or intermediate values unless the user explicitly asks for them or they are part of the final observed answer.",
        "- If the expected final deliverable is an image, map, preview, plotted figure, or GeoTIFF visualization, set ans to exactly <image>.",
        "- If the question asks for assignments, closest pairs, missing objects, or statistics, include all required items from the relevant OBSERVATION; do not omit names or values.",
        "- Do not invent entities, relationships, statistics, or conclusions that are not supported by OBSERVATION.",
    ]
)

# ------------------------------------------------------------------------- #
# The evaluator is run from the SkillEarth repository root by eval_e2e.sh.
# Still add that root explicitly so imports remain correct if the CWD changes.
# ------------------------------------------------------------------------- #
def _ensure_skillearth_on_path() -> None:
    candidates = [
        os.environ.get("SKILLEARTH_REPO", ""),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")),
    ]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "skillearth")) and c not in sys.path:
            sys.path.insert(0, c)
            return


_ensure_skillearth_on_path()


@dataclass(frozen=True)
class _SkillRecord:
    folder: str
    doc: Any


class SkillEarth(tp_model):
    """Skill-augmented agent model."""

    def __init__(
        self,
        pretrained: str = "gpt-4o-mini",
        skillbank_dir: str = "",
        planner: str = "",  # only "llm" is valid for evaluation
        top_k: int = 1,
        min_score: float = 0.0,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        max_model_len: Optional[int] = None,
        retries: int = 3,
        embeddings_provider: str = "hashing",
        use_skill: Optional[bool] = None,
    ):
        self.model_name = pretrained
        self.temperature = float(temperature)
        self.retries = int(retries)
        self.system_prompt = RS_AGENT_PROMPT
        if use_skill is None:
            use_skill = os.environ.get("SKILLEARTH_USE_SKILL", "1").lower() in {
                "1", "true", "yes", "on"
            }
        self.use_skill = bool(use_skill)
        self.max_new_tokens = None
        self.default_max_new_tokens = int(
            os.environ.get("SKILLEARTH_GENERIC_MAX_TOKENS", "1024")
        )
        self.max_model_len = int(max_model_len) if max_model_len else 8192
        self.top_k = int(top_k)
        self.min_score = float(min_score)

        # --- resolve skillbank -------------------------------------------- #
        skillbank_dir = skillbank_dir or os.environ.get("OEA_SKILLBANK_DIR", "")
        if not skillbank_dir:
            guess = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../../../skillbank")
            )
            skillbank_dir = guess
        self.skillbank_dir = skillbank_dir

        self._bank = None
        self._skills: List[_SkillRecord] = []
        if self.use_skill:
            self._load_skillbank(embeddings_provider)
            self._forced_skill_by_question = self._load_forced_skill_map()
        else:
            self._forced_skill_by_question = {}

        # --- resolve planner backend -------------------------------------- #
        self._llm = None
        if planner == "":
            planner = "llm"
        self.planner = planner
        if self.planner != "llm":
            raise ValueError(
                f"SkillEarth evaluation requires planner=llm, got {self.planner!r}"
            )
        if self.planner == "llm":
            self._init_llm(api_key, base_url)
            if self._llm is None:
                raise RuntimeError(
                    "SkillEarth evaluator requested planner=llm, but the local "
                    "OpenAI-compatible LLM client could not be initialized"
                )

        try:
            logger.info(
                f"[SkillEarth] planner={self.planner} model={self.model_name} "
                f"use_skill={self.use_skill} skillbank={self.skillbank_dir} "
                f"skills={len(self._skills)} "
                f"top_k={self.top_k}"
            )
        except RuntimeError:
            pass

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #
    def _load_skillbank(self, embeddings_provider: str) -> None:
        try:
            from skillearth.oea.skill.skillbank import SkillBank

            self._bank = SkillBank(self.skillbank_dir)
            self._skills = [
                _SkillRecord(folder=folder.name, doc=doc)
                for folder, doc in self._bank.skills()
            ]
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"failed to load SkillEarth skillbank from {self.skillbank_dir}: {e}"
            ) from e
        if not self._skills:
            raise RuntimeError(
                f"SkillEarth skillbank contains no SKILL.md files: {self.skillbank_dir}"
            )

    @staticmethod
    def _normalize_question(text: str) -> str:
        return " ".join(str(text or "").strip().lower().split())

    def _load_forced_skill_map(self) -> Dict[str, str]:
        """Load an opt-in exact question-to-skill mapping for diagnostic runs."""
        path = os.environ.get("SKILLEARTH_FORCE_SKILL_MANIFEST", "").strip()
        if not path:
            return {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            mapping = payload.get("question_to_folder", {})
            if not isinstance(mapping, dict):
                return {}
            result = {
                self._normalize_question(question): str(folder)
                for question, folder in mapping.items()
                if self._normalize_question(question) and str(folder).strip()
            }
            return result
        except Exception:  # noqa: BLE001
            return {}

    @staticmethod
    def _llm_available(api_key: Optional[str]) -> bool:
        return bool(
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("SKILLEARTH_GENERIC_LLM_URL")
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("ZHIPUAI_API_KEY")
            or os.environ.get("BIGMODEL_API_KEY")
        )

    def _init_llm(self, api_key: Optional[str], base_url: Optional[str]) -> None:
        # SkillEarth uses the standard OpenAI-compatible API directly.  This
        # handles the local vLLM endpoint without importing the old skillearth
        # package or requiring OPENAI_API_KEY to be set.
        try:
            from openai import OpenAI

            resolved_url = (
                base_url
                or os.environ.get("SKILLEARTH_GENERIC_LLM_URL")
                or os.environ.get("OPENAI_BASE_URL")
            )
            key = (
                api_key
                or os.environ.get("SKILLEARTH_GENERIC_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or "EMPTY"
            )
            timeout = float(os.environ.get("SKILLEARTH_TIMEOUT_S", "600"))
            kwargs = {"api_key": key, "timeout": timeout}
            # GPT runs use the explicit retry loop in _respond_llm below.
            # Disable the SDK's rapid nested retries so a gateway 429 can
            # back off before the next request rather than amplifying load.
            if self.model_name == "gpt-5.6-terra":
                kwargs["max_retries"] = 0
            if resolved_url:
                kwargs["base_url"] = resolved_url
            self._openai = OpenAI(**kwargs)
            self._llm = self._openai
            self._llm_kind = "openai"
        except Exception as e:  # noqa: BLE001
            logger.error(f"[SkillEarth] OpenAI-compatible client initialization failed: {e}")
            self._llm = None

    def _build_llm_cfg(self, api_key: Optional[str], base_url: Optional[str]) -> Dict[str, Any]:
        if os.environ.get("SKILLEARTH_GENERIC_LLM_URL"):
            url = os.environ["SKILLEARTH_GENERIC_LLM_URL"]
            cfg = {
                "provider": "generic",
                "model": self.model_name,
                "api_key": os.environ.get("SKILLEARTH_GENERIC_API_KEY", ""),
                "url": url,
                "base_url": url,
            }
            timeout_s = os.environ.get("SKILLEARTH_TIMEOUT_S", "")
            if timeout_s:
                cfg["timeout_s"] = int(timeout_s)
            max_tokens = os.environ.get("SKILLEARTH_GENERIC_MAX_TOKENS", "")
            if max_tokens:
                cfg["max_tokens"] = int(max_tokens)
            max_input_chars = os.environ.get("SKILLEARTH_GENERIC_MAX_INPUT_CHARS", "")
            if max_input_chars:
                cfg["max_input_chars"] = int(max_input_chars)
            return cfg
        if os.environ.get("DASHSCOPE_API_KEY"):
            return {
                "provider": "dashscope",
                "model": self.model_name,
                "api_key": os.environ["DASHSCOPE_API_KEY"],
                "base_url": os.environ.get(
                    "DASHSCOPE_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode",
                ),
            }
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if key:
            return {
                "provider": "openai",
                "model": self.model_name,
                "api_key": key,
                "base_url": base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com"),
            }
        return {"provider": "mock"}

    # ------------------------------------------------------------------ #
    # Conversation helpers (mirror OpenAIModels)
    # ------------------------------------------------------------------ #
    def generate_conversation_fn(self, text, role="user"):
        return [{"role": role, "content": text}]

    def append_conversation_fn(self, conversation, text, role):
        conversation.append({"role": role, "content": text})
        return conversation

    def getitem_fn(self, meta_data, idx):
        return meta_data[idx]

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    # ------------------------------------------------------------------ #
    # Skill retrieval + context integration
    # ------------------------------------------------------------------ #
    def _question_of(self, conversation: List[Dict[str, Any]]) -> str:
        for turn in conversation:
            if turn.get("role") == "user":
                val = str(turn.get("content", ""))
                val = val.replace("<AGENT_PROMPT>", "")
                if "Question:" in val:
                    val = val.split("Question:", 1)[1]
                return val.strip()
        return ""

    def _retrieve(self, question: str):
        if not self._skills:
            raise RuntimeError("SkillEarth skillbank is empty during retrieval")
        forced_folder = self._forced_skill_by_question.get(self._normalize_question(question))
        if forced_folder:
            for rec in self._skills:
                if rec.folder == forced_folder:
                    return rec
            logger.warning("[SkillEarth] forced skill folder is not loaded: %s", forced_folder)

        query_vector = self._hash_features(question)
        scored = [
            (rec, self._cosine(query_vector, self._hash_features(self._search_text(rec.doc))))
            for rec in self._skills
        ]
        rec, score = max(scored, key=lambda item: (item[1], item[0].folder))
        if self.min_score and score < self.min_score:
            return None
        try:
            logger.info("[SkillEarth] retrieved skill=%s score=%.4f", rec.folder, score)
        except RuntimeError:
            pass
        return rec

    @staticmethod
    def _section_content(value: Any) -> Any:
        return getattr(value, "content", value)

    @classmethod
    def _search_text(cls, doc: Any) -> str:
        # SkillDoc stores only its stable front-matter fields and sections.
        # Older skillbanks may carry a ``type`` attribute, but the current
        # dataclass intentionally does not, so retrieval must not require it.
        parts = [
            str(getattr(doc, "name", "")),
            str(getattr(doc, "description", "")),
            str(getattr(doc, "type", "") or ""),
        ]
        parts.append(str(cls._section_content(doc.when_to_use) or ""))
        for field in (doc.plan, doc.grounded_observation_routing):
            content = cls._section_content(field) or []
            if isinstance(content, str):
                parts.append(content)
            else:
                parts.extend(str(item) for item in content)
        for row in doc.tool_rows:
            parts.extend([str(row.name), str(row.parameters), str(row.description)])
        return "\n".join(parts)

    @staticmethod
    def _tool_sequence(doc: Any) -> List[str]:
        """Read the persisted skill structure from SkillDoc metadata."""
        metadata = getattr(doc, "metadata", {}) or {}
        sequence = metadata.get("tool_sequence", []) if isinstance(metadata, dict) else []
        return [str(tool) for tool in sequence if str(tool).strip()]

    @staticmethod
    def _features(text: str) -> List[str]:
        normalized = " ".join(str(text or "").lower().split())
        words = re.findall(r"[a-z0-9_]+", normalized)
        compact = re.sub(r"\s+", "", normalized)
        chars = [compact[i : i + 2] for i in range(max(0, len(compact) - 1))]
        return words + chars

    @classmethod
    def _hash_features(cls, text: str, dims: int = 2048) -> Dict[int, float]:
        vector: Dict[int, float] = {}
        for feature in cls._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            slot = int.from_bytes(digest, "big") % dims
            vector[slot] = vector.get(slot, 0.0) + 1.0
        return vector

    @staticmethod
    def _cosine(left: Dict[int, float], right: Dict[int, float]) -> float:
        if not left or not right:
            return 0.0
        dot = sum(value * right.get(key, 0.0) for key, value in left.items())
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def _skill_context(self, rec) -> str:
        """Read the retrieved SKILL.md into an injectable context block."""
        if rec is None:
            return ""
        doc = rec.doc
        skill_md = ""
        folder = str(getattr(rec, "folder", "") or "")
        if folder:
            md_path = os.path.join(self.skillbank_dir, folder, "SKILL.md")
            try:
                with open(md_path, "r", encoding="utf-8") as fh:
                    skill_md = fh.read().strip()
            except Exception:
                skill_md = ""
        if skill_md:
            return "\n".join(
                [
                    "## Retrieved SkillEarth SKILL.md",
                    "Follow this skill if it matches the question. Use its exact tool names, exact argument names, plan, tool sequence, and observation routing.",
                    "",
                    skill_md,
                    "",
                    "Structural tool sequence: " + " -> ".join(self._tool_sequence(doc)),
                    "",
                    "Runtime rule: output exactly ONE valid JSON object for the current step only. Do not write user:, assistant:, OBSERVATION:, markdown, or a second JSON object. Use exactly ONE action per step, feed each observation into the next step, and finish with a Terminate action carrying the answer.",
                    TERMINATE_RULES,
                ]
            )

        # Fallback for in-memory records without an on-disk SKILL.md.
        lines = ["## Retrieved SkillEarth (follow this if it matches the question)"]
        lines.append(f"Skill: {doc.name}")
        when_to_use = self._section_content(doc.when_to_use)
        if when_to_use:
            lines.append(f"When to use: {when_to_use}")
        lines.append("Plan:")
        for i, step in enumerate(self._section_content(doc.plan) or [], 1):
            lines.append(f"  {i}. {step}")
        lines.append("Recommended tool sequence: " + " -> ".join(self._tool_sequence(doc)))
        routing = self._section_content(doc.grounded_observation_routing) or []
        if routing:
            lines.append("Grounded observation routing:")
            for item in routing:
                lines.append(f"  - {item}")
        lines.append("Tool list:")
        for row in doc.tool_rows:
            lines.append(f"  - {row.name}({row.parameters})")
        lines.append(
            "Output exactly ONE valid JSON object for the current step only. Do not "
            "write user:, assistant:, OBSERVATION:, markdown, or a second JSON object. "
            "Use exactly ONE tool per step, in this order, feeding each observation "
            "into the next step, and finish with a Terminate action carrying the answer."
        )
        lines.append(TERMINATE_RULES)
        return "\n".join(lines)

    def _prepend_system(self, conversation, skill_ctx: str):
        sys_content = self.system_prompt
        if skill_ctx:
            sys_content = sys_content + "\n\n" + skill_ctx
        if not conversation or conversation[0].get("role") != "system":
            return [{"role": "system", "content": sys_content}] + conversation
        conversation[0]["content"] = sys_content
        return conversation

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def set_generation_config(self, generation_configs: dict = None) -> None:
        if generation_configs is not None:
            self.generation_config = generation_configs
            self.max_new_tokens = self.generation_config.get("max_length", 256)
        else:
            self.generation_config = {}

    def generate(self, batch):
        if not batch:
            return None
        self.max_new_tokens = self.max_new_tokens or self.default_max_new_tokens
        is_dynamic = all(isinstance(x, DynamicBatchItem) for x in batch)

        outputs: List[str] = []
        items = batch if is_dynamic else [None]
        # step mode: batch is a list of messages (batch["input"])
        if not is_dynamic:
            conv = batch
            rec = None
            if self.use_skill:
                question = self._question_of(conv)
                rec = self._retrieve(question)
            text = self._respond(conv, rec, step_index=self._round_of(conv))
            return [text]

        for item in items:
            conv = item.conversation
            # Skill retrieval is trajectory-level.  Select the skill once from
            # the original question and persist only its serializable context
            # on DynamicBatchItem; every later tool/observation round reuses it.
            if self.use_skill and not item.skill_retrieval_done:
                question = self._question_of(conv)
                rec = self._retrieve(question)
                item.retrieval_query = question
                item.skills_context = self._skill_context(rec)
                item.retrieved_skill_names = [rec.folder] if rec is not None else []
                item.skill_retrieval_done = True
            elif not self.use_skill:
                item.skills_context = ""
                item.retrieved_skill_names = []
                item.skill_retrieval_done = True
            round_idx = item.current_round  # 0 on first call
            text = self._respond(
                conv,
                None,
                step_index=round_idx,
                skill_ctx=item.skills_context,
            )
            item.model_response.append(text)
            self.append_conversation_fn(item.conversation, text, "assistant")
            outputs.append(text)
        return [outputs[-1]] if outputs else None

    def _round_of(self, conv) -> int:
        # number of assistant turns already present == step index
        return sum(1 for t in conv if t.get("role") == "assistant")

    def _respond(
        self,
        conversation,
        rec,
        step_index: int,
        skill_ctx: Optional[str] = None,
    ) -> str:
        if self._llm is None:
            raise RuntimeError("SkillEarth LLM client is unavailable during generation")
        return self._respond_llm(conversation, rec, skill_ctx=skill_ctx)

    # ---- llm planner -------------------------------------------------- #
    def _respond_llm(
        self,
        conversation,
        rec,
        *,
        skill_ctx: Optional[str] = None,
    ) -> str:
        if skill_ctx is None:
            skill_ctx = self._skill_context(rec)
        conv = self._prepend_system(list(conversation), skill_ctx)
        conv = self._truncate(conv)
        for attempt in range(self.retries):
            try:
                request_kwargs = dict(
                    model=self.model_name,
                    messages=conv,
                )
                # GPT-5.6 rejects the legacy Chat Completions `max_tokens`
                # field and supports only its default temperature. Ark's
                # OpenAI-compatible models keep the existing parameters.
                if self.model_name == "gpt-5.6-terra":
                    request_kwargs["max_completion_tokens"] = self.max_new_tokens
                else:
                    request_kwargs["max_tokens"] = self.max_new_tokens
                    request_kwargs["temperature"] = self.temperature
                resp = self._openai.chat.completions.create(**request_kwargs)
                return self._clean_model_output(resp.choices[0].message.content)
            except Exception as e:  # noqa: BLE001
                logger.error(f"[SkillEarth] LLM error {e} (attempt {attempt + 1})")
                error_text = str(e).lower()
                if self.model_name == "gpt-5.6-terra" and (
                    "429" in error_text or "rate limit" in error_text
                ):
                    base_sleep_s = float(
                        os.environ.get("SKILLEARTH_GPT_RETRY_SLEEP_S", "15")
                    )
                    sleep_s = min(base_sleep_s * (2**attempt), 120.0)
                    logger.warning(
                        "[SkillEarth] GPT rate limited; waiting %.1fs before retry %s/%s",
                        sleep_s,
                        attempt + 2,
                        self.retries,
                    )
                    time.sleep(sleep_s)
        raise RuntimeError(
            f"SkillEarth planner failed after {self.retries} LLM attempts; aborting evaluation"
        )

    def _clean_model_output(self, raw: Any) -> str:
        """Return exactly one valid OEA action JSON object when possible."""
        text = str(raw or "").strip()
        obj = self._extract_first_json_object(text)
        if obj is None:
            repaired = self._repair_action_json(text)
            if repaired is not None:
                normalized = self._normalize_action_object(repaired)
                if normalized is not None:
                    return json.dumps(normalized, ensure_ascii=False)
                return json.dumps(repaired, ensure_ascii=False)
            return text
        normalized = self._normalize_action_object(obj)
        if normalized is not None:
            return json.dumps(normalized, ensure_ascii=False)
        return json.dumps(obj, ensure_ascii=False)

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[Any]:
        decoder = json.JSONDecoder()
        for i, ch in enumerate(str(text or "")):
            if ch != "{":
                continue
            try:
                obj, _end = decoder.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
        return None

    @staticmethod
    def _normalize_action_object(obj: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(obj, dict):
            return None
        thought = obj.get("thought", "")
        actions = obj.get("actions", [])
        if not isinstance(actions, list):
            actions = []
        if not actions:
            return {"thought": str(thought or ""), "actions": []}
        first = actions[0]
        if not isinstance(first, dict):
            return {"thought": str(thought or ""), "actions": []}
        name = first.get("name", "")
        args = first.get("arguments", {})
        if not isinstance(args, dict):
            args = {}
        if str(name or "") == "Terminate" and not str(args.get("ans", "")).strip():
            fallback_ans = str(thought or "").strip()
            if fallback_ans:
                args = dict(args)
                args["ans"] = fallback_ans
        return {
            "thought": str(thought or ""),
            "actions": [{"name": str(name or ""), "arguments": args}],
        }

    def _repair_action_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Best-effort repair for common local-LLM JSON noise."""
        s = str(text or "")
        thought = ""
        m = re.search(r'"thought"\s*:\s*"((?:\\.|[^"\\])*)"', s, re.DOTALL)
        if m:
            try:
                thought = json.loads('"' + m.group(1) + '"')
            except Exception:
                thought = m.group(1)
        name = ""
        m = re.search(r'"name"\s*:\s*"([^"]+)"', s)
        if m:
            name = m.group(1)
        args: Dict[str, Any] = {}
        m = re.search(r'"arguments"\s*:\s*(\{.*?\})\s*[}\]]', s, re.DOTALL)
        if m:
            try:
                args = json.loads(m.group(1))
            except Exception:
                args = {}
        if name:
            return {"thought": str(thought or ""), "actions": [{"name": name, "arguments": args}]}
        if '"actions"' in s:
            return {"thought": str(thought or ""), "actions": []}
        return None

    @staticmethod
    def _flatten_user(conv) -> str:
        parts = []
        for t in conv:
            if t.get("role") == "system":
                continue
            parts.append(f"{t.get('role')}: {t.get('content')}")
        return "\n".join(parts)

    def _truncate(self, conv):
        # keep system + first user + most recent turns under a rough budget
        budget = self.max_model_len - (self.max_new_tokens or 256) - 200
        approx = sum(len(str(t.get("content", ""))) for t in conv) // 4
        while approx > budget and len(conv) > 4:
            del conv[2:4]
            approx = sum(len(str(t.get("content", ""))) for t in conv) // 4
        return conv


# The tf_eval model registry expects this exported class name for
# --model skillearth.
SkillEarthLM = SkillEarth
