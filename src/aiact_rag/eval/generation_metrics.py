"""LLM-judged answer metrics.

* **Faithfulness** (RAGAS definition): decompose the answer into atomic claims and check each against
  the retrieved excerpts only. Score = supported / total claims. Measures hallucination w.r.t. context.
* **Answer relevance** (RAGAS definition): the judge writes N questions that the answer would answer;
  score = mean cosine similarity between those and the original question (same bi-encoder as retrieval),
  forced to 0 when the answer is non-committal. Measures whether the answer addresses the question.
* **Answer correctness**: the judge splits the gold reference answer into key points and marks each as
  covered / partial / missing / contradicted in the system answer. Score = (covered + 0.5·partial) / n.
  Uses the human-written references, so it also catches faithful-but-incomplete answers.
* **Abstention**: unanswerable items should yield ``answerable=false``; answerable ones should not.
* **Citation precision**: share of cited excerpts that contain a gold provision (no LLM needed).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings
from ..embeddings import DenseEncoder, cosine
from ..generation import Answer, format_context
from ..llm import LLM

FAITHFULNESS_SYSTEM = """You are a meticulous fact-checker evaluating a retrieval-augmented answer about \
the EU AI Act. You judge ONLY whether each statement is supported by the provided excerpts — not whether \
it is true in general. Background knowledge must not be used to mark a claim as supported."""

FAITHFULNESS_PROMPT = """<excerpts>
{context}
</excerpts>

<question>{question}</question>

<answer>
{answer}
</answer>

1. Decompose the answer into atomic, self-contained factual claims (one fact each; resolve pronouns). \
Ignore citation markers like [2], hedges, and sentences that only say information is missing.
2. For each claim decide:
   - "supported": the excerpts state it or it follows directly from them;
   - "unsupported": the excerpts do not contain it, or contradict it.
Quote the shortest excerpt span that supports each supported claim."""

FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["supported", "unsupported"]},
                    "evidence": {"type": "string"},
                },
                "required": ["claim", "verdict", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

RELEVANCE_SYSTEM = "You help evaluate whether answers address the question they were given."

RELEVANCE_PROMPT = """<answer>
{answer}
</answer>

Write {n} different questions that this answer would be a direct, complete response to. Also decide whether \
the answer is non-committal (evasive, vague, or only says it cannot answer)."""

RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {"type": "array", "items": {"type": "string"}},
        "noncommittal": {"type": "boolean"},
    },
    "required": ["questions", "noncommittal"],
    "additionalProperties": False,
}

CORRECTNESS_SYSTEM = """You grade answers about the EU AI Act against a reference answer written by an \
expert. Judge substance, not wording; extra correct detail is not penalised."""

CORRECTNESS_PROMPT = """<question>{question}</question>

<reference_answer>
{reference}
</reference_answer>

<system_answer>
{answer}
</system_answer>

Split the reference answer into its key points (the facts a correct answer must convey; at most 6). For \
each, state whether the system answer "covered", "partial" (mentioned but incomplete or imprecise), \
"missing", or "contradicted" (the system answer states something incompatible)."""

CORRECTNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "point": {"type": "string"},
                    "status": {"type": "string", "enum": ["covered", "partial", "missing", "contradicted"]},
                },
                "required": ["point", "status"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["points"],
    "additionalProperties": False,
}


@dataclass
class GenerationScores:
    faithfulness: float | None
    answer_relevance: float | None
    correctness: float | None
    abstained: bool
    abstention_correct: bool
    citation_precision: float | None
    n_claims: int
    unsupported_claims: list[str]
    contradicted_points: list[str]


class Judge:
    def __init__(self, llm: LLM, settings: Settings, encoder: DenseEncoder, n_questions: int = 3):
        self.llm = llm
        self.settings = settings
        self.encoder = encoder
        self.n_questions = n_questions

    def _call(self, system: str, prompt: str, schema: dict) -> dict:
        return self.llm.complete(
            system=system,
            user=prompt,
            model=self.settings.judge_model,
            effort=self.settings.judge_effort,
            max_tokens=16000,
            schema=schema,
        ).json()

    def faithfulness(self, ans: Answer) -> tuple[float | None, list[dict]]:
        data = self._call(
            FAITHFULNESS_SYSTEM,
            FAITHFULNESS_PROMPT.format(
                context=format_context(ans.contexts), question=ans.question, answer=ans.answer
            ),
            FAITHFULNESS_SCHEMA,
        )
        claims = data["claims"]
        if not claims:
            return None, claims  # nothing asserted (e.g. pure abstention): undefined, excluded from the mean
        return sum(c["verdict"] == "supported" for c in claims) / len(claims), claims

    def answer_relevance(self, ans: Answer) -> float:
        data = self._call(
            RELEVANCE_SYSTEM, RELEVANCE_PROMPT.format(answer=ans.answer, n=self.n_questions), RELEVANCE_SCHEMA
        )
        if data["noncommittal"] or not data["questions"]:
            return 0.0
        q = self.encoder.embed_queries([ans.question])[0]
        gen = self.encoder.embed_queries(data["questions"][: self.n_questions])
        return sum(cosine(q, g) for g in gen) / len(gen)

    def correctness(self, ans: Answer, reference: str) -> tuple[float, list[dict]]:
        data = self._call(
            CORRECTNESS_SYSTEM,
            CORRECTNESS_PROMPT.format(question=ans.question, reference=reference, answer=ans.answer),
            CORRECTNESS_SCHEMA,
        )
        points = data["points"]
        if not points:
            return 0.0, points
        weight = {"covered": 1.0, "partial": 0.5, "missing": 0.0, "contradicted": 0.0}
        return sum(weight[p["status"]] for p in points) / len(points), points

    def score(self, ans: Answer, gold_units: list[str], reference: str, answerable: bool) -> GenerationScores:
        abstained = not ans.answerable
        cited = [c for c in ans.citations]
        citation_precision = (
            sum(1 for c in cited if set(c.unit_ids) & set(gold_units)) / len(cited)
            if cited and gold_units
            else None
        )
        if not answerable:
            return GenerationScores(
                faithfulness=None,
                answer_relevance=None,
                correctness=None,
                abstained=abstained,
                abstention_correct=abstained,
                citation_precision=None,
                n_claims=0,
                unsupported_claims=[],
                contradicted_points=[],
            )
        faith, claims = self.faithfulness(ans)
        relevance = self.answer_relevance(ans)
        correct, points = self.correctness(ans, reference)
        return GenerationScores(
            faithfulness=faith,
            answer_relevance=relevance,
            correctness=correct,
            abstained=abstained,
            abstention_correct=not abstained,
            citation_precision=citation_precision,
            n_claims=len(claims),
            unsupported_claims=[c["claim"] for c in claims if c["verdict"] != "supported"],
            contradicted_points=[p["point"] for p in points if p["status"] == "contradicted"],
        )
