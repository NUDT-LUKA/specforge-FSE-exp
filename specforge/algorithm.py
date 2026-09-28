"""SpecForge's evaluated Dafny specification-repair algorithm.

The implementation follows Algorithm 1 in the paper: establish V1, evaluate
frozen positive and negative witnesses (V2), append postconditions guided by
an accepted negative witness, re-verify, and enforce the preservation guard
(V3). Proof recovery is injected through an adapter so an upstream prover can
be used without vendoring its implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Protocol

from .guard import guarded_success
from .tfvr import Faithful, _verified_success, assess_faithful


class CompletionProvider(Protocol):
    def complete(self, system: str, prompt: str) -> str: ...


class ProofRecoverer(Protocol):
    def recover(self, source: str) -> str | None: ...


@dataclass
class RepairResult:
    source: str
    verified: bool
    faithful: Faithful
    status: str
    repair_attempts: int = 0
    retained_edits: int = 0
    events: list[str] = field(default_factory=list)


_REPAIR_SYSTEM = (
    "You strengthen an under-constrained Dafny postcondition. Output only "
    "additional Dafny `ensures` clauses, one per line, each beginning with "
    "`ensures`. Do not output executable code, markdown fences, or prose."
)
_ENSURES_LINE = re.compile(r"^ensures\s+\S")


def parse_ensures(text: str) -> list[str]:
    clauses: list[str] = []
    for line in (text or "").splitlines():
        value = line.strip().strip("`").strip()
        if _ENSURES_LINE.match(value):
            clauses.append(value)
    return clauses


def append_ensures(source: str, clauses: list[str]) -> str | None:
    lines = source.split("\n")
    indices = [i for i, line in enumerate(lines) if re.match(r"\s*ensures\s", line)]
    if not indices:
        return None
    insert_at = indices[-1] + 1
    indent = re.match(r"(\s*)", lines[indices[-1]]).group(1)
    additions = [indent + clause for clause in clauses]
    return "\n".join(lines[:insert_at] + additions + lines[insert_at:])


def _repairable(assessment: Faithful) -> bool:
    return (
        assessment.faithful is False
        and assessment.accepts_positives is True
        and assessment.rejects_negatives is False
        and bool(assessment.witness)
    )


class SpecForge:
    """Witness-guided, append-only Dafny contract repair."""

    def __init__(
        self,
        provider: CompletionProvider,
        verifier,
        *,
        repair_rounds: int = 3,
        proof_recoverer: ProofRecoverer | None = None,
    ):
        self.provider = provider
        self.verifier = verifier
        self.repair_rounds = repair_rounds
        self.proof_recoverer = proof_recoverer

    def run(self, source: str, positives: list[dict], negatives: list[dict]) -> RepairResult:
        original = source
        current = source
        events: list[str] = []

        if not _verified_success(self.verifier.verify(current, None)):
            if self.proof_recoverer is None:
                assessment = assess_faithful(
                    current, positives, negatives, self.verifier, source_verified=False
                )
                return RepairResult(current, False, assessment, "unverifiable", events=events)
            recovered = self.proof_recoverer.recover(current)
            if not recovered or not _verified_success(self.verifier.verify(recovered, None)):
                assessment = assess_faithful(
                    current, positives, negatives, self.verifier, source_verified=False
                )
                events.append("proof_recovery_failed")
                return RepairResult(current, False, assessment, "unverifiable", events=events)
            guard_ok, _ = guarded_success(original, recovered, True, allow_code_edit=False)
            if not guard_ok:
                assessment = assess_faithful(
                    current, positives, negatives, self.verifier, source_verified=False
                )
                events.append("proof_recovery_guard_rejected")
                return RepairResult(current, False, assessment, "guard_rejected", events=events)
            current = recovered
            events.append("proof_recovered")

        assessment = assess_faithful(
            current, positives, negatives, self.verifier, source_verified=True
        )
        retained = 0
        attempts = 0
        feedback = ""

        for _ in range(self.repair_rounds):
            if not _repairable(assessment):
                break
            attempts += 1
            witness = assessment.witness
            expected = self._expected(positives, witness.get("inputs"))
            examples = "; ".join(
                f"{item.get('inputs')}->{item.get('out_lits')}" for item in positives[:5]
            )
            prompt = (
                "The Dafny method below verifies, but its postcondition accepts an "
                f"unintended return. For input {witness.get('inputs')}, it accepts "
                f"{witness.get('wrong')} while the intended return is {expected}. "
                f"Intended examples: {examples}. Add one or more postconditions that "
                "remain true of the body and reject the witnessed wrong return. "
                f"{feedback}\n\n```dafny\n{current}\n```"
            )
            try:
                clauses = parse_ensures(self.provider.complete(_REPAIR_SYSTEM, prompt))
            except Exception as exc:
                events.append(f"provider_error:{type(exc).__name__}")
                break
            if not clauses:
                feedback = "The previous response contained no parseable ensures clause."
                events.append("no_ensures_parsed")
                continue
            candidate = append_ensures(current, clauses)
            if candidate is None:
                events.append("splice_failed")
                break
            if not _verified_success(self.verifier.verify(candidate, None)):
                feedback = "The proposed clauses did not verify; revise them."
                events.append("candidate_unverifiable")
                continue
            guard_ok, _ = guarded_success(original, candidate, True, allow_code_edit=False)
            if not guard_ok:
                feedback = "The proposed edit failed the preservation guard."
                events.append("guard_rejected")
                continue
            current = candidate
            retained += 1
            feedback = ""
            events.append("edit_retained")
            assessment = assess_faithful(
                current, positives, negatives, self.verifier, source_verified=True
            )

        final_verification = _verified_success(self.verifier.verify(current, None))
        final_guard, _ = guarded_success(original, current, final_verification, allow_code_edit=False)
        final_assessment = assess_faithful(
            current,
            positives,
            negatives,
            self.verifier,
            source_verified=final_verification,
        )
        success = final_guard and final_assessment.faithful is True
        status = "success" if success else (
            "unfaithful" if final_assessment.faithful is False else "inconclusive"
        )
        return RepairResult(
            current,
            final_verification,
            final_assessment,
            status,
            repair_attempts=attempts,
            retained_edits=retained,
            events=events,
        )

    @staticmethod
    def _expected(positives: list[dict], inputs):
        for item in positives:
            if item.get("inputs") == inputs:
                return item.get("out_lits")
        return positives[0].get("out_lits") if positives else "?"

