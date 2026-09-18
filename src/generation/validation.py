"""Deterministic post-generation grounding checks. The model's own `evidence_status` is a claim,
not a fact: every cited `iso_reference` is checked against the references actually present in the
supplied evidence. Anything not found is dropped, `requires_human_review` is forced on, and
`evidence_status` is downgraded accordingly. `supporting_sources` and `confidence` are computed
here, from the evidence and the reranker's own relevance scores -- never taken from the model.
"""

import math
import re
from collections import defaultdict
from typing import Dict, List, Sequence

from src.generation.models import AuditFinding, EvidenceStatus, FindingType, GeneratedFinding, SupportingSource
from src.ingestion.references import extract_references
from src.vectorstore import SearchHit

# A canonical reference (references.py) is "<standard> <local part>", e.g. "ISO/IEC 27001 A.5.18"
# or "RGPD art. 33". Strips the standard so a bare "A.5.18" from the model can still be matched.
_STANDARD_PREFIX_RE = re.compile(r"^(?:ISO/IEC\s+\d+|ISO\s+\d+|RGPD)\s+", re.IGNORECASE)

# A pure Markdown table separator row ("|---|---|", "| - | - |") carries no content once dumped
# into a plain Word/PDF paragraph -- just stray pipes and dashes. Matched only when a "|" is
# present, so this never touches an ordinary sentence that happens to contain a dash.
_MARKDOWN_SEPARATOR_LINE_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
_MARKDOWN_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")

# A measured particular about the audited organisation: a number carrying an audit-style unit
# ("14 mois", "12 utilisateurs", "4 équipements"). Deliberately narrow -- a figure with no unit is
# not matched, which is what keeps standard/algorithm numbering (A.5.18, 27002, TLS 1.2, AES-256)
# out of this check without needing to special-case any of them.
# --- Contradiction observation / constat ------------------------------------------------------
# The corpus is, by design, a set of worked examples of *deficiencies*: every chunk retrievable for
# a given control describes something going wrong (the A.5.18 example finding, NC-2026-014, the
# matching improvement opportunity). Retrieval on a control therefore returns only deficiency text
# whatever the auditor actually reported, and the model completes the pattern -- so an observation
# stating that a measure *is* in place still yields a non-conformity. Live-observed on exactly that
# A.5.18 case, and the draft scored 0.758 because every other signal (real citation, strong
# evidence, high reranker score) was genuinely fine.
#
# Detection is deliberately one-directional: only "the auditor reported conformity, the draft
# reports a deficiency" is flagged. The opposite direction is the normal, correct workflow.
_ACTS = r"(?:effectu|realis|document|applique|formalis|trac|journalis|historis|revu|test|valide|approuve|mis)"

# Any hint of a shortcoming in the observation disarms the check entirely. Generous on purpose:
# a missed contradiction only costs a confidence penalty (the finding is flagged for review either
# way), while a false positive would contradict an auditor who reported a real problem.
_DEFICIENCY_CUE_RE = re.compile(
    r"\bn['e]\s*\w*\s*pas\b|\bne\s+\w+\s+pas\b|\bn'(?:est|a|ont|sont|existe|etait|avait)\b"
    r"|\baucune?\b|\bjamais\b|\bsans\b|\babsen(?:ce|t)|\bmanque|\binsuffisan"
    rf"|\bnon[\s-]{_ACTS}|\bpas\s+(?:encore\s+)?{_ACTS}"
    r"|\bnon[\s-](?:conform|defini|couvert)"
    r"|\bdefaut\b|\bdefaillan|\blacune|\bfaille|\becart\b|\bobsolet|\bdeprecie|\bexpire|\bperime"
    r"|\bincomplet|\bpartiel|\bechec|\bpas\s+de\b|\bdepourvu|\bomission|\boubli"
    r"|\bretard|\bhors\s+delai|\bvulnerab|\bignore"
    r"|\bnot\b|\bno\s+\w|\bnever\b|\bwithout\b|\black(?:s|ing)?\b|\bmissing\b|\bfail(?:s|ed|ure)?\b"
    r"|\binsufficient\b|\boutdated\b|\bexpired\b|\bincomplete\b|\babsence\b",
    re.IGNORECASE,
)

# Auxiliary + participle with nothing in between, so "n'a pas été réalisée" does not match while
# "est réalisée" does.
_ASSERTED_IN_PLACE = (
    r"\bconformement\s+a\b|\bconformes?\s+(?:a|aux)\b|\best\s+conforme|\bsont\s+conformes"
    rf"|\b(?:est|sont)\s+(?:bien\s+)?{_ACTS}"
    r"|\b(?:est|sont)\s+(?:en\s+place|a\s+jour|mis\s+en\s+oeuvre)"
    r"|\brespecte\b|\bsatisfait\s+a\b|\bcorrectement\b|\bdispose\s+d"
    r"|\b(?:is|are)\s+(?:performed|documented|implemented|reviewed|tested|carried\s+out|in\s+place|up\s+to\s+date)\b"
    r"|\bin\s+accordance\s+with\b|\bcompl(?:y|ies)\s+with\b"
)
# On the observation side a bare participle is accepted too, for the telegraphic register auditors
# often use ("Revue trimestrielle effectuée, preuves fournies"). Restricted to stems that cannot
# also be read as a noun: "revue", "document", "test", "trace" and "mise" are all ordinary nouns in
# an audit sentence, and accepting them bare made "La dernière revue des accès remonte à 14 mois"
# -- a plain statement of fact, with no claim that anything is in place -- read as conformity.
_UNAMBIGUOUS_PARTICIPLE = r"(?:effectu|realis|applique|formalis|journalis|historis|valide|approuve|fourni)"
_CONFORMITY_OBSERVATION_RE = re.compile(
    _ASSERTED_IN_PLACE + rf"|\b{_UNAMBIGUOUS_PARTICIPLE}(?:e|ee|es|ees|s)?\b", re.IGNORECASE,
)
_CONFORMITY_FINDING_RE = re.compile(_ASSERTED_IN_PLACE, re.IGNORECASE)

_ACCENT_FOLD = str.maketrans("áàâäéèêëíìîïóòôöúùûüç", "aaaaeeeeiiiioooouuuuc")

_QUANTIFIED_CLAIM_RE = re.compile(
    r"\b(\d{1,4})\s*(?:%|mois|ans?|ann[ée]es?|jours?|semaines?|heures?|minutes?|fois|"
    r"utilisateurs?|comptes?|[ée]quipements?|serveurs?|postes?|machines?|incidents?|"
    r"tickets?|sites?|employ[ée]s?|collaborateurs?|personnes?|applications?)\b",
    re.IGNORECASE,
)


def _strip_markdown_table_artifacts(text: str | None) -> str | None:
    """The model occasionally drafts a list as a Markdown table (observed live -- an access-rights
    breakdown written as `| Utilisateur | Droit |` rows) even though this text is meant for a
    formal report paragraph, not a chat UI that renders Markdown: the raw pipes and separator rows
    show up as literal garbage ("| |", "|-|") in the final document. A no-op whenever the text has
    no "|" at all -- the overwhelming majority of findings -- so ordinary prose is never touched.
    """
    if not text or "|" not in text:
        return text
    cleaned_lines: List[str] = []
    for line in text.split("\n"):
        if _MARKDOWN_SEPARATOR_LINE_RE.match(line):
            continue  # a separator row carries no content
        row_match = _MARKDOWN_ROW_RE.match(line)
        if row_match:
            cells = [c.strip() for c in row_match.group(1).split("|") if c.strip()]
            if cells:
                cleaned_lines.append(" ; ".join(cells))
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip() or None

# The model sometimes drafts a bare, generic type label as the whole `finding` text (observed live,
# not hypothetical -- e.g. finding="Opportunité d'amélioration" while finding_type=non_conformite).
# When that bare label names a *different* type than finding_type itself, that is a deterministic,
# checkable self-contradiction -- same "never trust the model's own consistency" posture as the
# evidence_status/iso_reference check below, just for a different pair of fields.
_GENERIC_LABEL_TYPES = {
    "constat": FindingType.CONSTAT,
    "non-conformité": FindingType.NON_CONFORMITE,
    "non conformité": FindingType.NON_CONFORMITE,
    "observation": FindingType.OBSERVATION,
    "opportunité d'amélioration": FindingType.OPPORTUNITE_AMELIORATION,
    "opportunite d'amelioration": FindingType.OPPORTUNITE_AMELIORATION,
}


def _label_claims_type(finding_text: str) -> FindingType | None:
    return _GENERIC_LABEL_TYPES.get(finding_text.strip().rstrip(".:").lower())


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _local_part(canonical_reference: str) -> str:
    return _STANDARD_PREFIX_RE.sub("", canonical_reference).strip().lower()


def reference_index(evidence: Sequence[SearchHit]) -> tuple[Dict[str, str], Dict[str, List[str]]]:
    """(exact lowercased form -> canonical, local part -> canonical forms sharing it)."""
    by_exact: Dict[str, str] = {}
    by_local: Dict[str, List[str]] = defaultdict(list)
    for hit in evidence:
        for ref in hit.payload.get("references", []):
            by_exact[ref.strip().lower()] = ref
            if ref not in by_local[_local_part(ref)]:
                by_local[_local_part(ref)].append(ref)
    return by_exact, by_local


def _resolve_reference(ref: str, by_exact: Dict[str, str], by_local: Dict[str, List[str]]) -> str | None:
    """Matches one model-cited reference against the evidence's canonical references, trying, in
    order: (1) an exact canonical match; (2) an unambiguous bare local id ("A.5.18"); (3) re-parsing
    the model's own citation -- which is free text, not guaranteed to already be in canonical form
    ("27002 §5.15" has no "ISO/IEC" word and a "§" the first two steps don't strip -- both observed
    live, dropping a real, well-evidenced citation) -- with the same reference extractor
    `src/ingestion/references.py` uses to canonicalise the knowledge base itself, then matching
    *that* form. This is not fuzzy matching: it is the exact same deterministic, rule-based
    canonicalisation the evidence's own references were indexed with, applied to the model's raw
    citation string instead of to a knowledge-base document.

    Stays conservative (returns None) wherever there is real ambiguity to resolve. Two distinct
    places that can happen: the raw citation, as literally written, might already collide with more
    than one evidenced standard (step 2); or -- easy to miss -- a *re-parsed* candidate might look
    unambiguous on its own (`extract_references` defaults a bare "A.x.y" to ISO/IEC 27001 by audit
    convention, a fine rule for reading a knowledge-base document) while the evidence set actually
    offers that same local id under a *different* standard too, in which case accepting the
    re-parsed candidate would silently pick a side of an ambiguity the model's own text never
    resolved. Every re-parsed candidate is therefore re-checked against `by_local`, not just `by_exact`.
    """
    def _match(candidate: str) -> str | None:
        key = candidate.strip().lower()
        exact = by_exact.get(key)
        if exact is not None:
            return exact
        local = by_local.get(key, [])
        return local[0] if len(local) == 1 else None

    direct = _match(ref)
    if direct is not None:
        return direct
    if by_local.get(ref.strip().lower()):  # ambiguous as literally written -- don't try to route around it
        return None

    resolved = set()
    for candidate in extract_references(ref):
        if candidate.strip().lower() not in by_exact:
            continue
        if len(by_local.get(_local_part(candidate), [])) > 1:
            continue  # this candidate's local id is itself ambiguous in the evidence -- don't guess
        resolved.add(by_exact[candidate.strip().lower()])
    return next(iter(resolved)) if len(resolved) == 1 else None


class GroundingValidator:
    # Below this reranker-derived confidence, a cited reference is real but too weakly backed to
    # trust without a human looking at it. See README for how this threshold was chosen.
    LOW_CONFIDENCE_REVIEW_THRESHOLD = 0.35

    def validate(
        self, observation: str, generated: GeneratedFinding, evidence: Sequence[SearchHit],
        document_evidence: Sequence[SearchHit] = (),
    ) -> AuditFinding:
        notes: List[str] = []
        by_exact, by_local = reference_index(evidence)

        kept_references: List[str] = []
        dropped_references: List[str] = []
        for ref in self._cited_references(generated):
            canonical = _resolve_reference(ref, by_exact, by_local)
            if canonical is None:
                dropped_references.append(ref)
            elif canonical not in kept_references:
                kept_references.append(canonical)

        evidence_status = generated.evidence_status
        requires_review = generated.requires_human_review

        claimed_type = _label_claims_type(generated.finding)
        if claimed_type is not None and claimed_type != generated.finding_type:
            notes.append(
                f"The drafted finding text is the bare label '{generated.finding}', which "
                f"contradicts the model's own finding_type ({generated.finding_type.value}); flagged for review."
            )
            requires_review = True

        if dropped_references:
            notes.append("Removed reference(s) not present in the retrieved evidence: "
                         + ", ".join(dict.fromkeys(dropped_references)))
            requires_review = True
            if evidence_status == EvidenceStatus.SUPPORTED:
                evidence_status = EvidenceStatus.PARTIAL if kept_references else EvidenceStatus.INSUFFICIENT

        if not evidence:
            evidence_status = EvidenceStatus.INSUFFICIENT
            requires_review = True
            kept_references = []
            notes.append("No evidence was retrieved for this observation.")
        elif not kept_references and evidence_status == EvidenceStatus.SUPPORTED:
            evidence_status = EvidenceStatus.INSUFFICIENT  # "supported" but nothing verifiable backs it
            requires_review = True
            notes.append("evidence_status was 'supported' with no verifiable ISO reference; downgraded.")
        elif kept_references and evidence_status == EvidenceStatus.INSUFFICIENT:
            # The model hedged ("insufficient") while still citing a validated reference -- a
            # self-contradictory claim. Which half of the contradiction to trust is decided the
            # same way everything else here is: by the reranker's own independent score for the
            # cited reference, never by the model's word alone (live-observed: llama3:8b sometimes
            # says "insufficient" right after citing a reference the evidence genuinely, strongly
            # backs -- discarding that citation outright would throw away real traceability the
            # auditor is entitled to see).
            requires_review = True
            candidate_sources = self._supporting_sources(evidence, kept_references)
            # Judged one reference at a time: when a draft cites several, the well-evidenced ones
            # must survive a loosely-evidenced neighbour instead of the whole citation list being
            # discarded together.
            well_backed = [
                reference for reference in kept_references
                if self._reference_confidence(reference, candidate_sources) >= self.LOW_CONFIDENCE_REVIEW_THRESHOLD
            ]
            dropped_for_weak_backing = [r for r in kept_references if r not in well_backed]
            if well_backed:
                # Keep the citation available to the auditor -- but a self-contradiction is never
                # resolved into an automatic "supported": PARTIAL is the honest state, since the
                # model itself was not confident enough to commit, whatever the evidence says.
                evidence_status = EvidenceStatus.PARTIAL
                candidate_confidence = self._confidence(well_backed, candidate_sources, requires_review=False)
                notes.append(
                    f"evidence_status was 'insufficient' despite citing {', '.join(well_backed)}, which is "
                    f"well-backed by the evidence (confidence {candidate_confidence}); kept for the auditor's "
                    "traceability but flagged for review rather than trusted, since the model contradicted itself."
                )
                if dropped_for_weak_backing:
                    notes.append(
                        "Also cited, but too loosely backed by the evidence to keep alongside it: "
                        + ", ".join(dropped_for_weak_backing) + "."
                    )
                kept_references = well_backed
            else:
                notes.append(
                    f"evidence_status was 'insufficient' despite citing {', '.join(kept_references)}; "
                    "reference(s) removed to keep the two consistent."
                )
                kept_references = []

        if evidence_status == EvidenceStatus.INSUFFICIENT and not requires_review:
            # A live-observed gap: the model can correctly say "insufficient evidence, no
            # citation" while still (incorrectly, or just carelessly) leaving its own
            # requires_human_review claim False -- nothing else above touches `requires_review`
            # when there was nothing to drop and evidence_status was already insufficient from the
            # start. "Insufficient evidence to confirm a finding" must always mean a human looks at
            # it; this is never something the deterministic layer should leave to the model's word.
            requires_review = True
            notes.append("evidence_status is 'insufficient'; forced for human review regardless of the model's own flag.")

        supporting_sources = self._supporting_sources(evidence, kept_references)

        # A reference can be a real, verbatim reference from the evidence and still be the wrong
        # one for this specific claim: string presence alone cannot catch that (it would need
        # another judge, which just relocates the trust problem). The reranker's own score is an
        # independent, observable proxy for "is this evidence actually about the claim" -- weak
        # backing forces review even when the model itself was confident.
        weakly_backed = [
            reference for reference in kept_references
            if self._reference_confidence(reference, supporting_sources) < self.LOW_CONFIDENCE_REVIEW_THRESHOLD
        ]
        if weakly_backed:
            requires_review = True
            notes.append(
                "Reranker confidence is low for the cited reference(s) " + ", ".join(weakly_backed)
                + "; flagged for review regardless of the model's own assessment."
            )
        confidence = self._confidence(kept_references, supporting_sources, requires_review)

        if self._contradicts_observation(generated.finding, generated.finding_type, observation):
            requires_review = True
            notes.append(
                "The observation reports the measure as being in place, while the drafted finding "
                "reports a shortcoming; the draft contradicts the auditor rather than reporting "
                "them. Flagged for review and confidence capped."
            )
            confidence = min(confidence, self.LOW_CONFIDENCE_REVIEW_THRESHOLD)

        unsupported_quantities = self._unsupported_quantities(generated.finding, observation, document_evidence)
        if unsupported_quantities:
            requires_review = True
            notes.append(
                "The drafted finding states figure(s) the observation does not report: "
                + ", ".join(unsupported_quantities)
                + ". A measured particular can only come from the auditor or from an imported "
                "client document, never from the knowledge base's example findings; flagged for "
                "review and confidence capped."
            )
            # Capped rather than scaled: whatever the evidence says, a finding that attributes an
            # unreported figure to this client cannot sit above the bar that means "a human need
            # not look at this".
            confidence = min(confidence, self.LOW_CONFIDENCE_REVIEW_THRESHOLD)

        document_sources = self._document_sources(document_evidence)
        if document_sources:
            # Deterministic backstop, independent of prompt wording or model behaviour: client
            # document context was used, so a human must look at this finding before it ships,
            # regardless of what the model itself claims about evidence_status/requires_human_review.
            requires_review = True
            notes.append(
                "Client document context was used (imported document evidence); flagged for review."
            )

        return AuditFinding(
            observation=observation,
            # `or generated.finding`: `finding` is required, never optional -- a defensive fallback
            # for the practically-impossible case where a finding's *entire* text was markdown
            # table syntax and nothing else, so sanitizing it wouldn't leave a required field blank.
            finding=_strip_markdown_table_artifacts(generated.finding) or generated.finding,
            finding_type=generated.finding_type,
            requirement=_strip_markdown_table_artifacts(generated.requirement),
            iso_reference=kept_references,
            justification=_strip_markdown_table_artifacts(generated.justification),
            risk=_strip_markdown_table_artifacts(generated.risk),
            recommendation=_strip_markdown_table_artifacts(generated.recommendation),
            evidence_status=evidence_status,
            requires_human_review=requires_review,
            confidence=confidence,
            supporting_sources=supporting_sources,
            document_sources=document_sources,
            validation_notes=notes,
        )

    @staticmethod
    def _contradicts_observation(finding: str, finding_type: FindingType, observation: str) -> bool:
        """True when the auditor reported that a measure is in place and the draft reports the
        opposite. See the module-level comment for why the corpus makes this the model's default
        failure mode, and why the check only looks in this one direction.

        Three gates, each narrowing further, so that only a purely positive observation paired
        with a deficiency finding can trigger:
          1. any shortcoming mentioned in the observation -> not our case, never flag;
          2. no explicit statement that the measure is in place -> nothing to contradict;
          3. the finding repeating that same positive statement -> it is reasoning from the
             auditor's words ("la revue est effectuée annuellement, alors que la politique exige
             une périodicité semestrielle"), which is an interpretation, not a contradiction.
        """
        folded_observation = observation.lower().translate(_ACCENT_FOLD)
        if _DEFICIENCY_CUE_RE.search(folded_observation):
            return False
        if not _CONFORMITY_OBSERVATION_RE.search(folded_observation):
            return False
        folded_finding = finding.lower().translate(_ACCENT_FOLD)
        if _CONFORMITY_FINDING_RE.search(folded_finding):
            return False
        return finding_type == FindingType.NON_CONFORMITE or bool(_DEFICIENCY_CUE_RE.search(folded_finding))

    @staticmethod
    def _unsupported_quantities(
        finding: str, observation: str, document_evidence: Sequence[SearchHit],
    ) -> List[str]:
        """Figures the drafted finding states about the audited client that the auditor's own
        observation (and any imported client document) never mentions.

        The live failure this exists for: the knowledge base is, by design, full of *worked
        example* findings, and the model reuses one wholesale -- figures included -- when an
        observation touches the same control. One draft reported "dernière revue documentée :
        14 mois" for a client whose observation said the review *was* performed quarterly; the
        figure came verbatim from the A.5.18 example record in the corpus. Every other check here
        passes it (the citation is real, the evidence backs it, the reranker scores it highly),
        because the sentence genuinely *is* supported by the retrieved text -- it is simply not
        about this client.

        So this check deliberately does not ask whether the evidence contains the figure: for a
        measured particular, ISO evidence is precedent, not measurement of this engagement. Only
        the auditor's own observation, or a document imported for this session, can establish one.
        Scoped to `finding` alone -- the sentence that asserts what was observed here -- since
        risk and recommendation text is normative and generic by nature.
        """
        stated = {match.group(1) for match in _QUANTIFIED_CLAIM_RE.finditer(finding)}
        if not stated:
            return []
        reported = observation + " " + " ".join(hit.payload.get("text", "") for hit in document_evidence)
        reported_numbers = set(re.findall(r"\d{1,4}", reported))
        return sorted(stated - reported_numbers, key=lambda n: finding.find(n))

    @staticmethod
    def _cited_references(generated: GeneratedFinding) -> List[str]:
        """Every reference the model cited, from `iso_reference` *and* from the drafted prose.

        Observed live: llama3:8b writes the reference into the sentence it drafts ("... en écart
        avec les lignes directrices (ISO/IEC 27002 §5.17)") and leaves `iso_reference` empty, or
        fills it with the evidence's document title instead of a reference. The citation is then
        real, visible to the auditor in the report text, and genuinely backed by the evidence --
        but invisible to a check that only reads the structured field, so it was dropped and the
        finding scored as if nothing had been cited.

        Harvesting is done with the same extractor `src/ingestion/references.py` used to index the
        knowledge base, and changes nothing about what is *trusted*: every reference found here
        goes through the same evidence check as a structured one, and a prose citation the
        evidence does not back is dropped and flagged exactly like a structured one would be.
        """
        prose = (generated.finding, generated.requirement, generated.justification,
                 generated.risk, generated.recommendation)
        harvested: List[str] = []
        for text in prose:
            if text:
                harvested.extend(extract_references(text))
        return list(dict.fromkeys(list(generated.iso_reference) + harvested))

    @staticmethod
    def _document_sources(document_evidence: Sequence[SearchHit]) -> List[SupportingSource]:
        """Every chunk shown to the model as client-document context, verbatim -- there is nothing
        to validate a citation against here (these chunks carry no ISO references), so unlike
        `_supporting_sources` this is not filtered by what the model cited."""
        return [
            SupportingSource(
                chunk_id=hit.payload["chunk_id"], doc_id=hit.payload["doc_id"],
                doc_title=hit.payload["doc_title"], references=[], rerank_score=hit.score,
            )
            for hit in document_evidence
        ]

    @staticmethod
    def _supporting_sources(evidence: Sequence[SearchHit], kept_references: List[str]) -> List[SupportingSource]:
        def source(hit: SearchHit, references: List[str]) -> SupportingSource:
            return SupportingSource(
                chunk_id=hit.payload["chunk_id"], doc_id=hit.payload["doc_id"],
                doc_title=hit.payload["doc_title"], references=references, rerank_score=hit.score,
            )

        if not kept_references:
            # Nothing was verifiably cited; still surface what was retrieved, so a human reviewer
            # can see what the system considered before judging the evidence insufficient.
            return [source(hit, hit.payload.get("references", [])) for hit in evidence]
        return [
            source(hit, [r for r in hit.payload.get("references", []) if r in kept_references])
            for hit in evidence
            if any(r in kept_references for r in hit.payload.get("references", []))
        ]

    @staticmethod
    def _reference_confidence(reference: str, sources: List[SupportingSource]) -> float:
        """How well the evidence backs *this one* reference: the reranker's judgment for the single
        chunk that backs it best. Best rather than mean, because a reference is established by the
        one chunk that genuinely covers it -- further chunks carrying the same reference more
        loosely are corroboration, and averaging them in would let breadth of retrieval lower the
        score of a reference the evidence squarely supports.
        """
        scores = [_sigmoid(s.rerank_score) for s in sources if reference in s.references]
        return max(scores, default=0.0)

    @staticmethod
    def _confidence(kept_references: List[str], sources: List[SupportingSource], requires_review: bool) -> float:
        """Derived from the reranker's own relevance judgment for the chunks backing the *kept*,
        validated references -- never from the model's self-reported `evidence_status` or a
        confidence number (the model is never asked for one). Zero when nothing was verifiably
        cited, regardless of what the model claimed: a model that says "supported" while citing
        nothing checkable is not more trustworthy than one that says "insufficient".

        The cross-encoder's raw score is an unbounded logit; a sigmoid maps it to a (0, 1)
        relevance-like value (a common, if informal, reading of MS MARCO-style cross-encoder
        scores).

        Scored per reference and then averaged, rather than over the pooled chunks: pooling let a
        finding be punished for citing *more*. Live-observed on a two-reference draft whose first
        reference was squarely backed (0.948) and whose second was backed only loosely -- the
        pooled mean of the five chunks involved fell to 0.346, a hair under the review threshold,
        so both references were discarded and the finding came back citing nothing at all.
        """
        if not kept_references:
            return 0.0
        backing = [
            score for score in (GroundingValidator._reference_confidence(r, sources) for r in kept_references)
            if score > 0.0
        ]
        if not backing:
            return 0.0
        base = sum(backing) / len(backing)
        if requires_review:
            base *= 0.85
        return round(min(max(base, 0.0), 1.0), 3)
