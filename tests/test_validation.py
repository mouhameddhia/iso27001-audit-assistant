"""GroundingValidator: the model's evidence_status is a claim, not a fact -- these tests check the
deterministic layer that actually enforces grounding, independent of what the model says about itself.
"""

import pytest

from src.generation.models import EvidenceStatus, FindingType, GeneratedFinding
from src.generation.validation import GroundingValidator
from tests.conftest import make_hit

EVIDENCE = [
    make_hit("doc-a:s05:c01", "NC sur les accès.", doc_id="doc-a", score=1.2, references=["ISO/IEC 27001 A.5.18"]),
    make_hit("doc-a:s05:c02", "NC sur les vulnérabilités.", doc_id="doc-a", score=-3.0, references=["ISO/IEC 27001 A.8.8"]),
]


def generated(**overrides) -> GeneratedFinding:
    defaults = dict(
        finding="Le constat.", finding_type=FindingType.NON_CONFORMITE, iso_reference=[],
        evidence_status=EvidenceStatus.SUPPORTED, requires_human_review=False,
    )
    return GeneratedFinding(**{**defaults, **overrides})


class TestReferenceMatching:
    def test_exact_canonical_reference_is_kept(self):
        result = GroundingValidator().validate("obs", generated(iso_reference=["ISO/IEC 27001 A.5.18"]), EVIDENCE)
        assert result.iso_reference == ["ISO/IEC 27001 A.5.18"]
        assert result.validation_notes == []

    def test_matching_is_case_and_whitespace_insensitive(self):
        result = GroundingValidator().validate("obs", generated(iso_reference=["  iso/iec 27001 a.5.18  "]), EVIDENCE)
        assert result.iso_reference == ["ISO/IEC 27001 A.5.18"]

    def test_bare_local_id_resolves_when_unambiguous(self):
        result = GroundingValidator().validate("obs", generated(iso_reference=["A.5.18"]), EVIDENCE)
        assert result.iso_reference == ["ISO/IEC 27001 A.5.18"]
        assert result.validation_notes == []

    def test_bare_local_id_is_dropped_when_ambiguous_across_standards(self):
        evidence = EVIDENCE + [make_hit("doc-b:s01:c00", "Autre norme.", doc_id="doc-b", references=["ISO/IEC 27017 A.5.18"])]
        result = GroundingValidator().validate("obs", generated(iso_reference=["A.5.18"]), evidence)
        assert result.iso_reference == []
        assert result.requires_human_review is True

    def test_unsupported_reference_is_dropped_and_noted(self):
        result = GroundingValidator().validate("obs", generated(iso_reference=["ISO/IEC 27001 A.9.9"]), EVIDENCE)
        assert result.iso_reference == []
        assert "A.9.9" in result.validation_notes[0]
        assert result.requires_human_review is True

    def test_a_casually_formatted_but_unambiguous_citation_is_recognised_not_dropped(self):
        """Regression test for a real, live-reported bug: the model cited "27002 §5.15" (no
        "ISO/IEC" word, a "§" the exact/local-id matches above don't strip) for a finding that
        *was* genuinely backed by "ISO/IEC 27002 5.15" in the evidence -- and it was silently
        dropped, so the final report showed "aucune référence spécifique" despite a well-grounded
        citation. Re-parsing the model's own citation with the same reference extractor that
        canonicalised the evidence itself is not a fuzzy guess -- it's the same deterministic rules,
        just applied to the model's text instead of a knowledge-base document -- so this must
        resolve, not drop. Covers the exact live cases: no "ISO/IEC" prefix, and a "§" before the
        number embedded in a fuller descriptive citation."""
        for citation in ["27002 §5.15", "ISO/IEC 27002 §5.15", "ISO/IEC 27001:2022, Annexe A.5.18"]:
            evidence = EVIDENCE + [make_hit("doc-c:s01:c00", "Moindre privilège.", doc_id="doc-c",
                                            references=["ISO/IEC 27002 5.15"])]
            result = GroundingValidator().validate("obs", generated(iso_reference=[citation]), evidence)
            assert result.iso_reference != [], citation

    def test_a_citation_that_still_cannot_be_resolved_is_dropped_not_guessed(self):
        """The re-parsing fallback (above) is not a blank cheque: text that genuinely doesn't
        parse into a reference at all, or that the knowledge base's own extractor can't place,
        stays dropped -- the safe failure direction is still a false negative, not a fuzzy accept."""
        result = GroundingValidator().validate(
            "obs", generated(iso_reference=["quelque part dans les contrôles applicables"]), EVIDENCE,
        )
        assert result.iso_reference == []
        assert result.requires_human_review is True

    def test_a_reparsed_citation_ambiguous_across_evidenced_standards_is_still_dropped(self):
        """`extract_references` alone would default a bare "A.5.18" to ISO/IEC 27001 by audit
        convention -- correct for parsing a knowledge-base document, but wrong here: when the
        evidence set itself contains the same local id under two different standards, that is a
        real ambiguity the evidence raises, and re-parsing must not paper over it."""
        evidence = EVIDENCE + [make_hit("doc-b:s01:c00", "Autre norme.", doc_id="doc-b", references=["ISO/IEC 27017 A.5.18"])]
        result = GroundingValidator().validate(
            "obs", generated(iso_reference=["cf. l'annexe A.5.18"]), evidence,
        )
        assert result.iso_reference == []

    def test_mix_of_supported_and_unsupported_keeps_only_the_supported_one(self):
        result = GroundingValidator().validate(
            "obs", generated(iso_reference=["ISO/IEC 27001 A.5.18", "ISO/IEC 27001 A.9.9"]), EVIDENCE
        )
        assert result.iso_reference == ["ISO/IEC 27001 A.5.18"]

    def test_duplicate_citations_are_not_duplicated_in_output(self):
        result = GroundingValidator().validate(
            "obs", generated(iso_reference=["A.5.18", "ISO/IEC 27001 A.5.18"]), EVIDENCE
        )
        assert result.iso_reference == ["ISO/IEC 27001 A.5.18"]


class TestEvidenceStatusDowngrade:
    def test_supported_with_a_dropped_reference_and_no_other_becomes_insufficient(self):
        result = GroundingValidator().validate(
            "obs", generated(iso_reference=["A.9.9"], evidence_status=EvidenceStatus.SUPPORTED), EVIDENCE
        )
        assert result.evidence_status == EvidenceStatus.INSUFFICIENT

    def test_supported_with_one_dropped_and_one_kept_becomes_partial(self):
        result = GroundingValidator().validate(
            "obs", generated(iso_reference=["A.5.18", "A.9.9"], evidence_status=EvidenceStatus.SUPPORTED), EVIDENCE
        )
        assert result.evidence_status == EvidenceStatus.PARTIAL
        assert result.iso_reference == ["ISO/IEC 27001 A.5.18"]

    def test_supported_with_empty_iso_reference_is_downgraded(self):
        result = GroundingValidator().validate(
            "obs", generated(iso_reference=[], evidence_status=EvidenceStatus.SUPPORTED, requires_human_review=False), EVIDENCE
        )
        assert result.evidence_status == EvidenceStatus.INSUFFICIENT
        assert result.requires_human_review is True

    def test_no_evidence_forces_insufficient_even_if_the_model_claimed_support(self):
        result = GroundingValidator().validate(
            "obs", generated(iso_reference=["ISO/IEC 27001 A.5.18"], evidence_status=EvidenceStatus.SUPPORTED), []
        )
        assert result.evidence_status == EvidenceStatus.INSUFFICIENT
        assert result.iso_reference == []
        assert result.requires_human_review is True

    def test_honest_insufficient_with_no_citation_passes_through_unchanged(self):
        result = GroundingValidator().validate(
            "obs", generated(iso_reference=[], evidence_status=EvidenceStatus.INSUFFICIENT, requires_human_review=True), EVIDENCE
        )
        assert result.evidence_status == EvidenceStatus.INSUFFICIENT
        assert result.validation_notes == []  # nothing needed correcting


class TestLowConfidenceReviewOverride:
    def test_weakly_backed_reference_forces_review_even_if_the_model_did_not_ask_for_it(self):
        weak_evidence = [make_hit("doc-a:s01:c00", "x", doc_id="doc-a", score=-6.0, references=["ISO/IEC 27001 A.8.8"])]
        result = GroundingValidator().validate(
            "obs", generated(iso_reference=["ISO/IEC 27001 A.8.8"], requires_human_review=False), weak_evidence
        )
        assert result.requires_human_review is True
        assert any("low" in note.lower() for note in result.validation_notes)

    def test_strongly_backed_reference_does_not_force_review(self):
        strong_evidence = [make_hit("doc-a:s01:c00", "x", doc_id="doc-a", score=5.0, references=["ISO/IEC 27001 A.5.18"])]
        result = GroundingValidator().validate(
            "obs", generated(iso_reference=["ISO/IEC 27001 A.5.18"], requires_human_review=False), strong_evidence
        )
        assert result.requires_human_review is False

    def test_no_citation_is_not_penalised_a_second_time_by_the_low_confidence_note(self):
        result = GroundingValidator().validate(
            "obs", generated(iso_reference=[], evidence_status=EvidenceStatus.INSUFFICIENT, requires_human_review=True), EVIDENCE
        )
        assert result.validation_notes == []


class TestSelfContradictionResolvedByEvidenceStrength:
    """Live, reproducible bug: llama3:8b sometimes cites a real, well-evidenced reference while
    *also* labelling its own answer evidence_status="insufficient". The old behaviour trusted
    "insufficient" unconditionally and threw the citation away -- discarding real traceability the
    evidence itself supported. Fixed the same way every other judgment call in this validator is
    made: by the reranker's own independent score, never by which of the model's two contradictory
    claims happens to be checked first. Which half of the contradiction is "safe" to trust depends
    on whether the *evidence* backs the citation, not on the model's word either way -- hence four
    cases, crossing {supported, insufficient} x {strong, weak} evidence.
    """

    STRONG_EVIDENCE = [make_hit("doc-a:s01:c00", "x", doc_id="doc-a", score=5.0, references=["ISO/IEC 27001 A.5.18"])]
    WEAK_EVIDENCE = [make_hit("doc-a:s01:c00", "x", doc_id="doc-a", score=-6.0, references=["ISO/IEC 27001 A.5.18"])]

    def test_strong_evidence_plus_supported_keeps_the_citation_and_does_not_force_review(self):
        result = GroundingValidator().validate(
            "obs", generated(iso_reference=["ISO/IEC 27001 A.5.18"], evidence_status=EvidenceStatus.SUPPORTED,
                             requires_human_review=False),
            self.STRONG_EVIDENCE,
        )
        assert result.iso_reference == ["ISO/IEC 27001 A.5.18"]
        assert result.evidence_status == EvidenceStatus.SUPPORTED
        assert result.requires_human_review is False

    def test_strong_evidence_plus_insufficient_keeps_the_citation_but_forces_review(self):
        """The exact live bug: a real, strongly-backed citation must never be silently discarded
        just because the model also hedged -- but a self-contradiction is never auto-approved
        either, so this must never come back as `evidence_status=SUPPORTED` and must always be
        flagged for a human to look at."""
        result = GroundingValidator().validate(
            "obs", generated(iso_reference=["ISO/IEC 27001 A.5.18"], evidence_status=EvidenceStatus.INSUFFICIENT,
                             requires_human_review=False),
            self.STRONG_EVIDENCE,
        )
        assert result.iso_reference == ["ISO/IEC 27001 A.5.18"]
        assert result.evidence_status == EvidenceStatus.PARTIAL
        assert result.requires_human_review is True
        assert result.confidence > 0.0  # a real, kept, evidenced citation -- not treated as worthless
        assert any("contradicted itself" in note for note in result.validation_notes)

    def test_weak_evidence_plus_supported_does_not_accept_the_citation_automatically(self):
        """A model confidently saying "supported" is not enough on its own when the independent
        reranker signal says the match is weak -- forced to review either way, the same guarantee
        `TestLowConfidenceReviewOverride` already covers for the non-contradictory case."""
        result = GroundingValidator().validate(
            "obs", generated(iso_reference=["ISO/IEC 27001 A.5.18"], evidence_status=EvidenceStatus.SUPPORTED,
                             requires_human_review=False),
            self.WEAK_EVIDENCE,
        )
        assert result.iso_reference == ["ISO/IEC 27001 A.5.18"]  # a real reference, just weakly backed -- not dropped
        assert result.requires_human_review is True

    def test_weak_evidence_plus_insufficient_drops_the_citation_and_stays_insufficient(self):
        """Both signals agree the finding is not well-grounded: the pre-existing, conservative
        behaviour must be unchanged -- the citation is discarded, not kept "just in case"."""
        result = GroundingValidator().validate(
            "obs", generated(iso_reference=["ISO/IEC 27001 A.5.18"], evidence_status=EvidenceStatus.INSUFFICIENT,
                             requires_human_review=False),
            self.WEAK_EVIDENCE,
        )
        assert result.iso_reference == []
        assert result.evidence_status == EvidenceStatus.INSUFFICIENT
        assert result.requires_human_review is True
        assert result.confidence == 0.0


class TestSupportingSources:
    def test_only_chunks_backing_a_kept_reference_are_included(self):
        result = GroundingValidator().validate("obs", generated(iso_reference=["ISO/IEC 27001 A.5.18"]), EVIDENCE)
        assert [s.chunk_id for s in result.supporting_sources] == ["doc-a:s05:c01"]
        assert result.supporting_sources[0].references == ["ISO/IEC 27001 A.5.18"]

    def test_all_evidence_is_surfaced_when_nothing_was_kept(self):
        result = GroundingValidator().validate("obs", generated(iso_reference=["A.9.9"]), EVIDENCE)
        assert {s.chunk_id for s in result.supporting_sources} == {"doc-a:s05:c01", "doc-a:s05:c02"}

    def test_a_chunks_own_unrelated_references_are_not_shown_as_supporting(self):
        evidence = [make_hit("doc-a:s01:c00", "x", doc_id="doc-a", score=1.0,
                             references=["ISO/IEC 27001 A.5.18", "ISO/IEC 27001 A.8.8"])]
        result = GroundingValidator().validate("obs", generated(iso_reference=["ISO/IEC 27001 A.5.18"]), evidence)
        assert result.supporting_sources[0].references == ["ISO/IEC 27001 A.5.18"]


class TestInsufficientEvidenceAlwaysForcesReview:
    """Regression test for a real gap found live: the model can correctly abstain (empty
    iso_reference, evidence_status='insufficient') while still leaving requires_human_review=False
    on its own -- no other check above touches `requires_review` in that specific path (nothing was
    dropped, evidence wasn't empty, no kept references to contradict). 'Insufficient evidence to
    confirm a finding' must always mean a human looks at it, regardless of the model's own flag."""

    def test_forces_review_even_when_the_model_says_it_is_not_needed(self):
        result = GroundingValidator().validate(
            "obs", generated(iso_reference=[], evidence_status=EvidenceStatus.INSUFFICIENT,
                             requires_human_review=False),
            EVIDENCE,
        )
        assert result.requires_human_review is True
        assert any("forced for human review" in note.lower() for note in result.validation_notes)

    def test_does_not_duplicate_the_note_when_the_model_already_asked_for_review(self):
        result = GroundingValidator().validate(
            "obs", generated(iso_reference=[], evidence_status=EvidenceStatus.INSUFFICIENT,
                             requires_human_review=True),
            EVIDENCE,
        )
        assert result.requires_human_review is True
        assert not any("forced for human review" in note.lower() for note in result.validation_notes)


class TestConfidence:
    def test_zero_when_nothing_was_kept(self):
        result = GroundingValidator().validate("obs", generated(iso_reference=[], evidence_status=EvidenceStatus.INSUFFICIENT), EVIDENCE)
        assert result.confidence == 0.0

    def test_zero_when_no_evidence_even_if_model_claimed_support(self):
        result = GroundingValidator().validate("obs", generated(iso_reference=["ISO/IEC 27001 A.5.18"]), [])
        assert result.confidence == 0.0

    def test_positive_when_a_reference_is_kept_and_well_backed(self):
        result = GroundingValidator().validate("obs", generated(iso_reference=["ISO/IEC 27001 A.5.18"]), EVIDENCE)
        assert 0.0 < result.confidence <= 1.0

    def test_higher_reranker_score_gives_higher_confidence(self):
        strong = [make_hit("doc-a:s01:c00", "x", doc_id="doc-a", score=5.0, references=["ISO/IEC 27001 A.5.18"])]
        weak = [make_hit("doc-a:s01:c00", "x", doc_id="doc-a", score=-5.0, references=["ISO/IEC 27001 A.5.18"])]
        strong_result = GroundingValidator().validate("obs", generated(iso_reference=["ISO/IEC 27001 A.5.18"]), strong)
        weak_result = GroundingValidator().validate("obs", generated(iso_reference=["ISO/IEC 27001 A.5.18"]), weak)
        assert strong_result.confidence > weak_result.confidence

    def test_confidence_is_always_in_bounds(self):
        result = GroundingValidator().validate("obs", generated(iso_reference=["ISO/IEC 27001 A.5.18"]), EVIDENCE)
        assert 0.0 <= result.confidence <= 1.0


class TestConfidenceIsScoredPerReference:
    """Live-observed: a draft citing two references, the first squarely backed (0.948) and the
    second only loosely, came back citing *nothing*. Pooling every backing chunk into one mean
    dropped the aggregate to 0.346 -- a hair under the review threshold -- so both references were
    discarded together. A finding must not be punished for citing more."""

    MIXED_EVIDENCE = [
        make_hit("doc-a:s05:c04", "x", doc_id="doc-a", score=2.90, references=["ISO/IEC 27002 5.17"]),
        make_hit("doc-a:s04:c04", "x", doc_id="doc-a", score=0.48, references=["ISO/IEC 27002 5.17"]),
        make_hit("doc-a:s05:c02", "x", doc_id="doc-a", score=-1.77, references=["ISO/IEC 27002 8.13"]),
        make_hit("doc-a:s06:c02", "x", doc_id="doc-a", score=-4.19, references=["ISO/IEC 27002 8.13"]),
        make_hit("doc-a:s04:c02", "x", doc_id="doc-a", score=-5.71, references=["ISO/IEC 27002 8.13"]),
    ]
    CITED = ["ISO/IEC 27002 5.17", "ISO/IEC 27002 8.13"]

    def test_the_well_backed_reference_survives_a_loosely_backed_neighbour(self):
        result = GroundingValidator().validate(
            "obs",
            generated(iso_reference=self.CITED, evidence_status=EvidenceStatus.INSUFFICIENT),
            self.MIXED_EVIDENCE,
        )
        assert "ISO/IEC 27002 5.17" in result.iso_reference
        assert result.confidence > 0.0

    def test_the_loosely_backed_one_is_still_not_kept_silently(self):
        result = GroundingValidator().validate(
            "obs",
            generated(iso_reference=self.CITED, evidence_status=EvidenceStatus.INSUFFICIENT),
            self.MIXED_EVIDENCE,
        )
        assert "ISO/IEC 27002 8.13" not in result.iso_reference
        assert result.requires_human_review is True
        assert any("too loosely backed" in note for note in result.validation_notes)

    def test_extra_loosely_related_chunks_do_not_lower_a_references_own_score(self):
        alone = [self.MIXED_EVIDENCE[0]]
        with_corroboration = [self.MIXED_EVIDENCE[0], self.MIXED_EVIDENCE[1]]
        cited = ["ISO/IEC 27002 5.17"]
        assert (
            GroundingValidator().validate("obs", generated(iso_reference=cited), with_corroboration).confidence
            == GroundingValidator().validate("obs", generated(iso_reference=cited), alone).confidence
        )


class TestReferencesCitedInTheDraftedProse:
    """Live-observed: llama3:8b writes the reference into the sentence it drafts and leaves
    `iso_reference` empty (or fills it with the evidence's document title). The citation is real,
    backed by the evidence, and visible to the auditor -- but a check reading only the structured
    field dropped it and scored the finding as if nothing had been cited."""

    def test_a_reference_written_only_in_the_prose_is_recovered(self):
        result = GroundingValidator().validate(
            "obs",
            generated(finding="Les droits ne sont pas revus (ISO/IEC 27001 A.5.18).", iso_reference=[]),
            EVIDENCE,
        )
        assert result.iso_reference == ["ISO/IEC 27001 A.5.18"]
        assert result.confidence > 0.0

    def test_a_prose_reference_the_evidence_does_not_back_is_still_dropped(self):
        result = GroundingValidator().validate(
            "obs", generated(finding="Écart au regard d'ISO/IEC 27001 A.9.99.", iso_reference=[]), EVIDENCE,
        )
        assert result.iso_reference == []
        assert result.requires_human_review is True

    def test_prose_and_structured_citations_are_not_duplicated(self):
        result = GroundingValidator().validate(
            "obs",
            generated(finding="Manquement A.5.18 constaté.", iso_reference=["ISO/IEC 27001 A.5.18"]),
            EVIDENCE,
        )
        assert result.iso_reference == ["ISO/IEC 27001 A.5.18"]

    def test_the_risk_field_is_read_too_not_just_the_finding(self):
        result = GroundingValidator().validate(
            "obs", generated(risk="Exposition au regard d'ISO/IEC 27001 A.8.8.", iso_reference=[]), EVIDENCE,
        )
        assert result.iso_reference == ["ISO/IEC 27001 A.8.8"]


class TestContradictingTheObservation:
    """The corpus holds only worked examples of *deficiencies*, so retrieval on a control returns
    deficiency text whatever the auditor reported, and the model completes the pattern. Live-
    observed: an observation stating the access review *is* performed quarterly and documented
    produced a non-conformity saying it had not been performed on time -- scoring 0.758, because
    every other signal (real citation, strong evidence, high reranker score) was genuinely fine.

    The check only looks in that one direction. The cases below are the calibration set: the eight
    observations from the live test batch, plus the traps that a lexical rule is most likely to get
    wrong -- mixed sentences, deficiencies stated without a negation, neutral statements of fact,
    and observations that are positive in grammar but non-compliant in substance.
    """

    CONTRADICTIONS = [
        ("La revue des accès privilégiés est effectuée trimestriellement et documentée, conformément à la procédure interne.",
         "La revue trimestrielle des accès privilégiés n'a pas été réalisée dans les délais pour le dernier cycle."),
        ("Les sauvegardes sont testées mensuellement et les résultats sont documentés.",
         "Aucun test de restauration n'a été réalisé depuis la mise en production."),
        ("Le contrôle d'accès par badge est en place sur l'ensemble des locaux techniques.",
         "La salle serveurs est accessible par une porte non verrouillée pendant les heures de bureau."),
        ("La politique d'accès est documentée et approuvée par la direction.",
         "Les droits d'accès privilégiés ne sont pas revus périodiquement."),
        ("Le chiffrement est appliqué sur les serveurs de production.",
         "Le chiffrement n'est pas appliqué sur les postes nomades."),
        ("Revue trimestrielle effectuée, preuves fournies.", "La revue n'a pas été réalisée dans les délais."),
        ("Privileged access reviews are performed quarterly and documented.",
         "The periodic review of privileged access rights was not performed on time."),
    ]

    NOT_CONTRADICTIONS = [
        # The auditor already reported a shortcoming: the draft agrees with them.
        ("Le contrôle des accès privilégiés n'est pas revu périodiquement.",
         "Le contrôle des accès privilégiés n'est pas revu périodiquement."),
        ("La sécurité physique du site n'est pas satisfaisante.",
         "La sécurité physique n'est pas satisfaisante en raison de l'accès non contrôlé à la salle serveurs."),
        ("Le serveur SRV-PROD-042 est accessible avec le compte USER_JDUPONT sans authentification multifacteur.",
         "Le serveur est accessible sans authentification multifacteur."),
        # Conformity and shortcoming in the same sentence.
        ("La politique est documentée mais n'a pas été revue depuis 2019.", "La politique n'a pas été revue depuis 2019."),
        ("Les sauvegardes sont effectuées quotidiennement, cependant aucun test de restauration n'est réalisé.",
         "Aucun test de restauration n'est réalisé."),
        ("La procédure est conforme à la norme sans toutefois couvrir les prestataires externes.",
         "La procédure ne couvre pas les prestataires externes."),
        # A shortcoming stated without any negation.
        ("Les mots de passe par défaut du constructeur sont toujours en place sur les équipements.",
         "Les mots de passe par défaut sont toujours en place."),
        ("L'API publique accepte encore des connexions TLS 1.0.", "L'API accepte TLS 1.0, protocole déprécié."),
        # A neutral statement of fact claims nothing is in place -- "revue", "document" and "test"
        # are ordinary nouns here, not participles.
        ("La dernière revue des accès remonte à 14 mois.", "La revue n'a pas été réalisée dans les délais."),
        ("Le document de politique de sécurité date de 2019.", "La politique n'a pas été revue depuis 2019."),
        ("Le test de restauration a lieu une fois par an.", "La fréquence des tests de restauration est insuffisante."),
        # Positive in grammar, non-compliant in substance: the draft reasons from the auditor's own
        # words instead of denying them.
        ("La revue des accès est effectuée annuellement.",
         "La revue des accès est effectuée annuellement alors que la politique exige une périodicité semestrielle."),
        ("Les journaux sont conservés 7 jours.", "Les journaux sont conservés 7 jours, en deçà des 12 mois exigés."),
    ]

    @pytest.mark.parametrize("observation, finding", CONTRADICTIONS)
    def test_a_draft_denying_the_observation_is_flagged(self, observation, finding):
        result = GroundingValidator().validate(
            observation, generated(finding=finding, iso_reference=["ISO/IEC 27001 A.5.18"]), EVIDENCE,
        )
        assert result.requires_human_review is True
        assert any("contradicts the auditor" in note for note in result.validation_notes)
        assert result.confidence <= GroundingValidator.LOW_CONFIDENCE_REVIEW_THRESHOLD

    @pytest.mark.parametrize("observation, finding", NOT_CONTRADICTIONS)
    def test_an_honest_draft_is_not_flagged(self, observation, finding):
        result = GroundingValidator().validate(
            observation, generated(finding=finding, iso_reference=["ISO/IEC 27001 A.5.18"]), EVIDENCE,
        )
        assert not any("contradicts the auditor" in note for note in result.validation_notes)

    def test_a_positive_observation_may_still_yield_a_neutral_finding(self):
        """Reporting back that a control is correctly operated is not a contradiction."""
        result = GroundingValidator().validate(
            "La revue des accès est effectuée trimestriellement et documentée.",
            generated(finding="La revue des accès privilégiés est réalisée selon la périodicité définie.",
                      finding_type=FindingType.CONSTAT, iso_reference=["ISO/IEC 27001 A.5.18"]),
            EVIDENCE,
        )
        assert not any("contradicts the auditor" in note for note in result.validation_notes)

    def test_a_positive_observation_may_still_yield_an_improvement_opportunity(self):
        result = GroundingValidator().validate(
            "Le chiffrement est en place sur l'ensemble des postes, conformément à la politique.",
            generated(finding="Le dispositif pourrait être étendu aux services cloud managés.",
                      finding_type=FindingType.OPPORTUNITE_AMELIORATION, iso_reference=["ISO/IEC 27001 A.5.18"]),
            EVIDENCE,
        )
        assert not any("contradicts the auditor" in note for note in result.validation_notes)


class TestUnsupportedQuantities:
    """The corpus is full of worked *example* findings, and the model reuses one wholesale --
    figures included -- when an observation touches the same control. Live-observed: a draft
    reporting "dernière revue documentée : 14 mois" for a client whose observation said the review
    *was* performed quarterly, the figure coming verbatim from the A.5.18 example record. Every
    other check passes it, so it scored higher than the honest findings around it."""

    OBSERVATION = "La revue des accès privilégiés est effectuée trimestriellement et documentée."
    COPIED = "La revue n'a pas été réalisée dans les délais (dernière revue documentée : 14 mois)."

    def test_a_figure_the_observation_never_reports_forces_review(self):
        result = GroundingValidator().validate(
            self.OBSERVATION, generated(finding=self.COPIED, iso_reference=["ISO/IEC 27001 A.5.18"]), EVIDENCE,
        )
        assert result.requires_human_review is True
        assert any("does not report" in note for note in result.validation_notes)

    def test_such_a_finding_can_never_outrank_the_review_threshold(self):
        result = GroundingValidator().validate(
            self.OBSERVATION, generated(finding=self.COPIED, iso_reference=["ISO/IEC 27001 A.5.18"]), EVIDENCE,
        )
        assert result.confidence <= GroundingValidator.LOW_CONFIDENCE_REVIEW_THRESHOLD

    def test_a_figure_the_auditor_did_report_is_not_flagged(self):
        result = GroundingValidator().validate(
            "La dernière revue des accès remonte à 14 mois.",
            generated(finding=self.COPIED, iso_reference=["ISO/IEC 27001 A.5.18"]), EVIDENCE,
        )
        assert not any("does not report" in note for note in result.validation_notes)
        assert result.confidence > GroundingValidator.LOW_CONFIDENCE_REVIEW_THRESHOLD

    def test_a_finding_without_figures_is_untouched(self):
        result = GroundingValidator().validate(
            "Les accès ne sont pas revus.",
            generated(finding="Les accès privilégiés ne sont pas revus périodiquement.",
                      iso_reference=["ISO/IEC 27001 A.5.18"]),
            EVIDENCE,
        )
        assert not any("does not report" in note for note in result.validation_notes)

    @pytest.mark.parametrize("finding", [
        "Écart au regard d'ISO/IEC 27001 A.5.18 et de la clause 8.1.",
        "L'API accepte encore TLS 1.0 au lieu de TLS 1.2.",
        "Le chiffrement AES-256 n'est pas appliqué.",
    ])
    def test_standard_and_algorithm_numbering_is_not_mistaken_for_a_measurement(self, finding):
        result = GroundingValidator().validate(
            "obs", generated(finding=finding, iso_reference=["ISO/IEC 27001 A.5.18"]), EVIDENCE,
        )
        assert not any("does not report" in note for note in result.validation_notes)


class TestFindingLabelConsistency:
    """Regression tests for a real, live-observed failure mode: the model sometimes drafts a bare
    generic type label as the whole `finding` text (e.g. finding="Opportunité d'amélioration")
    while finding_type disagrees -- a self-contradiction the model's own output makes checkable
    without any extra LLM call, the same "never trust the model's self-consistency" posture as the
    evidence_status/iso_reference check above."""

    def test_bare_label_contradicting_finding_type_is_flagged(self):
        result = GroundingValidator().validate(
            "obs", generated(finding="Opportunité d'amélioration", finding_type=FindingType.NON_CONFORMITE,
                             iso_reference=["ISO/IEC 27001 A.5.18"]),
            EVIDENCE,
        )
        assert result.requires_human_review is True
        assert any("bare label" in note.lower() for note in result.validation_notes)

    def test_bare_label_agreeing_with_finding_type_is_not_flagged_for_this_reason(self):
        result = GroundingValidator().validate(
            "obs", generated(finding="Non-conformité", finding_type=FindingType.NON_CONFORMITE,
                             iso_reference=["ISO/IEC 27001 A.5.18"], requires_human_review=False),
            EVIDENCE,
        )
        assert not any("bare label" in note.lower() for note in result.validation_notes)

    def test_a_real_drafted_sentence_is_never_flagged_by_this_check(self):
        result = GroundingValidator().validate(
            "obs", generated(
                finding="Le contrôle des accès privilégiés n'est pas revu périodiquement.",
                finding_type=FindingType.NON_CONFORMITE, iso_reference=["ISO/IEC 27001 A.5.18"],
                requires_human_review=False,
            ),
            EVIDENCE,
        )
        assert not any("bare label" in note.lower() for note in result.validation_notes)

    @pytest.mark.parametrize("label,matching_type", [
        ("Constat", FindingType.CONSTAT),
        ("Non-conformité", FindingType.NON_CONFORMITE),
        ("Observation", FindingType.OBSERVATION),
        ("Opportunité d'amélioration", FindingType.OPPORTUNITE_AMELIORATION),
        ("  constat.", FindingType.CONSTAT),  # whitespace/punctuation/case-insensitive
    ])
    def test_every_canonical_label_maps_to_its_own_type(self, label, matching_type):
        other_type = FindingType.CONSTAT if matching_type != FindingType.CONSTAT else FindingType.OBSERVATION
        result = GroundingValidator().validate(
            "obs", generated(finding=label, finding_type=other_type, iso_reference=[]), EVIDENCE,
        )
        assert any("bare label" in note.lower() for note in result.validation_notes)


class TestRiskField:
    def test_risk_is_passed_through_from_the_model(self):
        result = GroundingValidator().validate(
            "obs", generated(risk="Perte de confidentialité en cas d'accès non autorisé."), EVIDENCE,
        )
        assert result.risk == "Perte de confidentialité en cas d'accès non autorisé."

    def test_risk_defaults_to_none(self):
        result = GroundingValidator().validate("obs", generated(), EVIDENCE)
        assert result.risk is None


class TestMarkdownArtifactsAreStripped:
    """Regression tests for a real, live-reported bug: the model occasionally drafted a list as a
    Markdown table (e.g. an access-rights breakdown), and the raw pipe/dash syntax showed up as
    literal garbage ("| |", "|-|") in the final Word/PDF report -- these fields are meant for a
    formal report paragraph, not a chat UI that renders Markdown."""

    def test_a_separator_row_is_removed_entirely(self):
        result = GroundingValidator().validate(
            "obs", generated(justification="Texte avant.\n| Nom | Droit |\n|-|-|\n| a | b |\nTexte après."),
            EVIDENCE,
        )
        assert "|-|" not in result.justification
        assert "---" not in result.justification

    def test_a_real_data_row_is_turned_into_readable_text_not_left_as_raw_pipes(self):
        result = GroundingValidator().validate(
            "obs", generated(justification="| Utilisateur | Droit |\n| Compte A | Écriture |"), EVIDENCE,
        )
        assert "|" not in result.justification
        assert "Utilisateur" in result.justification and "Droit" in result.justification

    def test_ordinary_prose_with_no_pipe_character_is_left_byte_for_byte_unchanged(self):
        text = "Un risque de fraude interne - facilité par une ségrégation des tâches insuffisante."
        result = GroundingValidator().validate("obs", generated(risk=text), EVIDENCE)
        assert result.risk == text

    def test_applies_to_every_free_text_field(self):
        result = GroundingValidator().validate(
            "obs",
            generated(
                finding="| Constat | a |\n|-|-|", requirement="| Exigence | a |\n|-|-|",
                justification="| Justification | a |\n|-|-|", risk="| Risque | a |\n|-|-|",
                recommendation="| Recommandation | a |\n|-|-|",
            ),
            EVIDENCE,
        )
        for field in (result.finding, result.requirement, result.justification, result.risk, result.recommendation):
            assert "|-|" not in field and "|" not in field


class TestDocumentSources:
    DOCUMENT_EVIDENCE = [make_hit("doc-client:s01:c00", "Le pare-feu accepte TLS 1.0.", doc_id="doc-client", score=2.0)]

    def test_absent_when_no_document_evidence_is_given(self):
        result = GroundingValidator().validate("obs", generated(iso_reference=["ISO/IEC 27001 A.5.18"]), EVIDENCE)
        assert result.document_sources == []

    def test_populated_from_document_evidence_verbatim(self):
        result = GroundingValidator().validate(
            "obs", generated(iso_reference=["ISO/IEC 27001 A.5.18"]), EVIDENCE, self.DOCUMENT_EVIDENCE,
        )
        assert len(result.document_sources) == 1
        assert result.document_sources[0].chunk_id == "doc-client:s01:c00"
        assert result.document_sources[0].doc_id == "doc-client"

    def test_forces_human_review_even_if_the_model_did_not_ask_for_it(self):
        result = GroundingValidator().validate(
            "obs",
            generated(iso_reference=["ISO/IEC 27001 A.5.18"], requires_human_review=False),
            EVIDENCE, self.DOCUMENT_EVIDENCE,
        )
        assert result.requires_human_review is True

    def test_forces_human_review_regardless_of_evidence_status(self):
        result = GroundingValidator().validate(
            "obs",
            generated(iso_reference=["ISO/IEC 27001 A.5.18"], evidence_status=EvidenceStatus.SUPPORTED,
                     requires_human_review=False),
            EVIDENCE, self.DOCUMENT_EVIDENCE,
        )
        assert result.requires_human_review is True
        assert any("document" in note.lower() for note in result.validation_notes)

    def test_document_evidence_never_carries_iso_references(self):
        result = GroundingValidator().validate("obs", generated(), EVIDENCE, self.DOCUMENT_EVIDENCE)
        assert result.document_sources[0].references == []
