"""Tests for the Stage 2 anonymizer: detection, consistency, ISO-terminology preservation."""

import pytest

from src.anonymization.anonymizer import PLACEHOLDER_RE, Anonymizer

CAHIER_DES_CHARGES_EXAMPLE = (
    "Le serveur SRV-PROD-01 de la société ABC utilise le compte admin.",
    "Le serveur SERVER_001 de la société CLIENT_001 utilise le compte USER_001.",
)

ISO_SENTENCES = [
    "Le contrôle des accès privilégiés n'est pas revu périodiquement selon A.5.18.",
    "Le chiffrement AES-256 et TLS 1.3 sont conformes à ISO/IEC 27001 et RGPD.",
    "L'authentification MFA et le contrôle RBAC ne sont pas appliqués (Clause 9.2, A.8.8).",
    "La revue des habilitations n'a pas été réalisée dans les délais définis par la politique.",
]


def test_cahier_des_charges_example_matches_exactly():
    anonymizer = Anonymizer()
    original, expected = CAHIER_DES_CHARGES_EXAMPLE

    anonymized, mapping = anonymizer.anonymize(original)

    assert anonymized == expected
    assert anonymizer.deanonymize(anonymized, mapping) == original


class TestEntityDetection:
    def test_ip_address(self):
        anonymizer = Anonymizer()
        anonymized, mapping = anonymizer.anonymize("Le serveur 192.168.1.20 n'est pas protégé.")
        assert "192.168.1.20" not in anonymized
        assert any(v == "192.168.1.20" for v in mapping.values())

    def test_email(self):
        anonymizer = Anonymizer()
        anonymized, mapping = anonymizer.anonymize("Contact : j.dupont@abc-consulting.fr pour les détails.")
        assert "j.dupont@abc-consulting.fr" not in anonymized
        assert any(v == "j.dupont@abc-consulting.fr" for v in mapping.values())

    def test_server_name_keyword_is_kept_value_is_hidden(self):
        anonymizer = Anonymizer()
        anonymized, mapping = anonymizer.anonymize("Le serveur SRV-PROD-01 hébergeait des données non chiffrées.")
        assert "SRV-PROD-01" not in anonymized
        assert "serveur" in anonymized.lower()
        assert list(mapping.values()) == ["SRV-PROD-01"]

    def test_user_account_french_and_english_forms(self):
        anonymizer = Anonymizer()
        for text, expected_value in [
            ("Le compte utilisateur: j.dupont dispose de droits admin.", "j.dupont"),
            ("Le compte j.martin a été désactivé.", "j.martin"),
            ("Username: j.durand a un accès permanent.", "j.durand"),
            ("Login: admin01 utilisé pour les tâches planifiées.", "admin01"),
        ]:
            _, mapping = anonymizer.anonymize(text)
            assert expected_value in mapping.values(), text

    def test_application_name(self):
        anonymizer = Anonymizer()
        anonymized, mapping = anonymizer.anonymize("Le logiciel SAP n'est pas correctement paramétré.")
        assert "SAP" not in anonymized
        assert list(mapping.values()) == ["SAP"]

    def test_company_name_after_societe_or_entreprise(self):
        anonymizer = Anonymizer()
        for text, expected_value in [
            ("Chez la société ABC, le contrôle n'est pas appliqué.", "ABC"),
            ("L'entreprise XYZ-Corp ne dispose pas de politique formalisée.", "XYZ-Corp"),
        ]:
            _, mapping = anonymizer.anonymize(text)
            assert expected_value in mapping.values(), text

    def test_known_entity_is_matched_before_generic_rules(self):
        anonymizer = Anonymizer(known_entities={"CLIENT": ["ABC Consulting"]})
        anonymized, mapping = anonymizer.anonymize("Chez ABC Consulting, le contrôle n'est pas appliqué.")
        assert "ABC Consulting" not in anonymized
        assert mapping["CLIENT_001"] == "ABC Consulting"

    def test_longest_known_entity_matched_first(self):
        anonymizer = Anonymizer(known_entities={"CLIENT": ["ABC", "ABC Consulting Group"]})
        anonymized, _ = anonymizer.anonymize("ABC Consulting Group a été audité.")
        assert "ABC Consulting Group" not in anonymized
        assert "Consulting Group" not in anonymized


class TestMultipleEntityTypesAndRepetition:
    def test_multiple_entity_types_in_one_sentence(self):
        anonymizer = Anonymizer()
        text = "Contact: j.dupont@abc.fr, IP 192.168.1.20, serveur SRV-01, application SAP."
        anonymized, mapping = anonymizer.anonymize(text)

        assert not any(v in anonymized for v in ("j.dupont@abc.fr", "192.168.1.20", "SRV-01", "SAP"))
        assert {"j.dupont@abc.fr", "192.168.1.20", "SRV-01", "SAP"} == set(mapping.values())
        assert anonymizer.deanonymize(anonymized, mapping) == text

    def test_repeated_entity_within_one_call_reuses_placeholder(self):
        anonymizer = Anonymizer()
        text = "Le serveur SRV-01 puis à nouveau le serveur SRV-01 est concerné."
        anonymized, mapping = anonymizer.anonymize(text)

        assert anonymized.count("SERVER_001") == 2
        assert mapping == {"SERVER_001": "SRV-01"}

    def test_repeated_entity_across_separate_calls_reuses_placeholder(self):
        """Same audit context (same Anonymizer instance) -> consistent pseudonyms across findings."""
        anonymizer = Anonymizer()
        first, _ = anonymizer.anonymize("Le serveur SRV-PROD-01 n'est pas à jour.")
        second, _ = anonymizer.anonymize("Le serveur SRV-PROD-01 accepte encore TLS 1.0.")

        assert "SERVER_001" in first and "SERVER_001" in second
        assert anonymizer.mapping == {"SERVER_001": "SRV-PROD-01"}

    def test_different_entities_get_increasing_placeholder_numbers(self):
        anonymizer = Anonymizer()
        anonymizer.anonymize("Le serveur SRV-A n'est pas conforme.")
        anonymizer.anonymize("Le serveur SRV-B n'est pas conforme.")
        assert anonymizer.mapping == {"SERVER_001": "SRV-A", "SERVER_002": "SRV-B"}

    def test_a_new_instance_starts_a_fresh_context(self):
        first_context = Anonymizer()
        first_context.anonymize("Le serveur SRV-A n'est pas conforme.")
        second_context = Anonymizer()
        anonymized, mapping = second_context.anonymize("Le serveur SRV-A n'est pas conforme.")
        assert mapping == {"SERVER_001": "SRV-A"}  # counting restarts, unaffected by the other context


class TestIsoTerminologyIsPreserved:
    @pytest.mark.parametrize("text", ISO_SENTENCES)
    def test_iso_and_security_terminology_survives_untouched(self, text):
        anonymizer = Anonymizer()
        anonymized, mapping = anonymizer.anonymize(text)
        assert anonymized == text
        assert mapping == {}

    def test_iso_terminology_around_a_real_entity_is_preserved(self):
        anonymizer = Anonymizer()
        text = "Sur le serveur SRV-01, le contrôle A.8.8 (ISO/IEC 27001) n'est pas appliqué (MFA absente)."
        anonymized, mapping = anonymizer.anonymize(text)

        assert set(mapping.values()) == {"SRV-01"}
        for term in ("A.8.8", "ISO/IEC 27001", "MFA"):
            assert term in anonymized


class TestNoUnnecessaryAnonymization:
    @pytest.mark.parametrize("text", [
        "Le serveur n'est pas correctement protégé.",       # keyword with no value, just a verb
        "Le compte est verrouillé après trois tentatives.",  # "est" is a stopword, not a value
        "L'application de gestion des tickets ne dispose pas de workflow.",  # "de" is a stopword
        "Les comptes utilisateurs ne sont pas revus.",        # plural: no keyword+value match
        "Aucune information sensible ici.",
    ])
    def test_sentence_without_a_real_value_is_left_untouched(self, text):
        anonymizer = Anonymizer()
        anonymized, mapping = anonymizer.anonymize(text)
        assert anonymized == text
        assert mapping == {}

    def test_deanonymize_is_idempotent_on_clean_text(self):
        anonymizer = Anonymizer()
        text = "Aucune information sensible ici."
        anonymized, mapping = anonymizer.anonymize(text)
        assert anonymized == text
        assert anonymizer.deanonymize(anonymized, mapping) == text


class TestPersonRule:
    def test_full_name_is_captured_not_just_the_first_word(self):
        anonymizer = Anonymizer()
        anonymized, mapping = anonymizer.anonymize("Responsable : Marie Dupont, contact : Jean-Paul Martin.")
        assert "Marie" not in anonymized and "Dupont" not in anonymized
        assert mapping == {"PERSON_001": "Marie Dupont", "PERSON_002": "Jean-Paul Martin"}

    def test_stops_before_an_unrelated_following_clause(self):
        anonymizer = Anonymizer()
        anonymized, mapping = anonymizer.anonymize("Responsable : Marie Dupont, contact secondaire non défini.")
        assert mapping == {"PERSON_001": "Marie Dupont"}

    def test_keyword_still_matched_case_insensitively(self):
        anonymizer = Anonymizer()
        _, mapping = anonymizer.anonymize("RESPONSABLE : Alice Bernard.")
        assert mapping == {"PERSON_001": "Alice Bernard"}

    def test_round_trip(self):
        anonymizer = Anonymizer()
        text = "Responsable : Marie Dupont, contact : Jean-Paul Martin."
        anonymized, mapping = anonymizer.anonymize(text)
        assert anonymizer.deanonymize(anonymized, mapping) == text


class TestPlaceholderNesting:
    """A later rule's value pattern (esp. the multi-word _PERSON_VALUE) must never swallow a
    placeholder an earlier rule already inserted -- that would nest one placeholder inside
    another's mapped value, and since deanonymize() is a flat placeholder->original substitution,
    the inner one would never resolve, leaking a placeholder token (not a real value, but still a
    correctness bug) into the final deanonymized text."""

    def test_a_placeholder_embedded_in_a_later_multi_word_match_is_not_rewrapped(self):
        anonymizer = Anonymizer()
        text = "Nom : Compte admin, Responsable : Marie Dupont, Serveur : SRV-PROD-01"
        anonymized, _ = anonymizer.anonymize(text)
        assert anonymizer.deanonymize(anonymized) == text
        # every placeholder actually used stays fully resolvable -- none references another
        for value in anonymizer.mapping.values():
            assert not any(PLACEHOLDER_RE.match(word) for word in value.split())


class TestRestore:
    def test_round_trip_resumes_placeholder_assignment(self):
        original = Anonymizer()
        original.anonymize("Le serveur SRV-A n'est pas conforme.")

        resumed = Anonymizer.restore(original.mapping)
        anonymized, _ = resumed.anonymize("Le serveur SRV-A puis le serveur SRV-B.")

        assert "SERVER_001" in anonymized  # SRV-A reuses its existing placeholder
        assert resumed.mapping["SERVER_002"] == "SRV-B"  # SRV-B continues the count, not _001 again

    def test_restore_with_no_prior_mapping_behaves_like_a_fresh_instance(self):
        resumed = Anonymizer.restore({})
        anonymized, mapping = resumed.anonymize("Le serveur SRV-A n'est pas conforme.")
        assert mapping == {"SERVER_001": "SRV-A"}

    def test_restore_accepts_known_entities(self):
        resumed = Anonymizer.restore({}, known_entities={"CLIENT": ["Northwind Traders"]})
        anonymized, _ = resumed.anonymize("Northwind Traders a été audité.")
        assert "Northwind Traders" not in anonymized


class TestDeanonymization:
    def test_deanonymize_without_explicit_mapping_uses_the_context_mapping(self):
        anonymizer = Anonymizer()
        anonymized, _ = anonymizer.anonymize("Le serveur SRV-01 et le compte j.dupont sont concernés.")
        assert anonymizer.deanonymize(anonymized) == "Le serveur SRV-01 et le compte j.dupont sont concernés."

    def test_round_trip_over_several_findings_in_the_same_context(self):
        anonymizer = Anonymizer()
        findings = [
            "Le serveur SRV-PROD-01 de la société ABC utilise le compte admin.",
            "Le serveur SRV-PROD-01 accepte encore des connexions TLS 1.0.",
            "L'application SAP n'a pas été mise à jour depuis 18 mois.",
        ]
        for finding in findings:
            anonymized, _ = anonymizer.anonymize(finding)
            assert anonymizer.deanonymize(anonymized) == finding
