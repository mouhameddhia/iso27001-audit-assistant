"""Reversible anonymization of auditor-written text (French or English).

Detects sensitive values with high-confidence, keyword- or pattern-triggered rules and replaces
them with neutral placeholders (CLIENT_001, SERVER_001, ...) before the text reaches any AI
component (embedding, lexical index, or later an LLM). A mapping is kept to restore the original
values afterwards (deanonymization) once a final answer has been generated.

Design choice, carried over from the original prototype: only high-confidence patterns are
anonymized (an explicit keyword such as "serveur"/"compte", or an unambiguous format such as an
IP or an email). Free-form NER-style guessing is deliberately avoided, because ISO/security text
is full of acronyms that look like identifiers (AES-256, TLS 1.3, ISO/IEC 27001, A.5.18, RGPD,
MFA, RBAC...) and must reach retrieval and generation untouched.

Consistency across an audit context: an Anonymizer instance keeps its placeholder assignments
for its whole lifetime, so calling `anonymize` several times (once per auditor finding, over one
engagement) maps the same original value to the same placeholder every time. Create one instance
per audit context; a new instance starts a fresh, unrelated set of placeholders.
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Pattern, Tuple


@dataclass
class Rule:
    """One detection rule: a compiled regex + the placeholder prefix to use.

    If the regex has a capture group, only the captured group is replaced (this keeps a keyword
    like "serveur"/"société" in the text and hides only the value that follows it).
    """
    name: str
    pattern: Pattern
    prefix: str


# Placeholder format: PREFIX_001, PREFIX_002, ...
PLACEHOLDER_RE = re.compile(r"^[A-Z]+_\d{3}$")

# A value: word characters and hyphens, with dots allowed only between them (so "j.dupont" and
# "SRV-PROD-01" match whole, but a sentence-final period is never swallowed into the value).
_VALUE = r"([\w-]+(?:\.[\w-]+)*)"
_MIN_VALUE_LENGTH = 2  # blocks a single stray letter split off by an apostrophe, e.g. "n" from "n'est"

# A person's name: 1-3 capitalised words ("Marie Dupont", "Jean-Paul Martin"), unlike a server or
# account name, which is normally one token. Each extra word needs its own leading capital, so the
# match stops before an ordinary lowercase word or punctuation ("Dupont, contact" stops at "Dupont").
_PERSON_VALUE = r"([A-ZÀ-Ý][\w-]*(?:\s+[A-ZÀ-Ý][\w-]*){0,2})"

# Ordinary function words that a keyword can be immediately followed by in a normal sentence
# ("le compte est verrouillé", "l'application de gestion") without introducing an actual value.
# A closed grammatical class, not tied to these documents' content.
_STOPWORDS = frozenset("""
    est sont était étaient une un les des ces cette cet cela
    qui que quoi dans sur avec pour sans sous chez vers depuis par
    doit doivent peut peuvent ne pas plus non oui et ou mais donc or ni car
    la le de du au aux ce se sa son ses leur leurs notre nos votre vos
    the is are was were a an of to in on for with by from as at
""".split())


def _keyword_rule(name: str, prefix: str, keywords: str, value: str = _VALUE) -> Rule:
    # The keyword is matched case-insensitively via a *scoped* inline flag, `(?i:...)`, so the
    # value pattern's own case sensitivity (needed for _PERSON_VALUE's capitalisation check) is
    # never affected -- a plain `re.IGNORECASE` on the whole pattern would silently defeat it.
    pattern = re.compile(rf"\b(?i:{keywords})\s*[:\s]\s*{value}")
    return Rule(name=name, pattern=pattern, prefix=prefix)


def default_rules() -> List[Rule]:
    return [
        # Unambiguous formats: safe to match anywhere in the text.
        Rule("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "EMAIL"),
        Rule("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "IP"),
        # Everything else only fires right after an explicit keyword, so ISO/security
        # terminology elsewhere in the sentence is never touched.
        _keyword_rule("SERVER", "SERVER", r"serveur|server|h[oô]te|host"),
        _keyword_rule("USERNAME", "USER", r"compte\s+utilisateur|utilisateur|compte|username|login"),
        _keyword_rule("APPLICATION", "APP", r"application|logiciel|progiciel"),
        # "client" is how an auditor most often names the audited organisation ("le client ACME
        # Corp"), and a company name is commonly several capitalised words, so this rule reads a
        # multi-word value like PERSON rather than the single token the other keyword rules take.
        _keyword_rule(
            "CLIENT", "CLIENT", r"soci[ée]t[ée]|entreprise|compagnie|company|clients?",
            value=_PERSON_VALUE,
        ),
        # Generic person-identifying keywords: useful for any auditor text, not just imported
        # documents, but especially needed there (e.g. a spreadsheet "Responsable" column).
        _keyword_rule("PERSON", "PERSON", r"responsable|contact|propri[ée]taire|owner|nom", value=_PERSON_VALUE),
        Rule("ADDRESS", re.compile(
            r"\b\d{1,4}\s+(?:rue|avenue|boulevard|street|road|ave|blvd|rd|st)\b[^,.\n]*", re.IGNORECASE,
        ), "ADDRESS"),
    ]


class Anonymizer:
    """Reversible anonymizer. One instance = one audit context (consistent placeholders)."""

    def __init__(self, known_entities: Dict[str, List[str]] | None = None, rules: List[Rule] | None = None):
        """
        Args:
            known_entities: optional category -> list of known sensitive strings for this
                engagement (e.g. {"CLIENT": ["ABC Consulting"]}). Matched first, as exact
                strings, before the generic rules run: the only safe way to hide a name that
                doesn't follow a recognisable keyword or format.
            rules: override the default detection rules (mainly for testing).
        """
        self.known_entities = known_entities or {}
        self.rules = rules if rules is not None else default_rules()
        self.mapping: Dict[str, str] = {}          # placeholder -> original
        self._reverse: Dict[str, str] = {}          # original -> placeholder (this run's consistency)
        self._counters: Dict[str, int] = {}

    @classmethod
    def restore(
        cls, mapping: Dict[str, str], known_entities: Dict[str, List[str]] | None = None,
        rules: List[Rule] | None = None,
    ) -> "Anonymizer":
        """Resumes an audit context from a previously saved `mapping` (e.g. persisted in an
        `AuditSession`), so pseudonym numbering continues instead of restarting: the next new
        value for a prefix that already has entries gets the next number, not `_001` again."""
        anonymizer = cls(known_entities=known_entities, rules=rules)
        anonymizer.mapping = dict(mapping)
        anonymizer._reverse = {original: placeholder for placeholder, original in mapping.items()}
        for placeholder in mapping:
            prefix, _, suffix = placeholder.rpartition("_")
            if prefix and suffix.isdigit():
                anonymizer._counters[prefix] = max(anonymizer._counters.get(prefix, 0), int(suffix))
        return anonymizer

    def _placeholder_for(self, prefix: str, value: str) -> str:
        if value in self._reverse:
            return self._reverse[value]
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        placeholder = f"{prefix}_{self._counters[prefix]:03d}"
        self.mapping[placeholder] = value
        self._reverse[value] = placeholder
        return placeholder

    def anonymize(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Replace sensitive values in `text` with placeholders.

        Returns (anonymized_text, mapping_for_this_text): `mapping_for_this_text` has only the
        placeholders that appear in the returned text, ready to pass to `deanonymize`. The full,
        cumulative mapping for the audit context is available on `self.mapping`.
        """
        used: Dict[str, str] = {}

        # Step 1: known entities (exact string match), longest first so "ABC Consulting Group"
        # is matched whole before "ABC Consulting".
        for category, names in self.known_entities.items():
            for name in sorted(names, key=len, reverse=True):
                if name and name in text:
                    placeholder = self._placeholder_for(category, name)
                    used[placeholder] = name
                    text = text.replace(name, placeholder)

        # Step 2: generic rules (emails, IPs, servers, users, applications, clients, addresses).
        for rule in self.rules:
            text = self._apply_rule(text, rule, used)

        return text, used

    def _apply_rule(self, text: str, rule: Rule, used: Dict[str, str]) -> str:
        def replace_match(match: re.Match) -> str:
            has_group = bool(match.groups())
            value = match.group(1) if has_group else match.group(0)

            # Don't re-anonymize a placeholder already inserted (e.g. by `known_entities`,
            # or "serveur SERVER_001" reappearing in a later finding of the same context). A
            # multi-word value (_PERSON_VALUE) can also *contain* one as just part of a longer
            # capture ("Compte USER_001") rather than being one outright -- checked word by word,
            # since wrapping it in a second placeholder would nest one inside another and leave
            # the inner one unresolved (leaking a placeholder token, not a real value, into the
            # final deanonymized text -- not a privacy leak, but a real correctness bug).
            if any(PLACEHOLDER_RE.match(word) for word in value.split()):
                return match.group(0)
            # An ordinary word right after the keyword ("le compte est verrouillé") is not a value,
            # and neither is a single letter split off by an apostrophe ("n" from "serveur n'est").
            if has_group and (len(value) < _MIN_VALUE_LENGTH or value.lower() in _STOPWORDS):
                return match.group(0)

            placeholder = self._placeholder_for(rule.prefix, value)
            used[placeholder] = value

            if has_group:
                # Keep the surrounding keyword, replace only the captured value.
                start, end = match.span(1)
                m_start, _ = match.span(0)
                return match.group(0)[:start - m_start] + placeholder + match.group(0)[end - m_start:]
            return placeholder

        return rule.pattern.sub(replace_match, text)

    def deanonymize(self, text: str, mapping: Dict[str, str] | None = None) -> str:
        """Restore original values in `text`. Defaults to the context's full cumulative mapping."""
        for placeholder, original in (mapping if mapping is not None else self.mapping).items():
            text = text.replace(placeholder, original)
        return text
