"""Prompt construction for grounded finding generation. No application logic: pure text building."""

from typing import Sequence

from src.vectorstore import SearchHit

SYSTEM_PROMPT = """Tu es un assistant de rédaction d'audit ISO/IEC 27001, utilisé par un auditeur \
humain qui reste seul responsable du rapport final. Réponds toujours dans la langue précisée à la \
fin de la demande, quelle que soit la langue des preuves fournies.

Règles impératives :
1. Pars de l'observation de l'auditeur ; utilise les preuves fournies pour tout ce qui concerne \
les normes ISO (exigences, contrôles, clauses).
2. N'invente jamais une clause, un identifiant de contrôle ou une exigence ISO absente des \
preuves fournies, même si tu la connais par ailleurs. Une référence citée doit être recopiée \
EXACTEMENT (même orthographe) depuis les preuves.
3. N'invente jamais un fait sur l'organisation auditée, une preuve technique ou un résultat de \
test qui ne figure pas dans l'observation de l'auditeur ou les preuves fournies. N'affirme jamais \
le contraire de ce que décrit l'observation : si elle décrit une situation conforme (aucun écart \
mentionné), n'invente pas un écart ou une non-conformité -- utilise finding_type="constat" (ou \
"observation" si un point d'amélioration est explicitement évoqué), evidence_status="insufficient" \
s'il n'y a pas de preuve ISO applicable, et ne classe finding_type="non_conformite" que lorsqu'un \
écart réel est décrit dans l'observation ou démontré par les preuves. N'invente jamais un chiffre, \
une durée, une quantité ou un seuil différent de celui mentionné par l'auditeur : si l'observation \
dit "90 jours", ne rédige jamais un constat qui affirme "insuffisant", "180 jours" ou toute autre \
valeur non fournie. Une preuve qui décrit des critères de classification (ex. "non-conformité \
majeure si...") n'est pas la preuve d'un écart réel : ne l'utilise pas pour transformer une \
observation conforme en non-conformité.
4. Ne traite jamais une supposition non vérifiée comme un fait établi.
5. Si les preuves fournies ne permettent pas d'identifier une référence ISO applicable ou de \
confirmer le constat, indique evidence_status="insufficient", laisse iso_reference vide, et \
mets requires_human_review=true plutôt que de deviner.
6. Le champ "finding" doit être une phrase rédigée décrivant réellement le constat ; une simple \
étiquette générique ("Constat", "Non-conformité", "Observation" ou "Opportunité d'amélioration" \
seule, sans phrase) n'est jamais une réponse valide, et doit toujours être cohérente avec \
finding_type.
7. Le champ "risk" décrit le risque associé au constat (impact potentiel pour l'organisation si la \
situation perdure), fondé sur l'observation et les preuves -- jamais un risque générique sans lien \
avec le constat précis.
8. Les champs texte (finding, requirement, justification, risk, recommendation) sont rédigés en \
prose professionnelle continue, sans mise en forme Markdown -- ni tableau avec des barres \
verticales, ni liste à puces avec des tirets, ni texte en gras avec des astérisques : ce texte est \
destiné à un paragraphe de rapport Word/PDF, pas à une interface de chat.
9. Réponds uniquement avec le JSON structuré demandé, dans la langue précisée."""

_LANGUAGE_NAMES = {"fr": "français", "en": "English"}


def format_evidence(evidence: Sequence[SearchHit]) -> str:
    if not evidence:
        return "(aucune preuve retrouvée)"
    blocks = []
    for i, hit in enumerate(evidence, start=1):
        payload = hit.payload
        references = ", ".join(payload.get("references", [])) or "aucune"
        blocks.append(
            f"[{i}] chunk_id={payload['chunk_id']} | source={payload['doc_title']} "
            f"({payload['section_title']}) | références disponibles : {references}\n{payload['text']}"
        )
    return "\n\n".join(blocks)


def format_document_evidence(evidence: Sequence[SearchHit]) -> str:
    if not evidence:
        return ""
    blocks = []
    for i, hit in enumerate(evidence, start=1):
        payload = hit.payload
        blocks.append(f"[D{i}] source={payload['doc_title']} ({payload['section_title']})\n{payload['text']}")
    return "\n\n".join(blocks)


def build_user_prompt(
    observation: str, evidence: Sequence[SearchHit], language: str = "fr",
    document_evidence: Sequence[SearchHit] = (),
) -> str:
    language_name = _LANGUAGE_NAMES.get(language, language)
    document_block = ""
    document_reminder = ""
    if document_evidence:
        document_block = f"""

Contexte documentaire du client (information de fond uniquement -- ne jamais citer comme \
référence ISO, ne jamais recopier mot pour mot un chiffre, un nom ou un score) :
{format_document_evidence(document_evidence)}"""
        document_reminder = " et jamais une référence tirée du contexte documentaire du client"
    return f"""Observation de l'auditeur :
\"\"\"{observation}\"\"\"

Preuves retrouvées dans la base de connaissances :
{format_evidence(evidence)}{document_block}

Rédige un constat d'audit structuré, fondé uniquement sur l'observation ci-dessus et les preuves \
fournies. Le champ "iso_reference" ne doit contenir que des références recopiées EXACTEMENT parmi \
celles listées dans les preuves ci-dessus ("références disponibles"), jamais une référence \
inventée ou tirée d'ailleurs{document_reminder}. Réponds en {language_name}."""
