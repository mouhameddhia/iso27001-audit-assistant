# ISO/IEC 27001 Audit-Report Assistant — Phase 2/3: Prototype IA + Développement

A local, privacy-preserving assistant that helps ISO/IEC 27001 auditors draft findings,
non-conformities, improvement opportunities and conclusions, grounded in an ISO knowledge base.
Everything runs on-premise (Ollama + Qdrant), as the cahier des charges forbids public AI services.

## Phase 2 architecture

```
Auditor input
  -> Anonymization            client names, IPs, servers, accounts, applications -> CLIENT_XYZ, SERVER_001...
  -> Retrieval                lexical (BM25)  +  semantic (Qdrant)
  -> Hybrid fusion            Reciprocal Rank Fusion
  -> Cross-encoder reranking  reorders the fused candidates
  -> LLM generation           grounded only in the retrieved ISO chunks, with clause references
  -> Grounding validation     every cited reference checked against the evidence, deterministically
  -> Structured audit finding constat / requirement / référence ISO / confidence / sources (JSON schema)
  -> Human review              approve / reject / edit / classify severity -- an AuditSession
  -> Report content            Word (.docx) and PDF, deanonymized, only approved findings

Offline, feeding the retrieval layer:
  data/raw -> parse/clean -> semantic units -> chunks + metadata -> embeddings (Ollama) -> Qdrant
```

| Stage | Scope | Status |
|---|---|---|
| **1. Knowledge ingestion** | parsing, semantic chunking, metadata, references, embeddings, Qdrant, benchmark | **done** |
| **2. Secure hybrid retrieval** | anonymization, BM25, semantic retrieval, RRF fusion, cross-encoder reranking, benchmark | **done** |
| **3. Generation & grounding** | structured LLM generation, deterministic grounding validation, confidence, deanonymized output | **done** |
| **4. Report content** | audit sessions, human review workflow, Word/PDF report generation | **done** |
| **Document import** | client PDF/DOCX/XLSX/CSV as additional retrievable evidence, session-scoped and human-gated | **done** |
| **Development phase (Phase 3)** | FastAPI + PostgreSQL backend, React frontend, auth/RBAC, journalisation | **in progress** — see [below](#development-phase-phase-3) |

`src/generation`'s `ollama_generator.py` and `src/pipeline/rag_pipeline.py` hold the earlier
in-memory prototype's generation step (see [Earlier prototype](#earlier-prototype)); Stage 3
replaces it with `src/generation/{models,prompts,generator,validation}.py` and `src/audit_assistant.py`.

## Stage 1 — Knowledge ingestion

### Where things live

```
src/
├── config.py                 all settings (env vars / .env), no hardcoded paths or endpoints
├── kb.py                     CLI: ingest | stats | search | evaluate
├── ingestion/
│   ├── parser.py             file loaders, normalization, KB template (header + sections), line reflow
│   ├── segmenter.py          sections -> typed semantic units (guidance, example, record, template)
│   ├── references.py         standard-aware extraction of ISO controls/clauses and RGPD articles
│   ├── chunker.py            units -> chunks: sizing rules, heading, metadata
│   ├── models.py             ParsedDocument, Section, Unit, KnowledgeChunk
│   └── pipeline.py           discover -> parse -> chunk -> embed (incremental) -> upsert -> verify
├── embeddings/embedder.py    OllamaEmbedder (batched; model and prefixes configurable)
├── vectorstore/qdrant_store.py   QdrantStore: the only module that imports qdrant_client
└── evaluation/retrieval.py   benchmark loading, relevance matching, metrics
evaluation/
├── kb_retrieval_benchmark.json   50 queries with chunking-independent relevance judgments
└── llm_context_experiment.py     experiment behind the "no LLM at ingestion" decision
data/raw/                     knowledge-base source documents
tests/                        unit, integration and end-to-end tests
```

### Knowledge base

The six sources required by the cahier des charges (§4.2) are in `data/raw/`:

| File | doc_id | doc_type | standard |
|---|---|---|---|
| iso27001_2022_full_KB.txt | doc-iso27001-2022-full | iso_standard | ISO/IEC 27001:2022 |
| iso27002_2022_full_KB.txt | doc-iso27002-2022-full | iso_standard | ISO/IEC 27002:2022 |
| iso27017_full_KB.txt | doc-iso27017-full | iso_standard | ISO/IEC 27017:2015 |
| iso27018_full_KB.txt | doc-iso27018-full | iso_standard | ISO/IEC 27018:2019 |
| iso27701_full_KB.txt | doc-iso27701-full | iso_standard | ISO/IEC 27701:2019 |
| politique_interne_cabinet_audit_KB.txt | doc-politique-interne-cabinet | internal_policy | ISO/IEC 27001 (subject) |

Each file has a header (`DOCUMENT`, `ID`, `Category`, `Language`) and sections that mirror the audit
report of the cahier des charges (§4.4). Sections 01–03 explain the standard (scope, audit team);
sections 04–08 are mostly **report material**: example findings, non-conformity records, observations,
improvement opportunities and conclusion templates. The internal policy states the firm's writing
and classification rules in the same sections.

To add a document, drop a `.txt`/`.md` file in `data/raw/` and run `python -m src.kb ingest`. Files that
follow the template get full metadata; other files are ingested as one unstructured section. Another
format needs one reader in `parser.LOADERS`.

### Chunking: semantic units

**Why not packing or windows.** A benchmark failure analysis showed the previous chunker (packing blocks
up to 80–140 words) putting several distinct findings in one chunk. For example, the A.8.8 vulnerability
finding was merged with the A.5.18 access-review finding. Their vectors blend two topics: neither ranks
for its own query, and such mixed chunks show up in the top 3 of many unrelated queries.

**Segmentation by shape** (`segmenter.py`, no per-document rules):

| Unit type | Recognised by | Example |
|---|---|---|
| `record` | identifier line + ≥ 2 `Label : value` lines; fields are parsed | `NC-2026-014 (Mineure)` / `Référence :` / `Énoncé :` / `Risque :` |
| `example` | one list item that is a quotation | `- "Référence A.8.8 (…) : le tableau de bord…"` |
| `template` | a quotation with `[placeholders]` | `"… l'équipe d'audit [recommande / ne recommande pas] …"` |
| `guidance` | any other paragraph or list | scope, requirements, criteria, firm rules |

A one-sentence block ending with ":" (`Exemples de non-conformités rédigées selon un format professionnel :`)
is an **introduction**. It is not embedded as a chunk on its own. It becomes the `context` of the units it
introduces, until the next plain paragraph or the next list with its own introduction line. A longer
paragraph ending with ":" is kept with the block it introduces. Line wraps are rejoined, including
compounds and ranges cut at a line end (`multi-`/`tenants`, `A.5.19-`/`A.5.23`).

**Assembly** (`chunker.py`):
1. Chunks never cross a section.
2. Every example, record and template is its own chunk.
3. A guidance unit under `CHUNK_MIN_WORDS` (40) is merged with the adjacent guidance under the same
   introduction, up to `CHUNK_MAX_WORDS` (200). This keeps "Non-conformité MINEURE dans les autres cas…"
   next to the MAJEURE criteria it completes.
4. A unit over the maximum is split between lines (list items, fields), then sentences, then words.
   No unit in the current corpus needs it.
5. No overlap. Each chunk has a heading `document › section › introduction`, embedded with its text,
   e.g. `ISO/IEC 27001:2022 › Non-conformités › Exemples de non-conformités rédigées selon un format professionnel`.

The corpus gives 110 chunks: 60 guidance, 35 examples, 10 records, 5 templates, 27–176 words
(median 60). 54 of them cite at least one reference.

### Standard-aware references

`references.py` turns every citation into a canonical, standard-qualified identifier. The documents cite
controls in many forms, and the same syntax can belong to different standards:

| In the text | Stored as |
|---|---|
| `A.5.18`, `Annexe A.5.18`, `(A.5.9 du SMSI)` | `ISO/IEC 27001 A.5.18` |
| `27002 §5.15`, `contrôle 8.24` (in ISO 27002) | `ISO/IEC 27002 5.15`, `ISO/IEC 27002 8.24` |
| `27018 §A.10.1` | `ISO/IEC 27018 A.10.1` (not 27001 Annex A) |
| `Annexe B, 8.2.2` (in ISO 27701) | `ISO/IEC 27701 B.8.2.2` |
| `CLD.9.5.1/9.5.2` | `ISO/IEC 27017 CLD.9.5.1`, `ISO/IEC 27017 CLD.9.5.2` |
| `A.5.19-A.5.23`, `clauses 4 à 10` | expanded into each control / clause |
| `Clause 10.2 du SMSI`, `Article 33 du RGPD` | `ISO/IEC 27001 clause 10.2`, `RGPD art. 33` |
| `(réf. POL-SEC-004, §4.2)` | nothing: a client document, not a standard |

Evidence is used in this order: a standard written next to the reference; a numbering scheme owned by
one standard (`CLD.x` is 27017, a bare `A.x.y` is 27001 Annex A); the document's own standard. Ambiguous
references are dropped rather than guessed. The same function can extract references from auditor
input in Stage 2. The previous regex stored `A.10.1` of ISO 27018 as if it were an ISO 27001 control,
and missed the `§`, `CLD`, `Annexe B` and RGPD forms.

### Payload stored with each vector

| Field | Example |
|---|---|
| chunk_id | `doc-iso27001-2022-full:s05:c01` (point id = UUIDv5 of it) |
| doc_id, doc_title, source_file | `doc-iso27001-2022-full`, `ISO/IEC 27001:2022 - …`, `iso27001_2022_full_KB.txt` |
| doc_type, standard, standard_version | `iso_standard`, `ISO/IEC 27001`, `2022` |
| category, category_label, language | `iso27001`, `ISO 27001`, `fr` |
| section_number, section_title, section_slug | `5`, `NON-CONFORMITÉS`, `non_conformites` |
| chunk_index, chunk_count | position inside the section |
| unit_type, context, heading | `record`, `Exemples de non-conformités …`, `ISO/IEC 27001:2022 › Non-conformités › …` |
| text, word_count | chunk text (for grounding and citation) |
| references | `["ISO/IEC 27001 A.5.18", "ISO/IEC 27001 clause 8.1"]` |
| record_id, record_label, fields | `NC-2026-014`, `Mineure`, `{"Référence": …, "Énoncé": …, "Preuve": …, "Risque": …}` |
| embedding_hash | SHA-256 of embedder fingerprint + embedded text (incremental ingestion) |

Keyword indexes: `doc_id`, `doc_type`, `standard`, `category`, `language`, `section_slug`, `unit_type`,
`record_label`, `references`. Filters take a value or a list (match any).

The record fields map directly onto the output required by the cahier des charges (§4.1: constat,
risque associé, référence ISO, non-conformité rédigée). Stage 4 can use them as structured few-shot
examples: `unit_type=record`, optionally `record_label=Majeure`.

### Evaluation

`evaluation/kb_retrieval_benchmark.json` has 50 queries in 6 categories: auditor findings (19),
normative questions (10), firm policy (9), drafting (4), explicit references (3) and English queries on
the French corpus (5). Relevance is given as **anchors** (a document and a phrase from it): a retrieved
chunk is relevant if it belongs to that document and contains the phrase. The same benchmark therefore
scores any chunking strategy or model. Queries paraphrase the documents in auditor language. They were
written before the experiments and not changed afterwards. A test checks that every anchor still exists
in `data/raw`.

Metrics: hit@1, hit@5, MRR@10, recall@5 (share of anchors in the top 5), and recall@500w (share of
anchors in the ranked chunks that fit in 500 words of context). The last one compares chunk sizes fairly,
since a larger chunk trivially contains more text.

**Chunking × embedding model** (MRR@10, same benchmark):

| Chunking | nomic-embed-text | bge-m3 |
|---|---|---|
| Phase 1 prototype: 220-word windows, 40 overlap | 0.50 | 0.69 |
| Previous Stage 1: section packing 80/140 words | 0.60 | 0.85 |
| **Semantic units 40/200 (current)** | 0.67 | **0.90** |

| Configuration | hit@1 | hit@5 | MRR@10 | recall@5 | recall@500w |
|---|---|---|---|---|---|
| section packing + nomic-embed-text (previous) | 0.50 | 0.80 | 0.60 | 0.66 | 0.66 |
| section packing + bge-m3 | 0.78 | 0.96 | 0.85 | 0.87 | 0.87 |
| **semantic units + bge-m3 (current)** | **0.84** | **1.00** | **0.90** | **0.93** | **0.94** |

- **Chunking** gains with both models. With bge-m3, reference queries go from MRR 0.54 to 1.00 and
  recall@500w from 0.87 to 0.94.
- **Embedding model**: `bge-m3` (multilingual, 1024 dimensions) is much stronger than `nomic-embed-text`
  on this French corpus. English queries reach MRR 1.00.
- **Sizes are not fragile**: with bge-m3, MIN/MAX of 0/200, 40/200, 80/200 and 40/120 give MRR 0.90,
  0.90, 0.89 and 0.88. With nomic, 40 was clearly better than no merging (0.67 vs 0.58).
- Per category (current): finding 0.97, English 1.00, reference 1.00, policy 0.94, drafting 0.80,
  normative 0.70.

The earlier chunkers were replaced, so their rows were measured before the change (the old section
packer was re-run from a verbatim copy to fill the bge-m3 column). Reproduce the current rows with
`python -m src.kb evaluate --in-memory`, overriding `EMBEDDING_MODEL`, `CHUNK_MIN_WORDS` or
`CHUNK_MAX_WORDS` through environment variables.

**LLM-assisted chunk context: measured, not adopted.** `evaluation/llm_context_experiment.py` has
`qwen2.5:7b` read each document and write one or two sentences situating every chunk ("contextual
retrieval"). That sentence is then embedded with the chunk:

| Embedded text (bge-m3) | hit@1 | hit@5 | MRR@10 | recall@5 | recall@500w |
|---|---|---|---|---|---|
| heading + text (current) | 0.84 | **1.00** | 0.90 | **0.93** | **0.94** |
| LLM context + heading + text | 0.84 | 0.96 | 0.90 | 0.89 | 0.91 |
| LLM context + text | **0.90** | 0.98 | **0.93** | 0.91 | 0.92 |

The best LLM variant ranks 3 more queries first but finds less of the relevant evidence. It also costs
one 7B-model call per chunk (11 minutes for 110 chunks on an RTX 2060) and makes ingestion depend on a
generative model. This corpus has explicit structure, so the deterministic heading already gives the
context. Stage 4 adds an LLM reranker, which reorders first-stage candidates: that stage needs recall,
not top-1 precision. The generated contexts are cached in `evaluation/llm_contexts.json`.

**Known limits, input for Stage 2**
- The benchmark is small (50 queries) and written by someone who read the corpus: differences under
  about 0.03 MRR are noise.
- Normative questions (MRR 0.70) always have their evidence in the top 5, but conclusion templates,
  which are generic, often rank first. Filtering or routing on `unit_type` by task should fix this.
- Queries that are exact identifiers (`CLD.8.1.5`) find the right chunks but not all of them (recall@5
  0.56): a job for BM25 and the `references` filter.

### Incremental, verified ingestion

- The same chunk always gets the same point id, so re-ingesting overwrites instead of duplicating.
- Only chunks whose `embedding_hash` changed are embedded: a new or edited document costs only its own
  chunks. A different model or document prefix changes the hash and re-embeds everything.
- Everything is embedded before the first write, so an Ollama failure leaves the collection untouched.
- Chunks removed from a document and documents removed from the folder are deleted (`--no-prune` keeps
  the latter).
- Point counts are checked per document and for the whole collection. A collection whose vector size
  doesn't match the model is rejected (use `--recreate`).

On the current corpus: first ingestion 29 s (110 chunks embedded), second 0.5 s (0 embedded).

### Setup

```bash
pip install -r requirements.txt
cp .env.example .env            # optional: only to change defaults

ollama pull bge-m3               # embedding model
docker compose up -d qdrant      # Qdrant on http://localhost:6333 (or use an existing server via QDRANT_URL)
```

Changing `EMBEDDING_MODEL` changes the vectors: run `python -m src.kb ingest --recreate` afterwards.
`nomic-embed-text` also needs `EMBEDDING_DOCUMENT_PREFIX="search_document: "` and
`EMBEDDING_QUERY_PREFIX="search_query: "`.

### Commands

```bash
python -m src.kb ingest                 # synchronise data/raw into QDRANT_COLLECTION
python -m src.kb ingest --recreate      # drop and rebuild
python -m src.kb stats                  # points, vector config, chunks per document / unit type / section
python -m src.kb search "Le contrôle des accès privilégiés n'est pas revu périodiquement."
python -m src.kb search "vulnérabilités non corrigées" --unit-type record
python -m src.kb search "écart constaté" --reference "ISO/IEC 27017 CLD.8.1.5"
python -m src.kb evaluate               # benchmark on the ingested collection
python -m src.kb evaluate --in-memory --details   # benchmark current settings without touching the server
```

### Tests

```bash
python -m pytest                        # everything; live tests skip if Ollama/Qdrant are unreachable
python -m pytest -m "not integration and not e2e"   # offline only (< 2 s)
python -m pytest -m e2e                 # real ingestion + quality gate on the live stack
```

Unit tests use small synthetic documents for behaviour. Checks on `data/raw` are generic invariants
(unique ids, lossless segmentation and chunking, size bounds, no template or line-wrap artefacts), so
adding a document doesn't break them.

| File | Level | What it checks |
|---|---|---|
| test_kb_parser.py | unit | header and sections, wrapped titles, reflow and hyphenated wraps, normalization, doc type and standard, fallback |
| test_segmenter.py | unit | block classification, record fields, introduction scope, lead-ins, lossless segmentation of the corpus |
| test_references.py | unit | 20 citation cases taken from the corpus, qualification rules, ranges, client documents, ambiguity |
| test_kb_chunker.py | unit | exemplar isolation, guidance merging, splitting order, metadata and heading, corpus bounds and losslessness |
| test_embedder.py | unit + integration | batching, prefixes, fingerprint, errors (mocked HTTP); live similarity |
| test_qdrant_store.py | unit + integration | collection lifecycle and validation, payloads and vectors, filters (value / any / list field), facets, deletions; in-process engine and server |
| test_ingestion_pipeline.py | integration (offline) | whole pipeline on the corpus, filters, incremental embedding, model change, sync, failures, dimension change |
| test_retrieval_benchmark.py | unit | benchmark file matches the corpus, relevance matching, budget recall, metrics |
| test_ingestion_e2e.py | end-to-end | real Ollama + Qdrant server: counts, full payload equality, quality gate (hit@5 ≥ 0.90, MRR@10 ≥ 0.80), filters, no re-embedding on re-run |

## Stage 2 — Secure hybrid retrieval

```
Auditor finding -> Anonymization -> BM25 ┐
                                  Semantic ┴-> RRF fusion -> Cross-encoder -> context (top-k)
```

Anonymizes the auditor's finding, retrieves with BM25 (exact terminology) and Stage 1's semantic
search in parallel, fuses the two ranked lists, then reranks the fused candidates with a
cross-encoder. Produces ranked context chunks for a future generation stage — no generation, no
report text, no deanonymization of a generated answer.

### Where things live

```
src/
├── anonymization/anonymizer.py  Anonymizer: keyword/pattern rules, consistent per audit context
└── retrieval/
    ├── types.py        the `Retriever` protocol (`.retrieve(query, top_k) -> List[SearchHit]`)
    ├── semantic.py      SemanticSearchRetriever: thin adapter over Stage 1's QdrantStore
    ├── bm25.py          BM25Retriever: independent lexical index, identifier-aware tokenizer
    ├── fusion.py        reciprocal_rank_fusion() + HybridRetriever
    ├── reranker.py      CrossEncoderReranker
    └── pipeline.py      SecureRetrievalPipeline (the flow above) + RerankedRetriever (benchmark adapter)
evaluation/kb_retrieval_benchmark.json  same 50 queries as Stage 1, now the Stage 2 regression suite
```

Every retrieval component speaks the same currency: `SearchHit(id, score, payload)`, `id` being
the chunk's Qdrant point id, so semantic, BM25 and fused results compare and deduplicate directly
without an adapter layer. `HybridRetriever`, `RerankedRetriever` and `SemanticSearchRetriever` all
implement the same `Retriever.retrieve(query, top_k)`, so `evaluation.retrieval.run_benchmark` (and
`kb.py evaluate --mode`) scores any of them, or the Stage 1 baseline, identically.

### Anonymization

Reversible, high-confidence rules only — the same design as the project's original prototype
anonymizer, extended per the cahier des charges (§5.5): unambiguous formats (IP, email) are always
matched; everything else (server, account, application, company) only right after an explicit
keyword ("serveur", "compte", "application"/"logiciel", "société"/"entreprise"), so ISO/security
vocabulary is never touched — nothing here depends on which documents happen to be in the
knowledge base.

```
"Le serveur SRV-PROD-01 de la société ABC utilise le compte admin."
  -> "Le serveur SERVER_001 de la société CLIENT_001 utilise le compte USER_001."
```

- **Consistent pseudonyms**: an `Anonymizer` instance keeps its placeholder assignments for its
  whole lifetime, so calling `.anonymize()` again in the same audit context maps `SRV-PROD-01` to
  `SERVER_001` again, instead of minting a new placeholder every call. A new `Anonymizer()` starts
  an unrelated context.
- **No unnecessary anonymization**: a value must be at least 2 characters and not a common function
  word (`est`, `de`, `the`, `for`, …, a small closed grammatical list). Without this, "le serveur
  n'est pas protégé" anonymized "n" (split off `n'est` by the apostrophe), and "le compte est
  verrouillé" anonymized the ordinary word "est" as if it were a username — both fixed.
- ISO terminology (`ISO/IEC 27001`, `A.5.18`, `MFA`, `RBAC`, `RGPD`, clause numbers) is never
  matched by any rule, so it reaches retrieval untouched.

### BM25 (`retrieval/bm25.py`)

An in-memory `rank_bm25.BM25Okapi` index built once over the chunks re-derived from `data/raw`
(parsing is fast and side-effect-free, so this never touches Qdrant or drifts from what's
embedded). The tokenizer is generic — not tied to any document's content — but identifier-aware:
a run of letters/digits joined by dots or colons is kept as one token, so `A.5.18`, `CLD.6.3.1`,
`9.2` and `27001:2022` match as whole identifiers instead of scattering into `a`, `5`, `18`. A
single-character token is dropped: a lone letter is never meaningful and mostly comes from an
apostrophe splitting a contraction (`d'accès` -> `d`, `n'a` -> `a`) — see the English-query finding
below for why that one character mattered.

### Hybrid fusion (`retrieval/fusion.py`)

Reciprocal Rank Fusion: a candidate's fused score is `Σ 1/(k + rank)` over every retriever that
returned it (`k = 60`, the usual IR default). RRF was chosen over combining raw scores because BM25
scores and cosine similarities are not on the same scale — comparing them directly would need
per-query normalisation to mean anything, while rank position already is directly comparable.
`HybridRetriever` asks each retriever for `RETRIEVAL_CANDIDATES_PER_RETRIEVER` (20) candidates and
fuses them; `RETRIEVAL_FUSION_TOP_K` (20) candidates are then handed to the reranker.

### Cross-encoder reranking (`retrieval/reranker.py`)

**Model choice, measured.** The knowledge base and expected auditor input are largely French, so an
English-only cross-encoder (the popular `cross-encoder/ms-marco-*` line) was never a candidate.
Two multilingual rerankers were downloaded and scored on this machine (CPU only — `torch.cuda.is_available()`
is `False` here despite the GPU, so both ran on CPU):

| Model | Params | Load+score, 4 pairs | Separates relevant/irrelevant? |
|---|---|---|---|
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | ~118M | 2.9 s | yes: 7.0, 2.5 vs −5.3, −6.9 |
| `BAAI/bge-reranker-v2-m3` | ~568M | 753.8 s (mostly the one-time download) | yes, more confidently: 0.99, 0.84 vs 0.002, 0.001 |

Both correctly rank relevant chunks above irrelevant ones on a quick check. `bge-reranker-v2-m3`'s
confidence is cleaner, but it scores about 17× slower per pair on this CPU (3.61 s vs 0.21 s for
the same 4 pairs, load time excluded) — at `RETRIEVAL_FUSION_TOP_K=20`, that is the difference
between the full 50-query benchmark's rerank step taking under a minute and one taking 15+ minutes,
on a prototype that should stay fast to iterate on. **Default:
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`** (`RERANK_MODEL`, `RERANK_DEVICE`,
`RERANK_BATCH_SIZE`, `RERANK_TOP_K` are all configurable — a deployment with GPU time to spare can
switch to `bge-reranker-v2-m3` by changing one setting).

The reranker scores `(query, heading + chunk text)` pairs — the same text Stage 1 embeds — and
reorders only the fused candidate set; it never runs over the whole collection.

### Evaluation

Same 50-query benchmark as Stage 1, now comparing three retrieval configurations with
`python -m src.kb evaluate --mode {semantic,hybrid,rerank}`:

| Mode | hit@1 | hit@5 | MRR@10 | recall@5 | recall@500w |
|---|---|---|---|---|---|
| A. semantic (Stage 1 baseline) | 0.84 | 1.00 | 0.90 | 0.93 | 0.94 |
| B. hybrid (BM25 + semantic, RRF) | 0.86 | 0.98 | 0.91 | 0.93 | 0.95 |
| **C. hybrid + cross-encoder rerank** | **0.92** | **1.00** | **0.95** | **0.97** | **0.97** |

Per category (hit@1 / MRR@10), A → B → C:

| Category | A | B | C |
|---|---|---|---|
| finding (19) | 0.95 / 0.97 | 1.00 / 1.00 | 1.00 / 1.00 |
| normative (10) | 0.50 / 0.70 | 0.70 / 0.85 | 0.80 / 0.86 |
| policy (9) | 0.89 / 0.94 | 0.89 / 0.90 | 0.89 / 0.93 |
| drafting (4) | 0.75 / 0.80 | 0.75 / 0.88 | 0.75 / 0.88 |
| reference (3) | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| english (5) | 1.00 / 1.00 | 0.60 / 0.72 | 1.00 / 1.00 |

**Adding a component is not automatically an improvement — measured, not assumed:**

- **BM25 alone clearly helps** the weaknesses Stage 1 flagged: exact-identifier queries (reference:
  recall@5 0.56 → 0.89, later 1.00 with reranking — a query like `CLD.8.1.5` now matches the literal
  control id) and normative questions (MRR 0.70 → 0.85: BM25's exact vocabulary match beats
  semantic similarity when a query names a specific mechanism).
- **But hybrid fusion regressed English queries** (hit@1 1.00 → 0.60) and, more mildly, `policy`
  (hit@5 1.00 → 0.89). Investigating why, query by query, by comparing `BM25Retriever.retrieve()`
  and `SemanticSearchRetriever.retrieve()` on the same query: two distinct causes, both inherent to
  combining a lexical and a semantic retriever, not bugs:
  1. *Noise from an accidental token.* Before the single-character-token fix above, the only
     English word that overlapped the French corpus vocabulary for one query was the letter "a"
     (from "n'a", "l'a"...) — BM25 then ranked purely by which chunks happened to contain that
     letter, unrelated to relevance. Fixed: dropping single-character tokens took that query's
     BM25 hits to zero (correctly — genuinely no shared vocabulary), which fixed it.
  2. *Real but imprecise cognate overlap.* Words identical in French and English — "audit",
     "client", "cloud", "IaaS", "services" — are common across many unrelated chunks. For two
     English queries, BM25 found real (not noisy) matches on these cognates and, through RRF,
     outranked semantic search's correct top-1 answer with a topically-related but wrong chunk.
     This is expected behaviour for rank fusion, not something to special-case away.
  - **The cross-encoder reranker recovers both** (English hit@1/MRR back to 1.00/1.00): unlike
    BM25, it is a genuinely multilingual model that judges relevance directly, regardless of the
    query's language, over the candidate set fusion already assembled.
- **Net effect**: reranking recovers every regression hybrid introduced and improves further
  (finding and reference both reach a perfect 1.00 across every metric; normative rises again,
  though it remains the weakest category — see limits below).

Reproduce: `python -m src.kb evaluate --mode semantic|hybrid|rerank` (add `--in-memory` to bypass
the server, `--details` for the per-query rank).

**Known limits, input for Stage 3**
- `normative` stays the weakest category (MRR 0.86): as noted in the Stage 1 write-up, generic
  conclusion templates can still outrank a specific normative passage. Filtering on
  `unit_type != template` for this query shape is a plausible follow-up, not yet implemented (it
  would need a query classifier, which felt like more machinery than this prototype needs for a
  0.09 MRR gap).
- The benchmark stays small (50 queries); differences under ~0.03 are noise, as in Stage 1.
- Anonymization is rule-based, not a NER model: it only recognises entities introduced by a
  handful of keywords or an unambiguous format. It will miss a company name mentioned without
  "société"/"entreprise" nearby, or a hostname without "serveur"/"host" — the same trade-off the
  original prototype's design made, now covering more entity types and staying consistent across
  a whole audit context.

### Setup

Cross-encoder reranking needs `sentence-transformers` (installed via `requirements.txt`) and a
one-time model download from Hugging Face (`hf_xet` is included for fast downloads). No extra
service to run — the model loads in-process, on CPU by default (`RERANK_DEVICE`).

### Commands

```bash
python -m src.kb search "Le contrôle des accès privilégiés n'est pas revu périodiquement."   # --mode rerank by default
python -m src.kb search "..." --mode semantic          # Stage 1 only
python -m src.kb search "..." --mode hybrid             # + BM25, RRF-fused, no reranking
python -m src.kb search "Chez la société Contoso, le serveur SRV-WEB-07 accepte encore TLS 1.0." --anonymize
python -m src.kb evaluate --mode rerank                 # the full regression benchmark
```

### Tests

```bash
python -m pytest -m "not integration and not e2e"   # offline only: 176 of 193 tests, < 2 s
python -m pytest -m e2e                              # real ingestion + quality gate on the live stack
python -m pytest                                      # everything: 193 tests, live tests skip if unreachable
```

| File | Level | What it checks |
|---|---|---|
| test_anonymizer.py | unit | the cahier des charges example verbatim, every entity type, consistency within/across calls and across instances, ISO-terminology preservation, no-op on ordinary sentences, round-trip deanonymization |
| test_bm25.py | unit + integration | tokenizer (identifiers, single-char tokens, case), exact-id and terminology ranking, empty query/index, real corpus |
| test_fusion.py | unit | RRF math (consensus, list-exclusive candidates, dedup, `top_k`, `k`'s effect), `HybridRetriever` wiring and candidate counts |
| test_reranker.py | unit + integration | scoring and reordering, heading prefixing, `top_k`, batch size, scoring/loading/missing-dependency errors (fake model, no download), live model on real chunks |
| test_retrieval_pipeline.py | integration (offline) + e2e | anonymize → fuse → rerank wiring, ISO terms untouched, context ⊆ candidates, pseudonym reuse across findings; live: the cahier des charges example and a finding with sensitive data, end to end |
| test_retrieval_benchmark.py | unit | `run_benchmark` against a fake retriever (mode-agnostic), relevance matching, budget recall, metrics |

## Stage 3 — Generation & grounding

```
result = AuditAssistant.from_settings().analyze(auditor_input)
```

Turns Stage 2's reranked evidence into a structured, traceable draft finding: LLM generation,
constrained to a typed schema, followed by deterministic validation that checks every claim the
model makes against the evidence it was actually given -- and never assumes any of it is correct
just because the model said so.

### Where things live

```
src/
├── audit_assistant.py         AuditAssistant: orchestrates retrieval -> generation -> validation -> deanonymize
├── assistant_cli.py            CLI: analyze | evaluate-grounding
└── generation/
    ├── models.py                GeneratedFinding (the LLM's schema) / AuditFinding (the validated result)
    ├── prompts.py                 system + user prompt construction
    ├── generator.py                FindingGenerator: evidence -> GeneratedFinding, via Ollama structured output
    └── validation.py                GroundingValidator: GeneratedFinding + evidence -> AuditFinding
evaluation/grounding_eval.json  11 realistic observations for hallucination-resistance evaluation
src/evaluation/generation.py    loads/runs grounding_eval.json, measures what the validator actually caught
```

`FindingGenerator` only turns evidence it is handed into a draft; it never queries Qdrant, BM25 or
the reranker. `GroundingValidator` only checks a `GeneratedFinding` against the evidence it was
generated from; it knows nothing about retrieval or Ollama. `AuditAssistant` is the only module that
calls both, in order, and is the only place deanonymization of the final output happens.

### Model selection, measured

Three chat-capable models are installed locally: `qwen2.5:7b`, `gemma3:4b`, `llama3:8b` (a fourth,
`llama3.2:3b-text`, is a base/completion model, not instruction-tuned, and was not considered). All
three return valid structured output via Ollama's schema-constrained `format` (Ollama 0.34): every
one of the three produced parseable JSON conforming to `GeneratedFinding` on a first test. The real
differentiator was **abstention discipline** -- given evidence that does not support a specific ISO
reference, does the model say so, or does it invent a plausible-sounding one?

Three probe cases, repeated across separate runs:

| Model | Insufficient evidence, unrelated context | Misleading (topically close, wrong control) | **Zero evidence at all** |
|---|---|---|---|
| `qwen2.5:7b` | abstained (3/3) | abstained (3/3) | **fabricated `A.12.1.1`, `A.12.1.2`, `A.12.1.3` (3/3)** -- ISO/IEC 27001:**2013** numbering, not even the 2022 numbering this KB uses |
| `gemma3:4b` | never abstained (0/3): always `evidence_status="supported"` | never abstained (0/3) | never abstained (0/3) |
| `llama3:8b` | abstained (3/3) | abstained (3/3) | abstained (3/3) |

`qwen2.5:7b` is the newer, larger model with native tool-calling support -- the more "obvious"
choice on paper -- and it is precisely the one that reliably hallucinated specific, plausible,
entirely fabricated control ids the moment no evidence was given at all, every single time it was
tried. `gemma3:4b` never once said "insufficient" even when nothing supported the claim, making its
`evidence_status` field useless as a signal. **`llama3:8b` is the configured default**
(`GENERATION_MODEL`, `GENERATION_BASE_URL`, `GENERATION_TEMPERATURE`, `GENERATION_MAX_TOKENS`,
`GENERATION_TIMEOUT` are all configurable) -- not because it is the newest or the biggest, but
because it was the only one of the three that reliably declined to answer when it had nothing to
answer from. The deterministic validator below exists precisely because none of this is trusted to
hold on every call: it is the actual safety net, regardless of which model is configured.

### Structured output

`GeneratedFinding` -- what the LLM's JSON schema allows it to produce -- has **no field for a chunk
id or a confidence number**. This is a structural choice, not a prompting one: the model cannot
invent a supporting source or a confidence score because it is never asked for either.

```python
class GeneratedFinding(BaseModel):
    finding: str
    finding_type: FindingType            # constat | non_conformite | observation | opportunite_amelioration
    requirement: str | None
    iso_reference: list[str]             # must be copied verbatim from the evidence -- checked, not trusted
    justification: str | None
    recommendation: str | None
    evidence_status: EvidenceStatus      # supported | partial | insufficient
    requires_human_review: bool
```

`GroundingValidator` builds the final `AuditFinding` from a `GeneratedFinding` plus the real
evidence: `supporting_sources` (which real chunks back which kept reference) and `confidence` are
computed here, deterministically, and appended -- the model never produces either.

### Grounding strategy: layered, not a longer prompt

```
Good retrieval (Stage 1) -> reranking (Stage 2) -> evidence-limited context (top 5 chunks only)
  -> grounded generation prompt (explicit "copy references verbatim, or say insufficient")
  -> structured output validation (Ollama schema + Pydantic: malformed JSON never reaches the app)
  -> reference validation (every citation checked against the evidence -- see below)
  -> confidence from an observable signal (reranker score, not the model's self-report)
  -> human review flag
```

**Reference validation** (`GroundingValidator.validate`, the core of the safety net):
1. Every `iso_reference` is checked, case/whitespace-insensitively, against the canonical
   references actually present in the supplied evidence's payloads.
2. A bare id the model wrote without its standard prefix (`A.5.18` instead of
   `ISO/IEC 27001 A.5.18`) is still accepted, but **only when it resolves to exactly one** canonical
   reference in the evidence set -- ambiguous across two standards, it is dropped rather than
   guessed, the same principle Stage 1's reference extractor uses.
3. Anything that resolves to neither is dropped, `requires_human_review` is forced on, and
   `evidence_status` is downgraded (`supported` → `partial` if something else was kept, else
   → `insufficient`).
4. **`evidence_status="insufficient"` and a non-empty `iso_reference` never coexist in the output.**
   `llama3:8b` sometimes hedges "insufficient" while still citing a reference that is genuinely
   present in the evidence (measured: see Evaluation) -- a self-contradictory claim from the
   model's own two fields. The validator enforces the invariant by clearing the reference, since
   "insufficient" must mean exactly "no confirmed reference" to be a useful signal to a reader.
5. A reference can be **real** (verbatim, present in the evidence) and still be the **wrong** one
   for this specific claim -- string presence alone cannot catch that; catching it reliably would
   need another judge, which just relocates the trust problem rather than solving it. This is why
   confidence is derived independently (next section) rather than trusted from the citation's mere
   presence, and why `requires_human_review` exists as the final layer regardless.

### Confidence: from the reranker, not the model

The model is never asked for a confidence number. `AuditFinding.confidence` is the mean of
`sigmoid(reranker_score)` over the chunks that back the *kept, validated* references (the
cross-encoder's raw score is an unbounded logit; sigmoid gives a `(0, 1)` relevance-like reading,
an informal but standard treatment of MS MARCO-style cross-encoder scores) -- **zero whenever
nothing was verifiably cited**, regardless of what the model claimed. A model saying "supported"
while citing nothing checkable is not more trustworthy than one honestly saying "insufficient".

A second, independent use of the same signal: when a citation is real but its backing reranker
score is weak (`confidence < LOW_CONFIDENCE_REVIEW_THRESHOLD = 0.35`), `requires_human_review` is
forced on **even if the model itself did not ask for review** -- this is exactly the case that
catches "real reference, wrong control" (see B-01/B-02/C-02 below): the citation stays visible
(it is real, after all), but a human is told to look at it.

### Anonymization boundary

The LLM only ever receives `retrieval_result.anonymized_query` (Stage 2's output) and evidence from
the knowledge base, which never contains client data. Deanonymization happens **after** validation,
in `AuditAssistant.analyze()`, as a local string substitution over the finding's text fields
(`observation`, `finding`, `requirement`, `justification`, `recommendation`) using the per-call
mapping Stage 2 already returns -- no AI involved, and `supporting_sources` (chunk/doc metadata from
the KB) is never touched by it, since it was never anonymized to begin with. Tested explicitly:
original entities are restored on output, ISO/security terminology is never altered by either
direction, and a `GenerationError` still deanonymizes the (unchanged) observation in its safe
fallback result.

### Failure handling

| Failure | Behaviour |
|---|---|
| Empty retrieval (no chunks at all) | `FindingGenerator` short-circuits before calling Ollama: `evidence_status=insufficient`, `requires_human_review=true`, the observation echoed as the finding |
| Ollama unreachable / times out / HTTP error | Wrapped in `GenerationError`; `AuditAssistant` returns a safe fallback `AuditFinding` (`insufficient`, `requires_human_review=true`, `confidence=0.0`, the error recorded in `validation_notes`) -- never a fabricated finding |
| Malformed JSON / schema-invalid output | Pydantic validation fails inside `FindingGenerator`, raised as `GenerationError`, handled the same way |
| Weakly-backed real citation | Not a hard failure: kept, but `requires_human_review` forced on (see Confidence) |
| Off-topic observation (no useful retrieval) | Evidence is still returned (Qdrant/BM25 always return their top-k, however weak) but the model + validator are expected to recognise nothing supports a specific reference; measured under category D below |

### Evaluation

**Retrieval** (Stage 1/2's 50-query benchmark, unchanged, re-run for regression) and **generation**
are measured separately -- retrieval MRR/hit@k says nothing about whether the generated text is
grounded, and neither is treated as a proxy for the other.

**Grounding / hallucination-resistance** (`evaluation/grounding_eval.json`, 11 observations distinct
from both the retrieval benchmark and the Stage 3 e2e tests, run through the *real* retrieval
pipeline -- no hand-picked evidence -- across the five categories from the brief):

```
python -m src.assistant_cli evaluate-grounding --details
```

| Metric | Result |
|---|---|
| Unsupported reference reaching the **final** output (must be 0) | **0 / 11** |
| Weakly-backed citation not flagged for review (must be 0) | **0 / 11** |
| Model attempted an unsupported citation, before validation | 2 / 11 |
| Cited a reference iff the case expected one | 6 / 11 (55%; varied 55-64% across repeated runs -- see below) |

Per category: A (strong evidence, n=3), B (insufficient, n=3), C (misleading, n=2), D (no useful
retrieval, n=2), E (ambiguous/multiple weak controls, n=1). D and E abstained correctly in every
run. **Not reporting "zero hallucinations": three distinct, real behaviours were observed, all
handled safely but worth naming honestly:**

1. **A-01**: the model cited `"ISO/IEC 27001:2022, Annexe A.5.18"` -- copied from descriptive prose
   in the evidence rather than the clean canonical form -- which the validator's exact/bare-id
   matching does not resolve. Dropped, review flagged. A conservative false negative (a real,
   correct reference gets flagged for review instead of auto-accepted) was judged the safe
   direction over loosening the matcher, which would risk false positives elsewhere; not "fixed"
   further, and reported as a limitation.
2. **B-01, B-02, C-02**: the model cited a reference that is **real and present in the evidence**,
   but the evidence backing it scored strongly negative on the reranker (e.g. B-01's `A.8.8`, a
   vulnerability-management control, cited for a log-retention observation) -- a real citation
   applied to the wrong claim. String-presence validation cannot catch this by itself; it is
   exactly what the low-confidence review override exists for, and did catch: all three are
   `requires_human_review=true` with `confidence <= 0.13`.
3. **Run-to-run variance**: repeating the same 11 cases across several runs (temperature 0.1, not
   0) gave 55-64% on the "cited iff expected" rate, entirely from the model's own label
   (`evidence_status`) flipping between runs on borderline cases -- not from anything in the
   deterministic layer, which produced identical safety numbers (0/11, 0/11) every time.

**End-to-end** (`tests/test_generation_e2e.py`, live Qdrant + Ollama + reranker + generation, six
observations on topics named in the brief -- privileged access, password rotation, logging,
asset management, supplier security, incident management, none overlapping the retrieval benchmark
or the grounding eval): every result is valid structured output; every `iso_reference` is backed by
a `supporting_sources` entry; `evidence_status=insufficient` always implies an empty
`iso_reference`; anonymized entities are restored in the observation while ISO terminology
(`A.5.18`, `MFA`, `RBAC`, `ISO/IEC 27001`, `RGPD`) survives both directions; the same server name
across two findings reuses the same pseudonym; an off-domain observation (a broken coffee machine)
never produces a confident citation; pointing the generator at an unreachable port produces the
safe fallback, not a crash or a fabricated finding.

### Commands

```bash
python -m src.assistant_cli analyze "Le contrôle des accès privilégiés n'est pas revu périodiquement."
python -m src.assistant_cli analyze "..." --lang en
python -m src.assistant_cli evaluate-grounding --details
```

### Tests

```bash
python -m pytest -m "not integration and not e2e"   # offline only: 234 of ~270 tests, ~12 s
python -m pytest -m e2e                              # real ingestion + retrieval + generation, live stack
```

| File | Level | What it checks |
|---|---|---|
| test_generation_models.py | unit | schema has no chunk-id/confidence field, enum validation, JSON round-trip, confidence bounds |
| test_prompts.py | unit | evidence formatting (chunk id, source, references, no-evidence case), language selection, forbidden-hallucination wording |
| test_generator.py | unit + integration | empty-evidence short-circuit, request construction (model/temperature/schema), valid/malformed/incomplete JSON, HTTP/timeout errors, live structured output |
| test_validation.py | unit | exact/bare/ambiguous reference matching, all `evidence_status` transitions incl. the insufficient-with-citation invariant, low-confidence review override, `supporting_sources` construction, confidence bounds and monotonicity |
| test_audit_assistant.py | unit + integration | orchestration wiring (no retrieval/generation logic duplicated here), deanonymization on output only, ISO terms untouched, safe fallback on `GenerationError`, live cahier-des-charges example |
| test_generation_e2e.py | end-to-end | 6 realistic observations end to end, structured-output validity, reference traceability, insufficient⟹empty invariant, anonymization boundary, pseudonym reuse, off-domain abstention, unreachable-Ollama failure handling |
| test_grounding_eval.py | end-to-end | the 11-case hallucination-resistance set: both safety properties gated at 0, category D abstention gated, full breakdown reported |

**Known limitations**
- Reference matching is exact or unambiguous-bare-id only (see A-01): a citation embedded in
  descriptive prose is dropped rather than fuzzily matched, trading recall for never risking a
  wrong match.
- String presence cannot prove topical correctness (see B-01/B-02/C-02): the low-confidence review
  override catches this via the reranker score, but does not remove the citation, only flags it.
- `llama3:8b`'s own `evidence_status`/`requires_human_review` self-labels vary run to run at
  temperature 0.1; the deterministic layer's output does not.
- No generation-quality ("does this read well") score is reported: that would need either human
  judgment or another LLM acting as judge, and the latter would just relocate the trust problem
  this whole design exists to avoid. Read the generated text.
- This is a 11-case grounding set and a 6-observation e2e set; genuinely small, like Stage 1/2's
  50-query benchmark.

### Next stage

Stage 4 (not implemented): assemble the structured `AuditFinding`s a human auditor has reviewed
into an actual Word/PDF report (the cahier des charges' §4.4 sections), plus persistence (audit
history, so an engagement's earlier findings and their review status can be retrieved) and, likely,
a light API/UI layer so `AuditAssistant.analyze()` is not only reachable through the CLI.

## Stage 4 — Report generation

```
AuditSession (metadata + AuditFindings, each pending review)
  -> Human review     approve / reject / edit / classify severity (majeure/mineure for NCs)
  -> build_report()   deterministic: cahier des charges §4.4 sections, only approved findings
  -> render_docx() / render_pdf()   two independent renderers, same Report model
```

Collects several `AuditFinding`s (Stage 3) from one audit engagement, gates them behind an
explicit human decision, and renders the approved ones into an actual Word or PDF report — the
cahier des charges' own risk mitigation for AI hallucination ("Validation humaine obligatoire")
implemented as a real gate, not a note in a document.

### Where things live

```
src/
├── audit_assistant.py           unchanged: AuditSession.add_finding() calls it, like the CLI already did
├── assistant_cli.py               + a `session` subcommand group (new | add-finding | list | review | report)
└── report/
    ├── session.py                  AuditSession, SessionMetadata, ReviewedFinding, ReviewDecision, Severity
    │                                + save_session()/load_session() -- one session = one JSON file
    ├── models.py                    Report, ReportSection, and block types: Paragraph, BulletList, Table
    ├── builder.py                    build_report(session) -> Report -- deterministic, no LLM call
    ├── docx_renderer.py                render_docx(report, path) -- python-docx
    └── pdf_renderer.py                  render_pdf(report, path) -- reportlab
```

`build_report()` only reads `session.approved` and `SessionMetadata`; it never calls Ollama, Qdrant
or the reranker. `docx_renderer.py` and `pdf_renderer.py` only walk `Report`/`ReportSection`/the
three block types — neither knows what an ISO finding or an audit session is, so the §4.4 section
mapping exists exactly once, in `builder.py`.

### Design decisions

- **JSON file persistence, not a database.** Every earlier stage in this prototype uses plain files
  over infrastructure (JSON chunks, JSON benchmarks). A `postgres`-backed store is the cahier des
  charges' target for the *full* system (§7), not this prototype phase. One session = one
  human-readable JSON file.
- **Severity (Majeure/Mineure) is a reviewer decision, not an LLM field.** The firm's own internal
  policy (`doc-politique-interne-cabinet`, section 05) classifies severity by criteria — systemic
  failure, impact on certification — that the retrieved evidence alone doesn't establish. Stage 3's
  `AuditFinding` schema stays untouched; `session review --approve --severity majeure|mineure`
  attaches it, and `build_report()` refuses to render an approved non-conformité with no severity
  set rather than silently omitting a field the firm's own policy requires.
- **The Conclusion is a deterministic template, not LLM-authored prose.** It reports real counts
  (non-conformités majeures/mineures, observations, opportunités, every referenced control) computed
  from the approved findings. The certification call stays human-owned, per the internal policy's
  cross-signature rule: originally rendered as an unresolved `[recommande / ne recommande pas]`
  bracketed placeholder (the same convention Stage 1's chunker recognises as a `template` unit in
  the knowledge base); revised in the Development-phase bug-fix round below into an explicit,
  human-set `AuditSession.certification_decision`, since shipping a raw template artifact in a
  downloaded deliverable turned out to read as a bug, not an intentional pending state.
- **Two independent renderers over one shared model, not docx→pdf conversion.** Converting a
  `.docx` to PDF needs Word or LibreOffice installed — a machine-specific dependency this project
  has avoided everywhere else. `reportlab` is pure Python; `docx_renderer.py` and
  `pdf_renderer.py` render the same `Report` independently, so the section logic is written once.
- **Only approved findings ever reach a report.** `session report` raises a clear error with zero
  approved findings, and `build_report()` reads nothing from pending or rejected findings.
  Confirmed end to end (`tests/test_report_e2e.py`): a real off-domain finding, rejected by the
  reviewer, was checked absent from both rendered files by its own distinctive wording.

### Commands

```bash
python -m src.assistant_cli session new --title "Rapport ISO/IEC 27001 - Client" --client "Client SA" \
    --scope "Systèmes d'information du siège." --standard "ISO/IEC 27001:2022" \
    --team "J. Dupont:Lead Auditor" --reference MISSION-001 -o session.json

python -m src.assistant_cli session add-finding session.json \
    "Le pare-feu applicatif accepte encore des connexions TLS 1.0 sur l'API exposée publiquement."

python -m src.assistant_cli session list session.json

python -m src.assistant_cli session review session.json 0 --approve --severity majeure --reviewer "J. Dupont"
python -m src.assistant_cli session review session.json 1 --reject

python -m src.assistant_cli session set-recommendation session.json --recommend --decided-by "J. Dupont, Auditeur Principal"

python -m src.assistant_cli session report session.json --format docx -o rapport.docx
python -m src.assistant_cli session report session.json --format pdf  -o rapport.pdf
```

A real run of this exact sequence (`Northwind Traders`, `MISSION-CLI-DEMO-01`, two findings, one
approved with severity, one rejected) produced a report with all eight §4.4 sections, the correct
client/team/standards table, one entry under Non-conformités with its cited references, correct
counts in the Conclusion (1 majeure, 0 mineure), and the bracketed recommendation placeholder — in
both `.docx` and `.pdf`, verified by reopening both files and reading their content back.

### A generation-quality observation, carried over honestly

Reviewing that real run: `llama3:8b`'s `finding` field is sometimes just the generic type label
("Non-conformité") rather than a restated sentence, while `recommendation`/`justification` carry
the actual substance. This is why the report builder shows `requirement`, `justification` and
`recommendation` as their own paragraphs rather than relying on `finding` alone — no information
from the model is lost to the report even when its headline field is terse. Not something Stage 4
tries to "fix" in the model (that's Stage 3's concern, already measured there); noted here because
it's exactly the kind of thing a reviewer benefits from seeing while approving a finding.

### Tests

```bash
python -m pytest -m "not integration and not e2e"   # offline only
python -m pytest -m e2e                              # real ingestion + retrieval + generation + report, live stack
```

| File | Level | What it checks |
|---|---|---|
| test_audit_session.py | unit | add-finding wiring (fake assistant), review transitions, severity scoped to non-conformité, edits/severity preserved across a re-review, JSON round-trip |
| test_report_builder.py | unit | every §4.4 section present and ordered, findings routed to the right section, pending/rejected excluded, edited text used over the original, Conclusion counts and reference list, both guard errors (no approved findings / unclassified non-conformité) |
| test_docx_renderer.py / test_pdf_renderer.py | unit | content round-trip -- reopen the written file with `python-docx` / `pypdf` and check headings, bold, bullet lists, table rows and Unicode/markup-like characters are really there |
| test_report_e2e.py | end-to-end | 3 real observations (one deliberately off-domain) through the real pipeline, realistic approve/reject/edit review, both formats rendered and reopened: real client name and references present, the edited text present, the rejected finding's distinctive wording absent from both files |

**Known limitations**
- No database/API/UI layer yet — named as next-stage work below, not attempted here.
- Report layout is functional, not a styled corporate template (no logo, custom fonts or firm
  letterhead) — no such template was provided to match.
- The Conclusion's counts and reference list are exactly reproducible from the approved findings;
  nothing about the Conclusion's *prose* (beyond the counts and the bracketed placeholder) is
  configurable per engagement yet.

### Next stage

Document import (importing existing client documents as additional retrievable evidence) is
implemented -- see the [Document import](#document-import) section below. Not implemented, named
for completeness (cahier des charges §7/§9): persistent, queryable audit history (a real database,
replacing the JSON-file session store once concurrent multi-auditor access is needed); and a web
UI/API so the CLI is not the only entry point.

## Document import

```
propose_import()   extract (PDF/DOCX/XLSX/CSV) -> anonymize -> write a reviewable sidecar file -> STOP
(auditor reviews the sidecar file's content by hand)
confirm_import()   re-read *that exact sidecar* -> chunk -> embed -> store in this session's own Qdrant collection
```

Imports client documents (previous audit reports, SoA, risk analyses, procedures, policies —
cahier des charges §4.3) as additional retrievable evidence for `session add-finding`. Client
documents are per-engagement and, by definition, full of exactly the sensitive content (server
names, IPs, employee names, company identity) the cahier des charges' top risk-table item exists
to protect — a different threat model from Stage 1's knowledge base, which is global, trusted, and
never anonymized because it never contains client data. Before implementing this, a Plan agent was
asked to adversarially stress-test the design against the actual code; it found four real gaps,
listed below as the design decisions that fix them.

### Where things live

```
src/documents/
├── extractors.py    extract_text(path) -- PDF (pypdf), DOCX (python-docx), XLSX/CSV (openpyxl/csv) -> plain text
├── ingestion.py        propose_import() / confirm_import() / session_chunks() / close_session_collection()
└── retrieval.py           build_document_retriever(session) -- Stage 2's BM25/Semantic/Hybrid, reused verbatim
```

Plus small additions to existing modules: `SessionMetadata.session_id`/`client_aliases` and
`AuditSession.anonymization_mapping`/`.anonymizer`/`.imported_documents` (`report/session.py`);
`AuditFinding.document_sources` (`generation/models.py`); a forced-review rule in
`GroundingValidator` and a labelled "client document context" prompt block
(`generation/{validation,prompts}.py`); a `session_id`-scoped `Anonymizer.restore()` classmethod
and a new `PERSON` detection rule (`anonymization/anonymizer.py`); five new `session` subcommands
(`assistant_cli.py`).

### Design decisions

- **A separate Qdrant collection per session**, never mixed into `iso27001_kb`
  (`f"{qdrant_collection}_client_{session_id}"`). Retrofitting a safe per-tenant filter onto the
  shared collection's optional, unenforced `filters` parameter is one missed argument away from
  leaking one client's document into another client's retrieval results; structurally separate
  collections make that bug class impossible rather than merely unlikely. `session close` deletes
  a session's collection once its report is generated.
- **Two-step human gate, sidecar-based.** `propose_import` never embeds anything — it writes the
  anonymized text to `data/imports/<session_id>/<doc_id>.anonymized.txt` and stops.
  `confirm_import` re-reads *that exact file*, not the original document again, so what the
  auditor reviewed is provably what reaches the model. Re-confirming an edited sidecar for the
  same `doc_id` replaces its old chunks (Stage 1's `delete_stale_chunks` pattern), not duplicates.
- **Four gaps a design-review pass found, fixed as default behaviour, not optional extras:**
  1. *The anonymizer's keyword-triggered rules are too narrow for document-scale text* (a
     one-sentence auditor finding has an implicit human backstop; a 40-page report nobody reads
     before it's embedded does not). Fixed: XLSX/CSV extraction wraps identity-bearing columns
     (nom, responsable, contact, propriétaire, owner, email, utilisateur) with a canonical trigger
     keyword the existing rules already recognise, and a new `PERSON` rule + `client_aliases` seed
     the anonymizer's `known_entities` so a bare repeated company name or person name is still
     caught with no keyword nearby.
  2. *Each CLI call got a fresh `Anonymizer()`*, silently resetting pseudonym numbering across
     invocations. Fixed: `Anonymizer.restore(mapping)` resumes a session's cumulative mapping, and
     every session command reads/writes `AuditSession.anonymization_mapping`.
  3. *Deanonymization only restored the current call's local map*, not the anonymizer's full
     cumulative one — silently failing to restore a placeholder introduced while anonymizing an
     imported document. Fixed in `audit_assistant.py`.
  4. *A non-citable prompt block stops fabricated ISO citations, not fact leakage.* Nothing
     stopped the model from quoting a real figure/name from document context verbatim. Fixed:
     `GroundingValidator` forces `requires_human_review=True` unconditionally whenever
     `document_sources` is non-empty — a deterministic backstop independent of prompt wording.
- **Client-document context never grounds an ISO citation.** It is a second, clearly labelled
  prompt block ("contexte documentaire du client... à ne jamais citer comme référence ISO"),
  populated into `AuditFinding.document_sources` verbatim (not filtered by citation, since there is
  nothing to validate a citation against) — unlike `supporting_sources`, which is. This instruction
  lives entirely in the *user* prompt, conditional on `document_evidence` being non-empty, not in
  the fixed `SYSTEM_PROMPT` sent on every call. It was first added as an always-present
  `SYSTEM_PROMPT` rule; a live A/B comparison against the grounding eval's off-domain abstention
  case (`D-01`, `test_off_domain_and_no_evidence_categories_abstain`) showed that addition alone
  measurably increased llama3:8b's rate of citing a weak, off-topic reference on document-less
  calls (5/6 pass with the rule present vs. 6/6, then 8/8, without it) — a small model can be
  sensitive to prompt changes even when they are explicitly conditional wording ("if X is
  provided...") and X never is. Moving the instruction into the conditional `document_block`
  removed that measurable regression entirely: a document-less call now sees byte-for-byte the
  same prompt Stage 3 already validated.
- **A finding still needs real ISO evidence to be drafted at all.** Client-document context is
  supplementary background, never a substitute for it: `FindingGenerator.generate()` still
  short-circuits to an insufficient-evidence result when the ISO evidence is empty, regardless of
  whether document evidence was found.

### Commands

```bash
python -m src.assistant_cli session new --title T --client "Client SA" --alias "Client" \
    --scope S --standard "ISO/IEC 27001:2022" --team "J. Dupont:Lead Auditor" -o session.json

python -m src.assistant_cli session import-document session.json rapport_precedent.pdf --doc-type previous_report
# -> writes an anonymized sidecar file and prints its path; review its content by hand, then:
python -m src.assistant_cli session confirm-import session.json <doc_id>

python -m src.assistant_cli session list-documents session.json
python -m src.assistant_cli session add-finding session.json "observation referencing the same entities"
python -m src.assistant_cli session close session.json   # deletes this session's Qdrant collection
```

A real run of this sequence (`Contoso SA`, an XLSX of service accounts with names/emails/server
IDs) produced a sidecar with zero real values in it, a confirmed import, and a finding correctly
marked `requires_review=True` with a populated `document_sources` entry; the final `.docx` had the
real client name back and no `PREFIX_NNN` placeholder token anywhere in it.

### Tests

```bash
python -m pytest -m "not integration and not e2e"   # offline only
python -m pytest -m e2e                              # real ingestion + retrieval + generation + report, live stack
```

| File | Level | What it checks |
|---|---|---|
| test_anonymizer.py | unit | `restore()` round-trip and counter continuation, the `PERSON` rule (full multi-word names, case-insensitive keyword), and a placeholder-nesting regression (a later multi-word match must never swallow an earlier placeholder) |
| test_document_extractors.py | unit | PDF/DOCX/XLSX/CSV -> plain text, including the identity-column keyword-wrapping behaviour |
| test_document_ingestion.py | unit (in-memory Qdrant) | propose/confirm two-step flow, sidecar-not-original confirmation, re-import replaces not duplicates, session isolation, pending documents excluded from `session_chunks` |
| test_validation.py / test_prompts.py | unit | `document_sources` populated verbatim and forces `requires_human_review`; the document-evidence prompt block is present only when document evidence is given |
| test_audit_assistant.py | unit | the document retriever is queried with the same anonymized query already used for ISO evidence; a placeholder introduced only via document evidence is still restored |
| test_document_e2e.py | end-to-end | the plan's exact scenario: import a document with a company name (caught only via `client_aliases`) and a server name, confirm, draft a finding referencing both, verify forced review, render the report, verify real values restored and zero placeholder tokens anywhere in it |

**Known limitations**
- No rerank pass over document evidence (unlike ISO evidence) — it is supplementary context, not
  a citable source, so this was judged not worth the added complexity yet.
- A session's document collection lives until `session close` deletes it; nothing currently expires
  it automatically.

## Development phase (Phase 3)

Cahier des charges §9: Phase 2 ("Prototype IA", above) validates the AI core; Phase 3
("Développement") turns it into the actual product, per §7's target architecture --
FastAPI + PostgreSQL + React, Qdrant and a local LLM (already in place since Stage 1/3).

### Audit before building: was the prototype good enough?

Before writing any product code, the prototype's actual output was re-examined against the
cahier des charges — not just re-reading the already-measured Stage 3/4 numbers, but running fresh
live generations (constats, non-conformités, observations, opportunités, conclusions) and reading
the text. Two categories of finding came out of it.

**Confirmed as already sound, unchanged:**
- The core safety property (§5.5, anonymisation before IA processing; no fabricated ISO reference
  ever reaches a report): still holds, reconfirmed by the full grounding eval suite.
- The knowledge base is a deliberately small, curated demo subset (7 Annex A controls across
  ~1,300 lines of source text, not the full 93-control Annex A) — many "insufficient evidence"
  abstentions are the system correctly refusing to fabricate on a topic the demo KB simply doesn't
  cover, not a grounding defect. Scaling the KB is a content task, not an AI-logic task.

**Three real weaknesses found live, fixed, and re-measured (not just patched and assumed fixed):**
1. **A schema gap against the firm's own internal policy.** The retrievable internal-policy
   document (`SECTION 05`) requires a non-conformity write-up to include a **"Risque"** field —
   `GeneratedFinding`/`AuditFinding` had no such field. Fixed: added `risk`, LLM-generated and
   grounded exactly like `justification`/`recommendation`, rendered in the report as "Risque : ...".
2. **A live, 100%-reproducible bug**: llama3:8b answered a French observation in English when the
   retrieved evidence was weak. Fixed: the language instruction moved earlier/more prominent in
   `SYSTEM_PROMPT`; confirmed 3/3 → 3/3 correct French across repeated trials after the fix.
3. **A live, reproducible fabrication pattern**: for an observation plainly describing a *compliant*
   situation, the model sometimes invented a non-conformity — once inventing a specific false
   detail not present anywhere in the observation or evidence. A targeted prompt rule
   (`SYSTEM_PROMPT` rule 3) and a new grounding-eval category (**F**, `evaluation/grounding_eval.json`)
   were added to measure it going forward. The prompt fix did **not** fully eliminate it (0%
   classification accuracy measured on both category-F cases even after the fix) — an honestly
   reported residual risk of an 8B local model, not hidden or claimed fixed. What the fix *does*
   guarantee, and what a new hard-gated test (`test_fabricated_non_conformite_is_always_forced_to_human_review`)
   checks on every category: such a finding is *always* forced to `requires_human_review=True`,
   so it can never silently reach a report — the cahier des charges' own "validation humaine
   obligatoire" mitigation is the actual backstop here, not the prompt. While broadening that test
   to cover every category, it also caught a second, unrelated real gap (two pre-existing cases
   where the model said "insufficient evidence" but had not itself asked for review) — fixed with
   a general invariant: `evidence_status="insufficient"` now always forces review, regardless of
   the model's own flag.

**Verdict:** sufficiently validated to build on, with the above three fixes applied and
re-measured, and category F's residual risk carried forward honestly as a known limitation (see
below) rather than treated as blocking — it is bounded by mandatory human review, which is not
optional in this design at any layer.

### Where things live

```
src/                          unchanged: the validated Stage 1-4 + Document Import library
api/                           FastAPI backend -- imports src/, never duplicates its logic
├── main.py                      app factory, CORS, lifespan (table creation + admin bootstrap)
├── db.py                          SQLAlchemy engine/session factory
├── models_orm.py                   User, AuditSessionRecord (JSONB), AuditLogEntry
├── repository.py                     AuditSession <-> Postgres (model_dump/model_validate, no new logic)
├── auth.py                             JWT + bcrypt, role-based `require_role(...)` dependency
├── audit_log.py                          §5.4 journalisation (append-only)
├── schemas.py                              request bodies only -- responses reuse src/'s own models
├── dependencies.py                           shared `get_record_and_session` (ownership/visibility)
└── routers/{auth,users,sessions,documents}.py
web/                           React (Vite + TypeScript) frontend
├── src/api/{client,types}.ts    typed fetch wrapper + JWT storage
├── src/auth/AuthContext.tsx       login state
├── src/pages/                       Login, session list (+ create), session detail (3 tabs:
│                                     Constats / Documents / Rapport), Utilisateurs
└── src/components/Layout.tsx
```

`api/` never reimplements retrieval, generation, anonymization, document ingestion or report
rendering -- every router calls straight into the same `AuditAssistant`, `AuditSession`,
`build_document_retriever`, `build_report`, `render_docx`/`render_pdf` the CLI already used. Only
persistence (JSON file -> Postgres JSONB) and an HTTP/auth layer are new.

### Design decisions

- **JSONB, not a fully normalised schema.** `AuditSessionRecord.data` holds exactly
  `AuditSession.model_dump()` -- the same object the CLI wrote to a JSON file. This reuses 100% of
  the already-validated `AuditSession` business logic (`add_finding`, `review`, `.approved`, ...)
  completely unchanged; only its storage backend moved. A handful of columns (`title`,
  `client_name`, `reference`, `owner_id`) are denormalised alongside it purely so listing/filtering
  sessions doesn't need to deserialise every row.
- **Local JWT auth now, SSO-shaped for later.** Cahier des charges §5.3 names Azure AD/SSO/MFA;
  those need external identity infrastructure this environment doesn't have. `get_current_user`'s
  contract (a dependency resolving a signed token to a `User`) is exactly what a real OIDC/SSO
  integration would also need to satisfy, so swapping the issuer later doesn't change any router.
  Three roles (auditeur / chef d'équipe / administrateur), least privilege: an auditeur only sees
  their own sessions; the other two see every session and can manage users.
- **A dedicated Postgres container, never the machine's other Postgres instance.** Same isolation
  principle as the per-session Qdrant collections: `docker run ... -p 5433:5432 ...`, its own
  database and user, never sharing a port or database with anything else already on the machine.
- **Journalisation (§5.4) as an append-only table**, logging login, session/document/finding
  actions and report downloads. Retention (>= 12 months) is a database-retention-policy concern for
  deployment, not application code.
- **The web layer adds zero new anonymization, grounding or isolation logic.** `add_finding`'s
  router builds the session's persistent `Anonymizer` and document retriever exactly as the CLI's
  `cmd_session_add_finding` does; `get_session`'s response excludes `anonymization_mapping` (real,
  deanonymized values, an internal artifact never needed by the UI) via `response_model_exclude`.

### Setup

```bash
docker run -d --name iso27001_audit_postgres -e POSTGRES_USER=audit_app \
    -e POSTGRES_PASSWORD=<choose one> -e POSTGRES_DB=iso27001_audit \
    -p 5433:5432 -v iso27001_audit_pgdata:/var/lib/postgresql/data postgres:15-alpine

pip install -r requirements.txt          # adds fastapi, uvicorn, sqlalchemy, psycopg2, bcrypt, pyjwt
cp .env.example .env                      # set DATABASE_URL / JWT_SECRET_KEY

cd web && npm install
```

### Commands

```bash
uvicorn api.main:app --reload --port 8000   # backend -- creates tables and bootstraps the first
                                              # administrateur account on first run (its one-time
                                              # generated password is logged, not printed to stdout)
cd web && npm run dev                        # frontend, http://localhost:5173 (proxies /api to :8000)
```

### Tests

```bash
python -m pytest -m "not integration and not e2e"        # offline (359 tests)
python -m pytest tests/test_api -m "integration or e2e"    # API: auth, RBAC, sessions, documents,
                                                              # full live workflow (35 tests)
```

| File | Level | What it checks |
|---|---|---|
| test_auth.py | integration (Postgres) | login success/failure/inactive user, `/auth/me`, admin-only user creation, role-based access on `/users` |
| test_sessions.py | integration (Postgres) | create/list/get, least-privilege visibility (auditeur vs chef d'équipe), `anonymization_mapping` never sent over the wire, review/report guard errors |
| test_documents.py | integration (Postgres) + e2e (confirm) | upload -> anonymized sidecar -> confirm, cross-session isolation, unsupported file type |
| test_e2e.py | end-to-end, live stack | the full workflow through real HTTP requests: login, import a document, review its sidecar, confirm, draft a finding referencing the same entities, verify forced review, approve, download the report, verify real values restored and zero placeholder tokens anywhere in it |

A real manual walkthrough (curl/Python `requests` against the running server, then the same flow
through the Vite dev server's `/api` proxy) produced a correct end-to-end result: an XLSX with
real names/emails/server IDs imported, a sidecar with zero real values in it, a finding with a real
drafted `risk` and the real client/server names restored, and a downloaded `.docx` with the new
"Risque :" paragraph and no placeholder token anywhere in it.

**Known limitations**
- **Category F residual risk** (see the audit above): llama3:8b can still fabricate a
  non-conformity for a compliant observation. Always forced to human review; not eliminated.
  Re-measure `evaluate-grounding`'s category F rate if the generation model is ever upgraded.
- **SSO/Azure AD/MFA (§5.3), TLS in transit and encryption at rest (§5.2)** are deployment-phase
  concerns (reverse proxy/TLS termination, disk/volume encryption, real IdP integration) — not
  implemented, and not things application code should fake.
- **No automated database migrations yet** (`alembic` is installed but not initialised) — schema
  changes currently mean re-running `Base.metadata.create_all`, fine for this development pass, not
  for a production upgrade path.
- The React UI is functional and covers the full workflow, not a polished, branded design — no
  corporate template was provided to match (same scoping call as Stage 4's report layout).

### Real manual-test bug fixes (2026-09-17)

The user generated a real audit report through the running application (`aa` as client/scope, a
genuine access-control observation) and reported several problems. Each was root-caused in the
actual implementation, not patched at the output:

- **ISO references silently dropped between the generated finding and the final report**
  (`GroundingValidator`, `src/generation/validation.py`). Root cause: the reference-matching
  fallback only recognised a citation already close to canonical form (exact string, or a bare
  local id with no standard prefix at all); a citation like `"27002 §5.15"` -- no "ISO/IEC" word, a
  "§" before the number, both completely normal ways to write it -- matched neither path and was
  dropped even though the evidence genuinely backed it. Fixed by reusing
  `src/ingestion/references.py`'s own reference extractor (the exact same deterministic rules the
  knowledge base's references were canonicalised with) as a third fallback, re-checking the
  re-parsed candidate's ambiguity against the evidence the same way the direct paths already do --
  a bare "A.x.y" defaulting to ISO/IEC 27001 by convention must not silently override a real
  ambiguity the evidence itself raises (two existing tests caught this on the first pass; both are
  now covered without weakening the "never guess when ambiguous" guarantee).
- **Raw Markdown table syntax (`| |`, `|-|`) leaking into the report.** Root cause: nothing stopped
  the model from drafting a list as a Markdown table in a free-text field meant for a formal report
  paragraph. Fixed at two layers, matching how grounding itself is layered: `SYSTEM_PROMPT` now
  says these fields must be plain prose (defense in depth, not solely relied on), and
  `GroundingValidator` deterministically strips Markdown separator rows and turns a real data row
  into a plain, readable list -- a no-op for the overwhelming majority of findings, which contain
  no "|" at all.
- **Unresolved `[recommande / ne recommande pas]` template placeholder shipped in a downloaded
  report.** This was a design gap, not a rendering bug: nothing in the system ever captured the
  Auditeur Principal's actual certification decision. Fixed by adding an explicit,
  human-only field (`AuditSession.certification_decision`, the same "a human decision, not an LLM
  field" pattern already used for non-conformité severity), settable via
  `POST /sessions/{id}/certification-decision` (chef d'équipe/administrateur only -- the closest
  match to "Auditeur Principal" in this role model) or `session set-recommendation` on the CLI.
  `build_report()` renders the resolved sentence once set, and an explicit "reste à confirmer par
  l'Auditeur Principal" note (never a raw bracket, never a silently invented default) otherwise.
- **Missing metadata rendered as `-` or `? - ?`.** Replaced everywhere with a professional
  `NOT_PROVIDED = "Non renseigné"` placeholder (`src/report/builder.py`), including a proper
  partially-known audit period ("2026-01-06 - Non renseigné" rather than losing the known side).
- **The auditor's original observation never appeared in the report**, only the model's drafted
  paraphrase. The firm's own internal policy (SECTION 04, retrievable in the knowledge base)
  requires "une description factuelle et neutre de la situation observée" as its own element of a
  constat; added as its own "Observation initiale :" paragraph, giving a reviewer full
  observation -> finding -> ISO reference -> risk -> report traceability in the rendered document.

11 new/changed regression tests (`test_validation.py`, `test_report_builder.py`,
`tests/test_api/test_sessions.py`), full suite re-verified with zero regression (359 offline + the
complete live integration/e2e suite), and the exact reported scenario re-run live end to end
against the running API and database -- confirmed fixed, not just unit-tested in isolation.

### "Preuves insuffisantes" follow-up: the KB and retrieval were never the problem

A follow-up question ("why insufficient evidence when I have real data in Qdrant?") led to a second
real fix, found by reproducing the user's exact observations live rather than guessing. Diagnosis
first: Qdrant had 110 points across all 6 `data/raw` documents (nothing missing, nothing stale), and
retrieval found strongly on-topic evidence for both observations (reranker scores 1.5-2.1, well
above the confidence threshold) -- so the knowledge base and retrieval were confirmed working
correctly. The actual cause was a self-contradiction at the generation step: llama3:8b sometimes
cites a real, well-evidenced reference while *also* labelling its own answer
`evidence_status="insufficient"` -- inconsistent even for the identical observation across repeated
tries. The validator's old rule always trusted the "insufficient" half and discarded the citation
outright, silently throwing away real traceability the evidence supported.

Fixed in `GroundingValidator.validate()` (`src/generation/validation.py`) by resolving the
contradiction the same way every other judgment call in this validator is made: by the reranker's
own independent confidence score for the cited reference, reusing the same
`LOW_CONFIDENCE_REVIEW_THRESHOLD` and `_confidence`/`_supporting_sources` methods the low-confidence
override already uses, rather than adding a second, parallel notion of "strong evidence." When the
citation is well-backed, it is kept (`evidence_status` becomes `PARTIAL`, never an automatic
`SUPPORTED`) and `requires_human_review` is forced on; when it is weak, the previous conservative
behaviour is unchanged -- the citation is dropped and `evidence_status` stays `insufficient`. Four
new regression tests cross {supported, insufficient} x {strong, weak} evidence
(`TestSelfContradictionResolvedByEvidenceStrength`, `test_validation.py`).

Re-running the live grounding eval after this change also caught an unrelated pre-existing test bug
(not a regression from this fix): `test_fabricated_non_conformite_is_always_forced_to_human_review`
was named and documented as a category-F-specific check but its assertion covered every category,
so it happened to pass only because every case in past runs coincidentally required review for
independent reasons. Once a category-A case legitimately came back clean and well-evidenced with no
contradiction -- correctly *not* requiring review, which is the point of confidence-based review,
not a gap in it -- the over-broad assertion failed. Narrowed to what it actually names (category F),
plus a new, correctly-scoped `test_insufficient_evidence_status_is_always_forced_to_human_review`
for the general invariant it was accidentally also covering. Full grounding eval suite (7 tests,
including both hard-gated anti-hallucination guarantees) re-verified passing after the correction.

### Next step

Not started: database migrations (Alembic), a styled frontend, deployment (Phase 5: containerising
the whole stack, TLS termination, real SSO), and Phase 4's dedicated security/functional test pass.

## Earlier prototype

The first prototype (in-memory, no vector database) is still runnable:

```bash
python -m src.main "Chez ABC Consulting, le serveur 192.168.1.20 n'est pas correctement protégé." --lang fr
```

It chunks `data/raw` into `data/processed/chunks.json` (`src/ingestion/document_loader.py`), anonymizes
the input (`src/anonymization`), retrieves with BM25 + in-memory cosine similarity fused by RRF
(`src/retrieval`), generates with a local Ollama model (`src/generation`) and restores the original
values. Options: `--lang fr|en`, `--model <ollama model>` (default `llama3`), `--top-k <n>`.
