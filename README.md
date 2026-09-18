# ISO/IEC 27001 Audit Assistant

Assistant IA qui aide les auditeurs ISO/IEC 27001 à rédiger leurs constats, non-conformités,
observations et opportunités d'amélioration, en s'appuyant sur une base de connaissances normative.

**Tout s'exécute en local.** Aucune donnée ne transite vers un service IA public : le modèle de
langage (Llama 3) est servi par Ollama sur la machine, conformément à l'exigence de confidentialité
du cahier des charges.

---

## Le problème

Un auditeur écrit une observation brute :

> « Le contrôle des accès privilégiés n'est pas revu périodiquement. »

Le système produit un constat structuré, rattaché à la norme, avec son risque et ses sources :

```json
{
  "finding": "Le contrôle des accès privilégiés n'est pas revu périodiquement, ce qui maintient
              des droits d'accès non justifiés et augmente la surface d'exposition.",
  "finding_type": "non_conformite",
  "iso_reference": ["ISO/IEC 27001 A.5.18"],
  "risk": "Maintien de droits d'accès non justifiés, augmentant la surface d'exposition
           en cas de compromission de compte.",
  "evidence_status": "partial",
  "requires_human_review": true,
  "confidence": 0.847,
  "supporting_sources": [{ "chunk_id": "...", "rerank_score": 2.1 }]
}
```

Le point clé : **`requires_human_review` et `confidence` ne viennent jamais du modèle**. Ils sont
calculés de manière déterministe en vérifiant chaque citation contre les preuves réellement
récupérées.

---

## Fonctionnalités

| Domaine | Ce qui est fait |
|---|---|
| **Rédaction assistée** | Génération de constats, non-conformités, observations et opportunités d'amélioration, en français ou en anglais |
| **Base de connaissances** | ISO/IEC 27001:2022, 27002:2022, 27017, 27018, 27701 et la politique interne du cabinet |
| **Recherche** | Retrieval hybride BM25 + sémantique, fusionné par RRF puis reranké par cross-encoder |
| **Référencement automatique** | Extraction et canonicalisation des clauses citées (`A.5.18`, `27002 §8.13`, `RGPD art. 33`) |
| **Anonymisation** | Noms de clients, IP, serveurs, comptes et applications remplacés avant tout appel au LLM, restaurés après |
| **Garde-fous anti-hallucination** | Vérification des citations, détection des chiffres non rapportés et des contradictions observation/constat |
| **Sessions d'audit** | Regroupement de constats, cycle de revue humaine (approuver / rejeter / modifier), décision de certification |
| **Rapports** | Export Word (.docx) et PDF |
| **Import documentaire** | Documents client (PDF, DOCX, XLSX, CSV) comme preuves additionnelles, cloisonnées par session |
| **API & interface** | FastAPI avec JWT et RBAC à 3 rôles, journal d'audit, frontend React |

---

## Architecture

```
Observation de l'auditeur
   │
   ├─► Anonymisation ............ CLIENT_001, SERVER_001, IP_001, USER_001
   │
   ├─► Retrieval ................ BM25 (lexical) + sémantique (Qdrant/bge-m3)
   │      └─► Fusion RRF ........ Reciprocal Rank Fusion
   │      └─► Reranking ......... cross-encoder, top-5 transmis au modèle
   │
   ├─► Génération ............... Llama 3 8B via Ollama, sortie JSON structurée
   │
   ├─► Validation du grounding .. chaque référence vérifiée contre les preuves réelles ;
   │                              confiance dérivée du score du reranker, jamais du modèle
   │
   ├─► Désanonymisation ......... valeurs réelles restaurées pour l'auditeur
   │
   └─► Session d'audit .......... revue humaine → rapport Word / PDF

Hors ligne, alimentant le retrieval :
   data/raw → parsing → découpage sémantique → embeddings → Qdrant
```

Détail technique complet de chaque étape : **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

---

## Technologies

| Couche | Choix | Alternative prévue au cahier des charges |
|---|---|---|
| Frontend | React 19, TypeScript, React Router 7, Vite 8 | React *ou* Angular |
| Backend | Python, FastAPI | Python + FastAPI |
| Base de données | PostgreSQL | PostgreSQL |
| Moteur vectoriel | Qdrant | ChromaDB *ou* Qdrant |
| Génération | Llama 3 8B via Ollama (local) | Azure OpenAI *ou* Llama local |
| Embeddings | bge-m3 (multilingue, 1024 dim) | — |
| Reranking | cross-encoder `mmarco-mMiniLMv2-L12-H384-v1` (CPU) | — |
| Authentification | JWT + bcrypt | SSO / Azure AD / MFA *(non implémenté)* |

---

## Structure du projet

```
├── src/
│   ├── ingestion/        parsing, découpage sémantique, extraction des références
│   ├── embeddings/       client d'embeddings (Ollama)
│   ├── vectorstore/      couche Qdrant
│   ├── retrieval/        BM25, sémantique, fusion RRF, reranking
│   ├── anonymization/    anonymisation réversible par règles
│   ├── generation/       prompts, génération structurée, validation du grounding
│   ├── documents/        import de documents client par session
│   ├── report/           sessions d'audit, rendu Word et PDF
│   ├── evaluation/       benchmarks retrieval et grounding
│   ├── kb.py             CLI base de connaissances
│   └── assistant_cli.py  CLI assistant complet
│
├── api/                  FastAPI : auth, users, sessions, documents, journal d'audit
├── web/src/              React : pages, contexte d'auth, client API
├── data/raw/             base de connaissances (6 documents)
├── evaluation/           jeux de test des benchmarks
├── tests/                445 tests unitaires et d'intégration + suite e2e
└── docs/ARCHITECTURE.md  documentation technique détaillée
```

---

## Démarrage rapide

### Prérequis

- Python 3.11+, Node 20+, Docker
- [Ollama](https://ollama.com) avec les modèles `llama3:8b` et `bge-m3`

```bash
ollama pull llama3:8b
ollama pull bge-m3
```

### Installation

```bash
# 1. Dépendances Python
pip install -r requirements.txt

# 2. Configuration (générez votre propre secret JWT)
cp .env.example .env
python -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_hex(32))"   # à reporter dans .env

# 3. Infrastructure (PostgreSQL + Qdrant)
docker compose up -d postgres qdrant

# 4. Indexation de la base de connaissances
python -m src.kb ingest

# 5. Backend
python -m uvicorn api.main:app --port 8000

# 6. Frontend (dans un autre terminal)
cd web && npm install && npm run dev
```

L'interface est alors disponible sur `http://localhost:5173`.

> Au premier démarrage sur une base vide, des comptes de développement sont créés
> (voir `api/main.py`). Ils sont destinés au développement local uniquement et doivent être
> remplacés avant tout usage réel.

---

## Utilisation en ligne de commande

Aucun besoin de lancer l'interface pour tester le moteur :

```bash
# État de la base vectorielle
python -m src.kb stats

# Recherche seule, sans génération (rapide)
python -m src.kb search "Les sauvegardes ne sont pas testées régulièrement" --top-k 5

# Pipeline complet sur une observation
python -m src.assistant_cli analyze "Le contrôle des accès privilégiés n'est pas revu périodiquement." --lang fr

# Benchmark de la qualité du retrieval
python -m src.kb evaluate

# Benchmark de résistance à l'hallucination
python -m src.assistant_cli evaluate-grounding --details
```

---

## Résultats mesurés

### Qualité du retrieval — `python -m src.kb evaluate`

50 requêtes réparties en 6 catégories, sur un corpus de 178 chunks.

| Catégorie | n | hit@1 | hit@5 | MRR@10 |
|---|---|---|---|---|
| finding | 19 | 0.89 | 1.00 | 0.95 |
| normative | 10 | 0.80 | 1.00 | 0.86 |
| policy | 9 | 0.89 | 1.00 | 0.91 |
| english | 5 | 0.80 | 1.00 | 0.90 |
| drafting | 4 | 0.75 | 1.00 | 0.88 |
| reference | 3 | 0.67 | 1.00 | 0.83 |
| **Global** | **50** | **0.84** | **1.00** | **0.91** |

La bonne source figure dans le top 5 pour **100 %** des requêtes. Seuils de qualité imposés par la
suite de tests : hit@5 ≥ 0.90, MRR@10 ≥ 0.80.

### Résistance à l'hallucination — `python -m src.assistant_cli evaluate-grounding`

13 cas couvrant : observations bien étayées, observations hors périmètre, demandes hors sujet,
et observations décrivant une situation **conforme** (piège classique du modèle qui invente une
non-conformité).

| Métrique | Résultat |
|---|---|
| Références non étayées dans la sortie finale *(doit être 0)* | **0 / 13** |
| Citations faiblement étayées non signalées pour revue *(doit être 0)* | **0 / 13** |
| Observations conformes transformées en fausse non-conformité *(doit être 0)* | **0 / 2** |
| Tentatives de citation non étayée par le modèle, avant validation | 8 / 13 |

La dernière ligne est la plus parlante : le modèle **tente** régulièrement de citer une référence
que les preuves ne contiennent pas — la couche de validation les bloque toutes.

### Base de connaissances

| | |
|---|---|
| Documents | 6 |
| Chunks indexés | 178 |
| Contrôles de l'Annexe A référencés | 41 |
| Exemples de conformité / d'écart | 14 / 21 |

### Tests

```bash
python -m pytest tests/ -q -m "not e2e"   # 445 tests
python -m pytest tests/ -q -m e2e          # suite e2e (Qdrant + reranker réels)
```

---

## Données d'exemple

La base de connaissances contient des exemples de constats, non-conformités et opportunités
d'amélioration **entièrement synthétiques** : identifiants (`NC-2026-014`), références internes
(`POL-SEC-004`), chiffres et dates sont fictifs et ne décrivent aucune organisation réelle.

Seules les exigences normatives citées en appui sont réelles et vérifiables auprès de leurs
sources officielles : ISO/IEC 27001 et 27002, ISO 19011, ainsi que les publications de l'ANSSI,
de la CNIL et du NIST.

---

## Limites connues

- **Pas de garde-fou thématique** : une question hors sujet produit tout de même un constat formaté
  (à confiance nulle et systématiquement envoyé en revue humaine, mais généré).
- **Détection de contradiction lexicale** : la règle qui repère un constat contredisant l'auditeur
  s'appuie sur des marqueurs linguistiques (français et anglais) ; une paraphrase inhabituelle peut
  lui échapper.
- **Chiffrement non couvert** : ni TLS en transit ni chiffrement au repos ne sont configurés dans
  cette phase de développement.
- **Authentification** : JWT nom d'utilisateur/mot de passe uniquement. Le SSO, Azure AD et le MFA
  demandés au cahier des charges ne sont pas implémentés.
- **Journalisation partielle** : connexions et modifications sont tracées ; la couverture des
  consultations et téléchargements ainsi que la rétention à 12 mois restent à compléter.

---

## Licence et contexte

Projet réalisé dans le cadre d'un cahier des charges d'assistance à l'audit ISO/IEC 27001.
Les normes ISO citées sont la propriété de l'ISO et ne sont pas redistribuées ici : la base de
connaissances contient des synthèses et des exemples rédigés pour le projet.
