# OpsMind Backend

FastAPI backend for PDF ingestion, baseline chat, and document-grounded RAG chat.

## Set up and run locally

From the repository root:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
set -a
source .env
set +a
uvicorn app.main:app --reload
```

The development server runs at <http://127.0.0.1:8000>.

## PostgreSQL and migrations

Install and start PostgreSQL locally, then create a database and matching local role. For example, with a local PostgreSQL installation:

```bash
createuser opsmind --createdb --pwprompt
createdb opsmind --owner=opsmind
```

For the example URL in `.env.example`, set the role password to `opsmind`, or change the URL to match the password you choose. Copy and source the file as shown above, then run the initial migration from the `backend/` directory:

Before migrating, install the pgvector PostgreSQL server extension for your PostgreSQL distribution. On Homebrew PostgreSQL, use `brew install pgvector`. The `opsmind` application role may not have permission to install extensions; if it is not a PostgreSQL superuser, connect as a database admin and run `CREATE EXTENSION IF NOT EXISTS vector` in the `opsmind` database first. The Alembic migration also runs this statement when it has permission.

```bash
alembic upgrade head
```

To see the current revision, use `alembic current`. To reverse the initial schema migration, use `alembic downgrade base`.

`DATABASE_URL` must be set in the environment for Alembic and for code that imports the database module. Keep real credentials in your local `.env`; do not commit that file.

## Chat endpoint

Set `OPENAI_API_KEY` to an API key for your OpenAI project and `OPENAI_MODEL` to a model available to that project. The example `.env` uses `gpt-6-luna`; replace its API key placeholder with your own key. The application does not load `.env` automatically, so source it in the shell using the commands above.

Apply the usage-event migration from the `backend/` directory:

```bash
alembic upgrade head
```

Send a request to `/chat`:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"How should I prepare for a team handoff?"}'
```

The response contains an answer and the model used. Raw input/output token counts, latency, success status, and a database-generated timestamp are recorded in `ai_usage_events`. Provider failures return a generic `502`; missing LLM configuration returns `503`.

## PDF document ingestion

The Python `pgvector` dependency provides SQLAlchemy support; the PostgreSQL server extension provides the `vector` type used by document chunks.

Set `OPENAI_EMBEDDING_MODEL` in `.env` to an OpenAI `text-embedding-3` model. The default is `text-embedding-3-small`; OpsMind requests 1536 dimensions to match the `vector(1536)` column. Embedding calls occur during document uploads and RAG questions. Tests mock those calls.

Run or upgrade the database schema from `backend/`:

```bash
alembic upgrade head
```

Upload a PDF with a `multipart/form-data` request. Use `uploaded_by` for an existing `users.id` and attach the file under `file`:

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -F 'uploaded_by=1' \
  -F 'file=@./example.pdf;type=application/pdf'
```

The endpoint rejects empty or invalid PDFs, unknown users, and duplicate SHA-256 content. It extracts text from each page without OCR, groups paragraphs under detected headings, and splits chunks to at most 800 `cl100k_base` tokens. Oversized blocks have a 75-token overlap between splits. It sends chunks to the configured embedding model in batches of up to 64 and stores the document and chunks in one transaction. The original PDF is not retained; a document becomes `ready` only after all chunks and vectors are stored.

## RAG chat endpoint

Upload a PDF first and use its returned `document_id` to ask about that document:

```bash
curl -X POST http://127.0.0.1:8000/rag/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"What does the document say about access rules?","document_id":1}'
```

`document_id` is optional. With it, retrieval searches only that ready document; without it, retrieval searches all ready documents. The endpoint embeds the question using the same configured embedding model used for document chunks, then retrieves the five nearest chunks by pgvector cosine distance. There is no similarity cutoff or vector index yet.

The response contains `answer` and a `sources` list with database-derived `document_id`, `filename`, `page_number`, `section_title`, `chunk_index`, and `distance`. The answer is instructed to cite source numbers such as `[1]`; these numbers refer to the order of the returned sources. If no chunks are found, the endpoint returns an insufficient-evidence answer with an empty `sources` list. The question and retrieved chunk text are sent to the configured OpenAI API when an answer is generated.

Run `alembic upgrade head` before using RAG usage tracking. Each RAG request gets one UUID `request_id`; its query embedding and generated answer each write an `ai_usage_events` row with that ID. Rows record the provider model, available token counts, call latency, and success status. Embeddings have zero output tokens. Existing usage rows are labeled `llm` by the migration and keep a null `request_id`.

New `/chat` and RAG usage rows also store an estimated USD cost based on the recorded model and token counts. Prices checked **2026-10-13** against the official OpenAI model pages: [`gpt-6-luna`](https://developers.openai.com/api/docs/models/gpt-6-luna) costs **$0.10 per 1M input tokens** and **$0.50 per 1M output tokens**; [`text-embedding-3-small`](https://developers.openai.com/api/docs/models/text-embedding-3-small) costs **$0.02 per 1M input tokens**. The formula is `input_tokens / 1,000,000 * input_price + output_tokens / 1,000,000 * output_price` for LLM calls, and the input term alone for embeddings. Estimates use uncached, short-context Standard prices; actual charges can differ for cached tokens, long context, other processing tiers, or regional premiums. Unknown model prices and failed calls without token usage leave `estimated_cost` null. Existing rows are not recalculated; model and raw token counts remain the source of truth.

## Retrieval evaluation

`evals/questions.json` contains fictional example cases. Replace the sample document IDs, questions, expected facts, and section titles with cases for PDFs in your own database before calculating a meaningful score. Each case needs an `id`, `question`, `expected_document_id`, `expected_fact`, `expected_section`, and `category`.

From `backend/`, with `.env` sourced and the database available, run:

```bash
python -m evals.run --questions evals/questions.json --output evals/results.json
```

Use `--top-k 3`, `--top-k 5`, or `--top-k 10` to compare retrieval depth. Evaluation passes this value to pgvector retrieval and saves it with Recall@1, Recall@k, Section@1, FactMatch, average RAG latency, and average estimated AI cost per request in `evals/results.json`. The API endpoint `/rag/chat` still retrieves five chunks. Each evaluation case gets a request ID and records its embedding and answer usage events. RAG latency covers embedding, retrieval, answer generation, and usage recording; optional judge calls are excluded from latency and cost. If a case has no usage events or an unknown model price, the average estimated cost is `null` rather than treating it as zero.

Add `--llm-judge` to make one additional OpenAI grading call per case:

```bash
python -m evals.run --questions evals/questions.json --output evals/results.json --llm-judge
```

For each case, the script runs the existing RAG answer flow, prints per-question Recall@1, Recall@5, Section@1, and FactMatch results, and saves ranked source details plus the generated answer to `evals/results.json`. Recall@1 is the fraction of questions whose expected document is the first result; Recall@5 is the fraction where it appears among the five closest chunks; Section@1 is the fraction whose first result has the expected section title. FactMatch is the fraction of generated answers containing the expected fact after lowercasing and normalizing punctuation and whitespace. The optional `--llm-judge` receives the question, expected fact, answer, and retrieved chunk text with source metadata. It accepts supported extra details and paraphrases, but marks unsupported or contradictory material claims and missing required facts incorrect. It reports LLMJudge accuracy and saves each `correct` result with a short reason. Running the script makes embedding and answer-generation API calls; the optional judge adds one grading call per case. The automated tests mock embeddings, retrieval, answer generation, and judge results.

## AI FinOps / RAG benchmark

The saved benchmark compares three retrieval depths using the same 10 synthetic questions and 4 synthetic documents. These are measured averages from one run per setting; the latency values below are in milliseconds and the costs are estimated AI API USD per RAG request.

| top_k | Recall@1 | Recall@k | Section@1 | FactMatch | LLM judge | Avg RAG latency (ms) | Est. AI API cost/request (USD) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 100% | 100% | 100% | 80% | 100% | 1725.6537625 | $0.000053582 |
| 5 | 100% | 100% | 100% | 90% | 100% | 1659.8823541999998 | $0.000075662 |
| 10 | 100% | 100% | 100% | 90% | 100% | 1770.0031 | $0.000121792 |

Monthly examples multiply each measured average estimated AI API cost/request by the request count. They are projections, not additional measurements:

| top_k | 1,000 requests | 10,000 requests | 100,000 requests | 500,000 requests |
| ---: | ---: | ---: | ---: | ---: |
| 3 | $0.053582 | $0.53582 | $5.3582 | $26.791 |
| 5 | $0.075662 | $0.75662 | $7.5662 | $37.831 |
| 10 | $0.121792 | $1.21792 | $12.1792 | $60.896 |

Keep `top_k=5` for `/rag/chat`: it matched `top_k=10` on the measured quality metrics, improved FactMatch over `top_k=3`, and had lower measured latency and estimated AI API cost than `top_k=10`. These costs exclude infrastructure and the optional LLM-judge evaluation calls. The benchmark is limited to 10 synthetic questions, 4 synthetic documents, and a single run, so the result is a starting point rather than a general performance guarantee.

![Estimated AI API cost per RAG request by top_k](benchmarks/results/cost_vs_top_k.png)

![Average RAG latency by top_k](benchmarks/results/latency_vs_top_k.png)

![Retrieval and answer quality by top_k](benchmarks/results/quality_vs_top_k.png)

Detailed results: [benchmark report](benchmarks/results/report.md), [summary CSV](benchmarks/results/summary.csv), and [monthly API projection CSV](benchmarks/results/monthly_api_projection.csv).

To run the automated tests, install the development requirements and run pytest:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Check the endpoints

In another terminal, run:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/version
```

The responses are `{"status":"healthy"}` and `{"version":"0.1.0"}`.
