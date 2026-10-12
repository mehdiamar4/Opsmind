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

Set `OPENAI_EMBEDDING_MODEL` in `.env` to an OpenAI `text-embedding-3` model. The default is `text-embedding-3-small`; OpsMind requests 1536 dimensions to match the `vector(1536)` column. Embedding calls occur only when a document is uploaded. Tests mock those calls.

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
