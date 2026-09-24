# Policy-Aware Customer Support Agent

A small agent that answers customer questions strictly from a set of policy
documents, returns a structured result, and escalates to a human when the
documents do not support a confident answer.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env     # then paste your Gemini key into .env
```

Get a free Gemini API key at https://aistudio.google.com/apikey. **A key is
required** — there is no offline fallback, so the app will not answer without
one. The test suite is the exception: it patches the model call with its own
stub, so `pytest` runs with no key and no network.

```bash
python app.py --check   # verifies the key and model before you run anything
```

**Free-tier quota.** Gemini's free tier allows 5 requests per minute and only
**20 requests per day, per model, per project**. `run_batch.py` paces itself to stay
under the per-minute limit. If you exhaust the daily budget, point `GEMINI_MODEL`
at a different model (for example `gemini-flash-lite-latest`) — the daily quota is
counted per model, so another one has its own budget.

## Run

```bash
python app.py                 # web UI at http://127.0.0.1:5001
python app.py --check         # verify the API key and model, then exit
```

Every question asked through the UI is appended to `logs/query_log.csv` with the
answer and every signal behind it. That file is gitignored: in production it
would hold raw customer questions, and so PII.

Batch and tests, for when a UI is not wanted:

```bash
python -m src.tester.run_batch      # all sample questions -> CSV (uses API quota)
python -m pytest src/tester -q      # tests: offline, no key needed
```

`run_batch.py` writes `src/tester/outputs/results_<timestamp>.csv`, one row per
question: the answer and citation alongside every signal behind them (retrieval
score, model confidence, whether the citation validated, escalation reason,
latency, any error). It exits non-zero if a graded question misses its expected
action, so it can gate a CI job. A committed example run is in
`src/tester/outputs/live_run.csv`.

## Example

Input:

```bash
python -m src.cli "My EMI auto debit failed last month. What charges will I have to pay?"
```

Output:

```json
{
  "query": "My EMI auto debit failed last month. What charges will I have to pay?",
  "category": "emi_and_payments",
  "answer": "A failed auto-debit (bounce) attracts a bounce charge of Rs. 500 per instance, plus applicable GST. This is separate from late payment interest.",
  "source": "FAQ-EMI-03 / What happens if my EMI payment bounces?",
  "confidence": "high",
  "action": "respond",
  "diagnostics": {
    "retrieval_score": 0.575,
    "retrieved": [
      {
        "citation": "FAQ-EMI-03 / What happens if my EMI payment bounces?",
        "score": 0.575
      },
      {
        "citation": "FAQ-EMI-03 / How do I register or replace a NACH mandate?",
        "score": 0.363
      },
      {
        "citation": "FAQ-EMI-03 / Can I change my EMI due date?",
        "score": 0.349
      },
      {
        "citation": "FAQ-EMI-03 / Is there a grace period?",
        "score": 0.291
      }
    ],
    "retrieval_confidence": "high",
    "model_confidence": "high",
    "model_grounded": true,
    "citation_valid": true,
    "insufficient_information": false,
    "provider": "gemini",
    "latency_ms": 6020
  }
}
```

The six top-level fields are the required contract. `diagnostics` is additive,
and exists so the escalation decision can be audited without enabling logging.

## 1. How does it work?

```
question -> RETRIEVE          keyword scoring over policy sections
                              top 4 sections; below a floor, escalate now
         -> GENERATE          one Gemini call, grounded only on those sections
         -> VERIFY            retrieval relevant? model grounded? citation real?
                              any failure -> low confidence -> escalate
         -> structured JSON
```

1. The three policy PDFs in `data/policies/` are split into one chunk per
   section, each carrying a stable citation like `POL-PREPAY-01 / Foreclosure charges`.
   `load_chunks()` is the only code that knows the documents are PDFs; everything
   downstream works on chunks, which is why swapping the format touched one function.
2. Retrieval is an IDF-weighted keyword match with a small customer-vocabulary
   synonym map. It scores each chunk 0–1 by how much of the question's
   information content it covers, and returns the top 3.
3. If nothing clears a minimum relevance floor, the agent escalates without
   calling the model at all.
4. Otherwise the top chunks go to Gemini with a grounding-only system prompt and
   a response schema, so the model returns typed JSON rather than prose. Transient
   failures (overload, rate limit) are retried with backoff, honouring the
   server's own retry hint; permanent ones are not retried.
5. Three signals then have to agree before the agent responds: retrieval found
   something relevant, the model reported the excerpts sufficient and grounded,
   and the citation it returned is one we actually supplied. Any failure
   downgrades confidence, and low confidence escalates.

## 2. Why this model / approach?

Gemini Flash has a genuinely free tier and native structured output, which
removes the usual JSON-parsing fragility. Retrieval is keyword-based on purpose:
the corpus is three small documents in the same vocabulary customers use, so
embeddings would add an index, a dependency and an opaque score for no accuracy
gain at this size — and every score here is explainable. The LLM sits behind a
one-function interface, so swapping providers is a single-file change.

## 3. What I would improve before production

- **PDF ingestion**: section headings are found by an explicit mark that
  `tools/make_pdfs.py` writes, which works because we generate the documents.
  Third-party PDFs need heading detection from font size and position
  (pdfplumber), plus handling for scans, multi-column layouts and tables.
- **Retrieval**: keyword matching fails on paraphrase with no shared vocabulary.
  Add embeddings with the keyword score retained as a hybrid signal and a
  reranker, once the corpus outgrows a few dozen chunks.
- **Evaluation**: thresholds here were tuned on five questions, which is not
  evidence. Needs a labelled set with adversarial and out-of-scope cases, and
  regression runs gating every prompt change.
- **PII handling**: `logs/query_log.csv` currently stores questions verbatim.
  Production needs redaction of account and phone numbers before both the prompt
  and the log, plus a retention policy on the log itself.
- **Policy versioning**: documents are read from disk at startup. Production
  needs versioned documents, an effective-date check, and answers pinned to the
  version that produced them.
- **Escalation path**: escalation currently returns a message. It should create a
  ticket with the retrieved context attached, and feed resolved escalations back
  as evaluation cases.
- **Throughput**: answers are generated one call at a time against a free-tier
  quota. Production needs a paid tier, a response cache for repeated questions,
  and a queue rather than per-request pacing.

## 4. One security / governance concern in financial services

Grounded answers about charges, foreclosure terms and KYC are effectively
regulated communication, so the binding requirement is auditability: for any
answer given to a customer, the firm must be able to reproduce which policy
version, which excerpt, and which model produced it. That is why every response
here carries an exact citation and its retrieval diagnostics, and why the agent
escalates rather than answering partially. The paired concern is data boundary —
customer questions carry PII, so redaction before the prompt and exclusion of raw
payloads from logs are prerequisites, not enhancements.

## 5. AI coding tools used

Built with Claude (Claude Code) in a single session. I used it to draft the
synthetic policy documents, scaffold the modules, and write the test suite, then
reviewed and corrected its output — the retrieval scoring metric, the
three-signal confidence rule and the threshold values were design decisions I
made and then had it implement. I verified behaviour by running the samples and
the failure-case tests rather than trusting the generated code as written.

## Notes

- `data/policies/` contains **synthetic** documents written for this exercise.
  They are not real policies of any company.
- Out of scope by design, per the brief: no auth, database, vector store,
  deployment or UI.

## Layout

```
app.py                      local web app (Flask)

src/backend/                the agent
  config.py                 thresholds and settings, all in one place
  retrieval.py              chunking, IDF index, synonym map
  llm.py                    the Gemini call, behind one function
  main.py                   the workflow: contract, confidence, escalation
  query_log.py              appends every answered question to a CSV

src/frontend/               the UI
  templates/index.html
  static/style.css

src/tester/                 tests, split by what they cover
  test_retrieval.py         does keyword search find the right section?
  test_llm_connection.py    provider, prompt, schema, retry classification
  test_agent.py             end-to-end workflow and every failure case
  test_query_log.py         the usage log, including failing safely
  run_batch.py              batch runner
  outputs/                  CSV results

data/policies/              3 synthetic policy PDFs (what the agent reads)
data/source/                the Markdown those PDFs are generated from
data/questions.json         5 sample questions with expected actions
tools/make_pdfs.py          regenerates the PDFs from the Markdown
logs/query_log.csv          every question asked through the UI (gitignored)
```
