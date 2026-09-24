# Policy-Aware Customer Support Agent

This is a small support agent that answers customer questions about loans, KYC
and EMI payments strictly from three policy documents, returns a structured
result, and hands the question to a human whenever the documents do not support
a confident answer. It runs as a local web app, and every question it is asked
is recorded to a CSV with the reasoning behind the answer.

## 1. How does it work?

A question first goes to a keyword search over the policy PDFs, which are split
into one chunk per numbered section, each carrying a citation such as
`POL-PREPAY-01 / Foreclosure charges`. An IDF-weighted score rates every chunk
from 0 to 1 on how much of the question it covers; the top four go forward, and
if nothing clears a relevance floor the agent escalates without calling the
model at all. Those sections go to Gemini with a grounding-only prompt and a
JSON response schema. Three signals must then agree before an answer is
returned: retrieval found something relevant, the model reported the excerpts
sufficient and grounded, and the citation it returned is one I actually supplied.
Any failure drops confidence to low, and low confidence escalates.

## Example

Asked through the UI: *"My EMI auto debit failed last month. What charges will I
have to pay?"*

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
    "retrieval_confidence": "high",
    "model_confidence": "high",
    "model_grounded": true,
    "citation_valid": true,
    "insufficient_information": false,
    "model": "gemini-3.6-flash",
    "latency_ms": 6020
  }
}
```

The six top-level fields are the contract. `diagnostics` is additive, so the
escalation decision can be audited without turning on logging.

For contrast, *"what interest rate will I get on a two wheeler loan in Pune?"*
returns `action: escalate` with an empty source — no policy covers pricing, and
the agent is built to say so rather than guess.

## 2. Why this model / approach?

I chose Gemini Flash for its free tier and native structured output, which
removes most of the usual JSON-parsing fragility. Retrieval is keyword-based on
purpose: three small documents written in the customer's own vocabulary do not
need embeddings, and every score stays inspectable, which matters when someone
asks why a customer was told something. The model sits behind a single function,
so changing provider is a one-file change.

## 3. What I would improve before production

- **Retrieval.** Keyword matching misses paraphrase with no shared words. Add
  embeddings as a hybrid signal with a reranker.
- **Evaluation.** Thresholds were tuned on five questions, not measured. Needs a
  labelled set with adversarial and out-of-scope cases, gating prompt changes.
- **PII.** Questions are logged verbatim. Redact account and phone numbers before
  both the prompt and the log, with a retention policy.
- **Policy versioning.** Documents load from disk at startup. Needs versions,
  effective-date checks, and answers pinned to the version that produced them.
- **Escalation.** Currently returns a message. Should open a ticket carrying the
  retrieved context, and feed resolved cases back as evaluation data.

## 4. One security or governance concern in financial services

Answers about charges, foreclosure terms and KYC are regulated communication, so
the binding requirement is auditability: for any answer given to a customer, the
firm must be able to reproduce which policy version and which
model produced it. That is why every response carries an exact citation and its
retrieval diagnostics, and why the agent escalates rather than answering
partially — an unverifiable answer is a worse outcome than no answer.

## 5. What AI coding tools I used

I built this with Claude Code in a single session. It drafted the synthetic
policy documents, scaffolded the modules and wrote the test suite. The design
decisions were mine: the scoring metric, the three-signal confidence rule and
the thresholds. I reviewed and corrected what it produced rather than accepting
it, and verified behaviour by running the samples and the failure-case tests.

## Beyond the time box

Identified but deliberately not built inside the time box.

**Engineering**

- **Version every prompt.** Log which prompt version and model produced each
  answer; otherwise a prompt edit silently changes answers already given.
- **Ask for proof.** Make the model quote the sentence it used, and reject the
  answer if that quote is not verbatim in the excerpt.
- **Model governance.** Use a model on the security team's approved list, prefer
  Indian data residency over a global endpoint per RBI localisation, and pin an
  exact model version so a provider-side upgrade cannot change answers.
- **Database for run bookkeeping.** Questions, answers, citations, confidence,
  escalations — for quality trends, audit on demand, and a real evaluation set.
- **API in front of it.** So the support console, WhatsApp or IVR can call it,
  separating the interface from the reasoning.

**Further agents**

- **Escalation agent.** Emails the right team with the question, the sections
  retrieved, and why it declined to answer.
- **Policy change watcher.** Re-runs past questions when a document changes and
  flags customers told something now out of date. I would build this first — it
  closes the auditability loop rather than only recording it.
- **Grievance triage.** Categorises and routes complaints, tracks the 30-day SLA.
- **Internal copilot.** Drafts a cited answer for a human to send. Lower risk,
  and the sensible first production deployment.
- **KYC document assistant.** Tells the customer which document is missing, or
  why theirs was rejected.

## Running it

Python 3.9 or newer. `pip install -r requirements.txt`, then a `.env` file in
the project root holding `GEMINI_API_KEY=...`.

```bash
streamlit run streamlit_app.py      # the support chat UI
python -m src.tester.run_batch      # all sample questions -> CSV
python -m pytest src/tester -q      # tests: offline, no key needed
```

Deployed on Streamlit Community Cloud, the key goes in the app's Secrets as
`GEMINI_API_KEY` instead of a `.env` file.

A key is required at runtime; there is no offline fallback. The tests are the
exception, because they patch the model call with their own stub.

## Known limits

- The policy documents are **synthetic**, written for this exercise. They are
  not real policies of any company.
- Sections are found by their numbering, which works because I author these
  documents. Third-party PDFs would need heading detection from font size and
  position, plus handling for scans and multi-column layouts.
- Gemini's free tier allows 5 requests per minute and 20 per day per model, so
  the batch runner paces itself. Pointing `GEMINI_MODEL` at another model gives
  a fresh daily budget.
- Deliberately out of scope, per the brief: no auth, database, vector store or
  deployment. The UI is intentionally minimal.

## Layout

```
streamlit_app.py            entry point: wires the agent to the UI

src/backend/                the agent
  config.py                 thresholds and settings, all in one place
  retrieval.py              PDF parsing, IDF index, synonym map
  llm.py                    the Gemini call, behind one function
  main.py                   the workflow: contract, confidence, escalation
  query_log.py              appends every answered question to a CSV

src/frontend/               the UI
  ui.py                     the support chat, in plain customer language

src/tester/                 tests, split by what they cover
  test_retrieval.py         does keyword search find the right section?
  test_llm_connection.py    prompt, schema, retry and error classification
  test_agent.py             the workflow end to end, and every failure case
  test_query_log.py         the usage log, including failing safely
  run_batch.py              runs every sample question, writes a CSV
  outputs/                  CSV results

data/policies/              3 synthetic policy PDFs (what the agent reads)
data/questions.json         10 test questions, 6 answerable and 4 escalating
logs/query_log.csv          every question asked through the UI (gitignored)
```
