# Resume Job Fit Scorer

A local-first CLI that scores how well a **resume** matches a **job description**, then shows the matches, the gaps, and a short explanation.

The default model path is [Ollama](https://ollama.com) on your machine. No cloud API key is required. Documents stay on localhost unless you opt into an OpenAI-compatible endpoint.

Built by Alfredo Cardona ([SilverBomb-Gaming](https://github.com/SilverBomb-Gaming)).

The sample candidate, **Jordan Hale**, is fictional. This is not the author's resume.

## What it is / isn't

**It is** a portfolio demo of a careful LLM evaluation loop:

1. Read a rubric.
2. Ask a model to extract concrete requirements from the job description.
3. Ask the model to compare those requirements to the resume, with quotes.
4. Compute a 0–100 score in code from that evidence.

**It isn't** an applicant tracking system, a hosted product, or a resume writer that may invent employers, tools, or years. A gap stays a gap when the resume is silent. Suggested bullets only rephrase facts the resume already states, and each one names the fact it came from so you can check it.

## 10-minute demo

You need Python 3.11+ and [Ollama](https://ollama.com) with a chat model. `llama3.2` is the default.

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .

ollama pull llama3.2
```

Score the same fictional automation resume against two postings:

```bash
fit-score score --resume samples/resume.md --jd samples/jd-strong-fit.md
fit-score score --resume samples/resume.md --jd samples/jd-weak-fit.md
```

What you should see:

- **Strong fit** (`samples/jd-strong-fit.md`, Harborline automation engineer): a high score. Matches should name things the resume actually says: Python, pytest, hardware-in-the-loop, Modbus/TCP, Linux, GitLab CI, SQL, about six years in manufacturing test, an EE degree.
- **Weak fit** (`samples/jd-weak-fit.md`, Brightpath senior iOS engineer): a clearly lower score. Gaps should name Swift, SwiftUI, UIKit, Xcode, TestFlight, the App Store, and Core Data. Those words are not on the resume.

The first run can sit for a bit while the model loads. Each score is two local chat calls (extract, then judge). Progress is printed on stderr. The report itself is stdout.

Machine-readable report:

```bash
fit-score score \
  --resume samples/resume.md \
  --jd samples/jd-strong-fit.md \
  --json
```

Other input shapes:

```bash
# Inline text
fit-score score --resume samples/resume.md --jd-text "Python and pytest required. Modbus/TCP required."

# Pipe whichever document you did not pass as a flag
cat samples/jd-weak-fit.md | fit-score score --resume samples/resume.md

# Same rubric the CLI uses when you omit --rubric
fit-score score --resume samples/resume.md --jd samples/jd-strong-fit.md --rubric samples/rubric.yaml
```

Judgments depend on the model. The number does not: once the matched / partial / missing calls are fixed, the score is arithmetic. Read the `Resume:` quote under each match. If a line looks invented, the quote is how you tell.

## How scoring works

```text
job description ──► extract requirements ──► compare to resume ──► score + gaps
                         (rubric)                  (quotes)         (this program)
```

1. **Rubric.** With no `--rubric`, the built-in rubric asks for four kinds of evidence from the posting: requirements, skills, years, and tools. A YAML or JSON file can replace the categories and the weights. `samples/rubric.yaml` is that default, written out so you can edit it.
2. **Extract.** The model lists concrete requirements. Each item is `must` or `nice` and includes a short quote from the job description (`jd_evidence`). Items with no quote are dropped. Must-haves are kept ahead of nice-to-haves, and the list is capped at 16.
3. **Compare.** The model judges each requirement index as `matched`, `partial`, or `missing` and must quote the resume for anything other than a gap. It also returns short "why" bullets and 2–4 rewrite suggestions.
4. **Guard the evidence in code**, not only in the prompt.
   - A `matched` or `partial` judgment with an empty resume quote is downgraded to a gap.
   - A missing index is a gap.
   - A `missing` judgment does not keep a resume quote.
   - Rewrites without a `grounded_in` fact are dropped.
5. **Score.** The model never picks the 0–100 number.

Default credit: matched = 1, partial = 0.5, missing = 0. Default weights: must-have coverage 0.75, nice-to-have coverage 0.25.

```text
must_ratio  = average credit across must-have requirements
nice_ratio  = average credit across nice-to-have requirements
score       = 100 * (0.75 * must_ratio + 0.25 * nice_ratio) / (0.75 + 0.25)
```

The result is rounded half-up and clamped to 0–100. If the posting produces only must-haves, or only nice-to-haves, that side is the whole score. The report's `Score math:` line shows the substituted formula.

The system prompt forbids fabricating employers and skills, forbids inventing experience, and tells the model: if the resume is silent, mark it as a gap. Those sentences are pinned by tests in `tests/test_prompts.py`.

## Configuration

Copy `.env.example` to `.env` in the working directory, or export the variables yourself. Existing environment variables win over `.env`.

| Variable | Default | Role |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server |
| `OLLAMA_MODEL` | `llama3.2` | Chat model |
| `OLLAMA_NUM_CTX` | `8192` | Context window sent to Ollama |
| `FIT_SCORE_PROVIDER` | `ollama` | `ollama` or `openai` |
| `FIT_SCORE_TIMEOUT` | `120` | Seconds per model call |
| `OPENAI_API_KEY` | empty | Only for the OpenAI-compatible path |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Compatible base URL, usually ending in `/v1` |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model name for that path |

```bash
# Remote or local OpenAI-compatible server (LM Studio, a proxy, api.openai.com, …)
export FIT_SCORE_PROVIDER=openai
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini
fit-score score --resume samples/resume.md --jd samples/jd-strong-fit.md
```

`api.openai.com` refuses to run without `OPENAI_API_KEY`. A local compatible server may omit the key. `--provider` and `--model` override the environment for one command.

Inputs are UTF-8 `.txt` or `.md` (`.markdown` is accepted). PDF is out of scope.

## Scope / out of scope

**In scope**

- One resume against one job description
- Plain text and Markdown
- A transparent score, named gaps, and truthful bullet rewrites
- Ollama by default, OpenAI-compatible chat as an option

**Out of scope**

- PDF, DOCX, and LinkedIn import. Export to `.txt` or `.md` first.
- Ranking many candidates, or storing applications
- Inventing skills, employers, titles, dates, or years the resume does not state
- A guarantee that two models will mark the same line `matched`. The arithmetic is stable after the judgments are.

## Layout

```text
src/fit_score/
  cli.py        # fit-score score …
  prompts.py    # system prompt and the two task prompts
  scoring.py    # evidence rules and the 0–100 score
  rubric.py     # built-in rubric and YAML/JSON loading
  llm.py        # Ollama and OpenAI-compatible clients
  models.py     # Pydantic report schema
  render.py     # text report
  io.py         # .txt/.md loading and .env
samples/
  resume.md
  jd-strong-fit.md
  jd-weak-fit.md
  rubric.yaml
tests/          # pytest, no live model
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

`pytest` mocks the model client. It does not start Ollama and does not call the network.

Exit codes: `0` success, `1` bad input, `2` rubric or model failure.

## License

MIT © 2026 Alfredo Cardona
