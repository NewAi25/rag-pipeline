# Deploying the live demo

This guide walks you through deploying the RAG Pipeline as a **read-only
public demo** anyone can try in a browser. Two targets are covered:

1. **Hugging Face Spaces (recommended)** — Docker SDK, fully matches the
   repo's existing Dockerfile, free CPU tier is enough for this workload.
2. **Streamlit Community Cloud (fallback)** — no Docker; uses
   `requirements.txt` directly.

The demo runs `app.py` with `DEMO_MODE=1`, which:

- disables PDF upload,
- auto-ingests `data/nist_ai_rmf_1.0.pdf` on first request,
- caps each browser session at **20 questions** (configurable),
- shows a banner explaining the shared-key rate limits.

---

## Prerequisites (one-time, no matter which target)

You'll need:

- A Gemini API key — get one **free** at
  [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
  This will be the single key that powers the public demo, so:
  - Use a **dedicated key**, not the one you use for development.
  - Be aware that any visitor's question consumes from its free-tier quota.
- The repo pushed to GitHub (already done at
  [github.com/NewAi25/rag-pipeline](https://github.com/NewAi25/rag-pipeline)).
- The NIST PDF in `data/` (already in the repo).

---

## Option 1 — Hugging Face Spaces (recommended)

### Step 1 — Create a Hugging Face account

1. Go to [huggingface.co/join](https://huggingface.co/join) and sign up
   (free). Confirm your email.
2. (Optional) Create an organization if you want the Space under an org
   name rather than your personal account.

> **You must do this step yourself** — I can't create the account for you.

### Step 2 — Create a new Space

1. Click your avatar (top-right) → **New Space**.
2. Fill in:
   - **Space name:** `rag-pipeline-demo` (or anything you like).
   - **License:** MIT.
   - **Select the SDK:** **Docker**, then **Blank** template.
   - **Hardware:** CPU basic (the free tier — enough for this workload).
   - **Visibility:** Public.
3. Click **Create Space**.

You'll land on the Space's page with an empty git repo on the right.

### Step 3 — Add your Gemini key as a secret

1. On the Space page → **Settings** tab.
2. Scroll to **Variables and secrets** → **New secret**.
3. **Name:** `GEMINI_API_KEY`
4. **Value:** your free Gemini key from `aistudio.google.com/apikey`.
5. Click **Save**.

While you're there, add the following **public variables** (Variables,
not Secrets — they're visible but not sensitive):

| Name | Value | Why |
|------|-------|-----|
| `DEMO_MODE` | `1` | Enables the read-only demo behavior in `app.py`. |
| `RETRIEVAL_MODE` | `hybrid` | Use the BM25+vector hybrid for the public demo. |
| `CHROMA_DIR` | `/data/chroma_db` | If you enabled persistent storage; otherwise leave `/app/chroma_db` (the Dockerfile default). |
| `ANONYMIZED_TELEMETRY` | `False` | Silence Chroma's posthog noise in logs. |

> **You must enter the secret yourself** — I can't see or write your key.

### Step 4 — Push the repo into the Space

The Space gave you a git URL like:
```
https://huggingface.co/spaces/<your-username>/rag-pipeline-demo
```

From a terminal in this repo (`c:/Rag-project/Rag-Pipeline`):

```bash
# Add the Space as a second remote (your existing `origin` -> GitHub stays)
git remote add hf https://huggingface.co/spaces/<your-username>/rag-pipeline-demo

# Copy the Space's README (with the YAML frontmatter HF needs) over the
# project README *for this push only*. We undo it right after.
cp huggingface-space/README.md README.hf.md
git stash --include-untracked   # set aside the helper file
git checkout -b hf-deploy
cp huggingface-space/README.md README.md
git add README.md
git commit -m "deploy: HF Space README with frontmatter"

# Push to the Space
git push hf hf-deploy:main

# Switch back to main locally
git checkout main
git branch -D hf-deploy
git stash pop   # restore the helper file
```

Hugging Face will build the Docker image (first build ~3–5 min). When
it finishes, the Space page shows **"Running"** and the app is live at:
```
https://<your-username>-rag-pipeline-demo.hf.space
```

### Step 5 — Test it

1. Open the Space URL.
2. Wait for the "Preparing the index" spinner (~30 s on first visit;
   this is the one-time PDF ingest using your API key).
3. Ask: *"What are the four functions of the AI RMF Core?"*
4. You should see a grounded answer citing `nist_ai_rmf_1.0.pdf#chunk-N`.

### Step 6 — Tell me the URL

Once it's live, send me the public URL and I'll add a **🔴 Live demo**
badge to the top of the GitHub README.

---

## Option 2 — Streamlit Community Cloud (fallback)

Use this if you don't want a Hugging Face account. **Note:** Streamlit
Cloud runs `pip install -r requirements.txt` — no Docker — so the build
is slightly less reproducible.

### Steps

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
   with the **GitHub account that owns the repo** (NewAi25).
2. Click **New app**.
3. **Repository:** `NewAi25/rag-pipeline`
   **Branch:** `main`
   **Main file path:** `app.py`
4. **Advanced settings → Python version:** 3.12.
5. **Advanced settings → Secrets** — paste this TOML, replacing the key:
   ```toml
   GEMINI_API_KEY = "your-real-key-here"
   DEMO_MODE = "1"
   RETRIEVAL_MODE = "hybrid"
   ANONYMIZED_TELEMETRY = "False"
   ```
6. Click **Deploy**.

First build: 3–5 minutes. When done, you'll get a `*.streamlit.app` URL.

### Streamlit Cloud caveats

- No persistent storage — every cold start re-ingests the PDF (~30 s
  and a few embedding API calls). That's fine for a low-traffic demo.
- Sleeps after inactivity. First visitor after a sleep waits longer.
- If you run into 429s during ingest, lower `CHUNK_SIZE_TOKENS` and
  re-deploy, or just wait — the retry helper handles transient quota.

---

## What I prepared for you (in this repo)

- [app.py](app.py) — picks up `DEMO_MODE=1` automatically; no code edits
  needed for deployment.
- [huggingface-space/README.md](huggingface-space/README.md) — the YAML-
  frontmatter README HF Spaces needs at the Space root.
- [Dockerfile](Dockerfile) — already exposes port 8501 implicitly; the
  Space config (`app_port: 8501`) tells HF to route traffic there.
- [.env.example](.env.example) — documents every variable the demo
  reads.

## What you must do yourself

- Sign up for Hugging Face (or Streamlit Cloud).
- Create the Space / app.
- Paste the API key into the Secrets panel — I can't see or write it.
- Click **Deploy**.
- Send me the URL when it's live so I can add the badge to the README.
