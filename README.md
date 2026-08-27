# TDS MLOps Policy Services (Q1-Q7)

One FastAPI app, seven deterministic endpoints. See main.py for the route list.

## Run locally
    pip install -r requirements.txt
    uvicorn main:app --reload

## Deploy
Push to a public repo, deploy as a Docker web service (Render, etc).
Submit the SAME base URL for every one of Q1-Q7 — each question's grader
hits its own specific path on that URL.

## Known limitations / where to expect debugging
These are extremely detailed specs (15-30+ precise rules each). This is a
careful best-faithful implementation, smoke-tested against hand-built cases
including several exact worked examples from the spec text itself, but full
compliance with every edge case across 7 endpoints this complex should be
expected to need iteration against the real grader - paste back any specific
mismatch and it can be fixed quickly, the same way we debugged earlier
GA services together.

Q2, Q5, Q6 use simple in-memory state (dict) for cross-request persistence
(runId / freezeId / session). This resets if the server process restarts -
fine for grading in one sitting, but not durable across a redeploy.
