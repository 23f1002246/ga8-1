"""
Combined deterministic MLOps policy service, 7 endpoints:
  POST /build-corpus      Q1  Immutable, leakage-safe training corpus
  POST /bqml               Q2  BigQuery ML two-phase experiment gate
  POST /promote             Q3  MLflow model promotion gate
  POST /adapt               Q4  PEFT intervention choice + run repair
  POST /quantize            Q5  Quantization freeze/select gate
  POST /pipeline             Q6  Content-addressed pipeline controller
  POST /verify-bundle        Q7  Model bundle + model card verifier

All logic is pure-Python and deterministic; Q2/Q5/Q6 keep simple in-memory
state (resets if the process restarts).
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from q1_corpus import build_corpus
from q2_bqml import bqml
from q3_promote import promote
from q4_adapt import adapt
from q5_quantize import quantize
from q6_pipeline import pipeline
from q7_verify_bundle import verify_bundle

app = FastAPI()


async def _json_or_none(request: Request):
    try:
        return await request.json()
    except Exception:
        return None


@app.get("/")
async def root():
    return {"ok": True, "endpoints": [
        "/build-corpus", "/bqml", "/promote", "/adapt", "/quantize", "/pipeline", "/verify-bundle"
    ]}


@app.post("/build-corpus")
async def route_build_corpus(request: Request):
    body = await _json_or_none(request)
    status, resp = build_corpus(body if body is not None else {})
    return JSONResponse(resp, status_code=status)


@app.post("/bqml")
async def route_bqml(request: Request):
    body = await _json_or_none(request)
    status, resp = bqml(body if body is not None else {})
    return JSONResponse(resp, status_code=status)


@app.post("/promote")
async def route_promote(request: Request):
    body = await _json_or_none(request)
    status, resp = promote(body if body is not None else {})
    return JSONResponse(resp, status_code=status)


@app.post("/adapt")
async def route_adapt(request: Request):
    body = await _json_or_none(request)
    status, resp = adapt(body if body is not None else {})
    return JSONResponse(resp, status_code=status)


@app.post("/quantize")
async def route_quantize(request: Request):
    body = await _json_or_none(request)
    status, resp = quantize(body if body is not None else {})
    return JSONResponse(resp, status_code=status)


@app.post("/pipeline")
async def route_pipeline(request: Request):
    body = await _json_or_none(request)
    status, resp = pipeline(body if body is not None else {})
    return JSONResponse(resp, status_code=status)


@app.post("/verify-bundle")
async def route_verify_bundle(request: Request):
    body = await _json_or_none(request)
    status, resp = verify_bundle(body if body is not None else {})
    return JSONResponse(resp, status_code=status)
