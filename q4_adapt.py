import re
from mlops_utils import is_finite_number, is_safe_nonneg_int, is_positive_safe_int, sort_dedupe_codes, utf8_sort_key

PRIORITY = ["prompt_only", "retrieval", "lora", "qlora"]


def _choose(body):
    policy = body.get("policy")
    candidates = body.get("candidates")
    if not isinstance(policy, dict) or not isinstance(candidates, list):
        return 400, {"error": "INVALID_INPUT"}

    by_name = {c.get("name"): c for c in candidates if isinstance(c, dict)}
    if set(by_name.keys()) != set(PRIORITY):
        return 400, {"error": "INVALID_INPUT"}

    total_costs = {}
    reason_codes = {name: [] for name in PRIORITY}
    eligible = []

    for name in PRIORITY:
        c = by_name[name]
        codes = []
        if c.get("available") is not True:
            codes.append("UNAVAILABLE")
        q = c.get("quality")
        if not (is_finite_number(q) and q >= policy.get("minQuality", 0)):
            codes.append("QUALITY_FLOOR")
        if policy.get("freshnessRequired") is True and c.get("freshness") is not True:
            codes.append("FRESHNESS_REQUIRED")
        lat = c.get("latencyMs")
        if is_finite_number(lat) and lat > policy.get("maxLatencyMs", float("inf")):
            codes.append("LATENCY_LIMIT")
        mem = c.get("memoryMb")
        if is_finite_number(mem) and mem > policy.get("maxMemoryMb", float("inf")):
            codes.append("MEMORY_LIMIT")
        labeled = c.get("labeledExamples")
        if is_safe_nonneg_int(labeled) and labeled > policy.get("maxLabeledExamples", float("inf")):
            codes.append("DATA_LIMIT")

        one_time = c.get("oneTimeCost", 0)
        recurring = c.get("recurringCost", 0)
        horizon = policy.get("horizonRequests", 0)
        total_cost = round(one_time + horizon * recurring, 12)
        total_costs[name] = total_cost
        if is_finite_number(total_cost) and total_cost > policy.get("maxTotalCost", float("inf")):
            codes.append("COST_LIMIT")

        reason_codes[name] = sort_dedupe_codes(codes)
        if not codes:
            eligible.append(name)

    selected = eligible[0] if eligible else None
    return 200, {"selected": selected, "eligible": eligible, "totalCosts": total_costs, "reasonCodes": reason_codes}


def _repair(body):
    codes = []

    # --- tokens / labels ---
    tokens = body.get("tokens")
    tokens_valid = isinstance(tokens, list) and len(tokens) > 0
    labels = []
    if tokens_valid:
        for t in tokens:
            if not isinstance(t, dict) or not is_safe_nonneg_int(t.get("id")) \
                    or t.get("role") not in ("system", "user", "assistant") \
                    or not isinstance(t.get("padding"), bool) or not isinstance(t.get("text"), str):
                tokens_valid = False
                break
    if not tokens_valid:
        codes.append("INVALID_TOKEN")
        labels = [-100] * (len(tokens) if isinstance(tokens, list) else 0)
    else:
        for t in tokens:
            if t["role"] == "assistant" and t["padding"] is False:
                labels.append(t["id"])
            else:
                labels.append(-100)

    # --- template applications ---
    template_pass = body.get("templateApplications") == 1

    # --- PEFT parameters ---
    params = body.get("parameters")
    allowed_targets = body.get("allowedTargets")
    params_valid = (
        isinstance(params, list)
        and isinstance(allowed_targets, list) and len(allowed_targets) > 0
        and len(set(allowed_targets)) == len(allowed_targets)
        and all(isinstance(x, str) and x != "" for x in allowed_targets)
    )
    trainable_params = []
    trainable_count = 0
    if params_valid:
        # Every parameter must have a unique string name and a positive safe-int numel.
        names_seen = set()
        for p in params:
            if not isinstance(p, dict) or not isinstance(p.get("name"), str) or p["name"] == "" \
                    or p["name"] in names_seen or not is_positive_safe_int(p.get("numel")):
                params_valid = False
                break
            names_seen.add(p["name"])
        if params_valid:
            # LoRA-trainable = allowed target AND name ends with lora_A/lora_B weight.
            lora_params = [p for p in params
                            if p.get("target") in allowed_targets
                            and (p["name"].endswith(".lora_A.weight") or p["name"].endswith(".lora_B.weight"))]
            if len(lora_params) == 0:
                # No trainable LoRA parameter present -> invalid PEFT config.
                params_valid = False
            else:
                lora_params.sort(key=lambda p: utf8_sort_key(p["name"]))
                trainable_params = [p["name"] for p in lora_params]
                total = 0
                for p in lora_params:
                    total += p["numel"]
                # "safely sum": result must stay a safe integer
                if total > (2**53 - 1):
                    params_valid = False
                    trainable_params = []
                    trainable_count = 0
                else:
                    trainable_count = total
    if not params_valid:
        codes.append("INVALID_PARAMETER")
        trainable_params = []
        trainable_count = 0

    peft_config_pass = params_valid and template_pass
    if not template_pass:
        codes.append("CHAT_TEMPLATE_COUNT")

    # --- inference mode / dropout ---
    if body.get("inferenceMode") is not False:
        codes.append("INFERENCE_MODE")
    if body.get("dropoutActiveDuringEval") is not False:
        codes.append("EVAL_DROPOUT_ACTIVE")

    # --- train/eval isolation ---
    train_ids = body.get("trainRowIds")
    eval_ids = body.get("evalRowIds")
    eval_isolated = (
        isinstance(train_ids, list) and isinstance(eval_ids, list)
        and len(train_ids) > 0 and len(eval_ids) > 0
        and all(isinstance(x, str) and x != "" for x in train_ids + eval_ids)
        and len(set(train_ids)) == len(train_ids) and len(set(eval_ids)) == len(eval_ids)
        and len(set(train_ids) & set(eval_ids)) == 0
    )
    if not eval_isolated:
        codes.append("EVAL_LEAKAGE")

    # --- artifact files ---
    artifact_files = body.get("artifactFiles")
    expected_files = {"adapter_config.json", "adapter_model.safetensors"}
    adapter_files_ok = isinstance(artifact_files, list) and set(artifact_files) == expected_files \
                       and len(artifact_files) == 2
    if not adapter_files_ok:
        if isinstance(artifact_files, list) and any(f not in expected_files for f in artifact_files):
            codes.append("FULL_MODEL_ARTIFACT")
        codes.append("ADAPTER_FILE_SET")
    adapter_files_out = sorted(expected_files, key=utf8_sort_key) if adapter_files_ok else []

    # --- lineage ---
    base_rev = body.get("baseRevision")
    dataset_digest = body.get("datasetDigest")
    code_digest = body.get("codeDigest")
    config_digest = body.get("configDigest")
    lineage_pass = (
        isinstance(base_rev, str) and re.fullmatch(r"[0-9a-f]{40}", base_rev) is not None
        and all(isinstance(d, str) and re.fullmatch(r"[0-9a-f]{64}", d) is not None
                for d in (dataset_digest, code_digest, config_digest))
    )
    if not lineage_pass:
        codes.append("MUTABLE_BASE_REVISION")

    expected_digests = body.get("expectedDigests", {})
    if isinstance(expected_digests, dict) and expected_digests:
        for k, v in expected_digests.items():
            actual = {"datasetDigest": dataset_digest, "codeDigest": code_digest, "configDigest": config_digest}.get(k)
            if actual != v:
                codes.append("LINEAGE_MISMATCH")
                break

    # --- batch math ---
    mb = body.get("microBatch")
    ga = body.get("gradientAccumulation")
    reps = body.get("replicas")
    expected_batch = body.get("expectedEffectiveBatch")
    batch_ok = (
        is_positive_safe_int(mb) and is_positive_safe_int(ga) and is_positive_safe_int(reps)
        and is_positive_safe_int(expected_batch) and mb * ga * reps == expected_batch
    )
    if not batch_ok:
        codes.append("EFFECTIVE_BATCH_MISMATCH")

    # --- checkpoint ---
    checkpoint = body.get("checkpoint")
    required_ckpt_keys = {"model", "optimizer", "scheduler", "step", "rng", "dataPosition"}
    checkpoint_complete = isinstance(checkpoint, dict) and required_ckpt_keys <= set(checkpoint.keys())
    if not checkpoint_complete:
        codes.append("INCOMPLETE_CHECKPOINT")

    # --- resume ---
    uw = body.get("uninterruptedWeights")
    rw = body.get("resumedWeights")
    tol = body.get("resumeTolerance")
    resume_pass = (
        isinstance(uw, list) and isinstance(rw, list) and len(uw) > 0 and len(uw) == len(rw)
        and all(is_finite_number(x) for x in uw) and all(is_finite_number(x) for x in rw)
        and is_finite_number(tol) and tol >= 0
    )
    if resume_pass:
        resume_pass = all(abs(a - b) <= tol for a, b in zip(uw, rw))
    if not resume_pass:
        codes.append("RESUME_DIVERGENCE")

    codes = sort_dedupe_codes(codes)

    return 200, {
        "labels": labels,
        "templatePass": template_pass,
        "trainableParams": trainable_params,
        "trainableCount": trainable_count,
        "peftConfigPass": peft_config_pass,
        "adapterFiles": adapter_files_out,
        "checkpointComplete": checkpoint_complete,
        "lineagePass": lineage_pass,
        "evalIsolated": eval_isolated,
        "evaluationDeterministic": body.get("dropoutActiveDuringEval") is False,
        "resumePass": resume_pass,
        "reasonCodes": codes,
    }


def adapt(body):
    if not isinstance(body, dict):
        return 400, {"error": "INVALID_INPUT"}
    op = body.get("operation")
    if op == "choose":
        return _choose(body)
    elif op == "repair":
        return _repair(body)
    return 400, {"error": "INVALID_INPUT"}
