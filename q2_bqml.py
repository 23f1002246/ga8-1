from mlops_utils import (
    parse_instant, sha256_hex, compact_json, is_safe_nonneg_int,
    is_positive_safe_int, is_finite_number, sort_dedupe_codes, utf8_sort_key,
)

# In-memory persistence keyed by runId. Resets on process restart.
_RUNS = {}


def _dedupe_rows(rows):
    groups = {}
    for r in rows:
        dt = parse_instant(r.get("eventTime"))
        key = (r.get("entity"), dt)
        groups.setdefault(key, []).append(r)
    out = []
    for key, group in groups.items():
        best = None
        for r in group:
            if best is None or r["version"] > best["version"] or \
               (r["version"] == best["version"] and utf8_sort_key(r["id"]) < utf8_sort_key(best["id"])):
                best = r
        out.append(best)
    return out


def _select(body):
    run_id = body.get("runId")
    if not (isinstance(run_id, str) and 0 < len(run_id) <= 128):
        return 400, {"error": "INVALID_INPUT"}

    rows = body.get("rows")
    trials = body.get("trials")
    limit = body.get("numTrialsLimit")
    forbidden = body.get("forbiddenFeatures", [])

    structurally_ok = (
        isinstance(rows, list) and len(rows) > 0
        and isinstance(trials, list)
        and is_positive_safe_int(limit)
        and isinstance(forbidden, list)
    )
    reason_codes = []
    if not structurally_ok:
        reason_codes.append("INVALID_INPUT")

    result = None
    if structurally_ok:
        ids_seen = set()
        rows_ok = True
        for r in rows:
            if not isinstance(r, dict) or r.get("id") in ids_seen:
                rows_ok = False
                break
            ids_seen.add(r.get("id"))
            if parse_instant(r.get("eventTime")) is None or parse_instant(r.get("predictionTime")) is None:
                rows_ok = False
                break
            if not is_safe_nonneg_int(r.get("version")):
                rows_ok = False
                break
            if r.get("split") not in ("TRAIN", "EVAL"):
                rows_ok = False
                break

        trial_ids_seen = set()
        trials_ok = True
        for t in trials:
            if not isinstance(t, dict) or t.get("trialId") in trial_ids_seen:
                trials_ok = False
                break
            trial_ids_seen.add(t.get("trialId"))
            if not is_safe_nonneg_int(t.get("trialId")):
                trials_ok = False
                break
            if t.get("status") not in ("SUCCEEDED", "FAILED"):
                trials_ok = False
                break

        if not (rows_ok and trials_ok):
            reason_codes.append("INVALID_INPUT")
        else:
            if len(trials) > limit:
                reason_codes.append("TRIAL_LIMIT_EXCEEDED")

            retained = _dedupe_rows(rows)
            train_rows = [r for r in retained if r["split"] == "TRAIN"]
            eval_rows = [r for r in retained if r["split"] == "EVAL"]

            all_feature_names = set()
            for r in retained:
                feats = r.get("features", {})
                if isinstance(feats, dict):
                    all_feature_names |= set(feats.keys())

            eligible_features = []
            for fname in all_feature_names:
                if fname in forbidden:
                    continue
                ok = True
                for r in retained:
                    feats = r.get("features", {})
                    if fname not in feats:
                        ok = False
                        break
                    fv = feats[fname]
                    avail = parse_instant(fv.get("availableAt")) if isinstance(fv, dict) else None
                    pred_t = parse_instant(r.get("predictionTime"))
                    if avail is None or pred_t is None or avail > pred_t:
                        ok = False
                        break
                if ok:
                    eligible_features.append(fname)
            eligible_features.sort(key=utf8_sort_key)

            eligible_trials = [t for t in trials if t["status"] == "SUCCEEDED" and is_finite_number(t.get("evalMetric"))]
            if not eligible_trials:
                reason_codes.append("NO_SUCCESSFUL_TRIAL")
                selected_trial = None
            else:
                selected_trial = eligible_trials[0]
                for t in eligible_trials[1:]:
                    if t["evalMetric"] > selected_trial["evalMetric"] or \
                       (t["evalMetric"] == selected_trial["evalMetric"] and t["trialId"] < selected_trial["trialId"]):
                        selected_trial = t

            train_ids = sorted([r["id"] for r in train_rows], key=utf8_sort_key)
            eval_ids = sorted([r["id"] for r in eval_rows], key=utf8_sort_key)

            digest_obj = {"trainRowIds": train_ids, "evalRowIds": eval_ids, "featureNames": eligible_features}
            dataset_digest = sha256_hex(compact_json(digest_obj).encode("utf-8"))

            reason_codes = sort_dedupe_codes(reason_codes)
            result = {
                "runId": run_id,
                "selectedTrialId": selected_trial["trialId"] if (selected_trial and not reason_codes) else None,
                "trainRowIds": train_ids,
                "evalRowIds": eval_ids,
                "featureNames": eligible_features,
                "datasetDigest": dataset_digest if not reason_codes else None,
                "reasonCodes": reason_codes,
            }

    if result is None:
        result = {
            "runId": run_id, "selectedTrialId": None, "trainRowIds": [], "evalRowIds": [],
            "featureNames": [], "datasetDigest": None,
            "reasonCodes": sort_dedupe_codes(reason_codes),
        }

    if run_id in _RUNS:
        stored = _RUNS[run_id]
        if stored["_input"] != body:
            return 409, {"error": "RUN_ID_CONFLICT"}
        return 200, stored["response"]

    _RUNS[run_id] = {"_input": body, "response": result, "selection_ok": len(result["reasonCodes"]) == 0,
                      "selected_trial_id": result["selectedTrialId"], "dataset_digest": result["datasetDigest"]}
    return 200, result


def _evaluate(body):
    run_id = body.get("runId")
    selected_trial_id = body.get("selectedTrialId")
    dataset_digest = body.get("datasetDigest")
    metric_floor = body.get("metricFloor")
    required_slices = body.get("requiredSlices")
    rows = body.get("rows")
    bytes_processed = body.get("bytesProcessed")
    max_bytes = body.get("maxBytes")

    def bad():
        return 200, {
            "runId": run_id, "selectedTrialId": selected_trial_id, "datasetDigest": dataset_digest,
            "testMetric": None, "criticalSlicePass": False, "decision": "reject",
            "bytesProcessed": bytes_processed if is_safe_nonneg_int(bytes_processed) else None,
            "reasonCodes": ["INVALID_INPUT"],
        }

    structurally_ok = (
        isinstance(run_id, str) and 0 < len(run_id) <= 128
        and is_safe_nonneg_int(selected_trial_id)
        and isinstance(dataset_digest, str) and len(dataset_digest) == 64
        and is_finite_number(metric_floor) and 0 <= metric_floor <= 1
        and isinstance(required_slices, dict)
        and isinstance(rows, list)
        and is_safe_nonneg_int(bytes_processed)
        and is_safe_nonneg_int(max_bytes)
    )
    if not structurally_ok:
        return bad()

    stored = _RUNS.get(run_id)
    lineage_ok = (
        stored is not None and stored.get("selection_ok")
        and stored.get("selected_trial_id") == selected_trial_id
        and stored.get("dataset_digest") == dataset_digest
    )
    if not lineage_ok:
        return 200, {
            "runId": run_id, "selectedTrialId": selected_trial_id, "datasetDigest": dataset_digest,
            "testMetric": None, "criticalSlicePass": False, "decision": "reject",
            "bytesProcessed": bytes_processed, "reasonCodes": ["INVALID_LINEAGE"],
        }

    codes = []
    rows_valid = True
    for r in rows:
        if not isinstance(r, dict) or r.get("label") not in (0, 1) or r.get("prediction") not in (0, 1) \
                or not (isinstance(r.get("slice"), str) and r.get("slice") != ""):
            rows_valid = False
            break
    if not rows_valid:
        codes.append("INVALID_TEST_ROW")

    byte_ok = bytes_processed <= max_bytes
    if not byte_ok:
        codes.append("BYTE_LIMIT")

    # criticalSlicePass is false for: invalid input, invalid lineage, invalid test row,
    # a missing required slice, or a failed slice floor. It does NOT reflect the
    # aggregate-accuracy gate or the byte-limit gate.
    critical_pass = rows_valid  # False immediately if any row was invalid

    test_metric = None
    if rows_valid and len(rows) > 0:
        correct = sum(1 for r in rows if r["label"] == r["prediction"])
        agg = round(correct / len(rows), 12)
        test_metric = agg
        if agg < metric_floor:
            codes.append("AGGREGATE_FLOOR")

        for slice_name, floor in required_slices.items():
            slice_rows = [r for r in rows if r["slice"] == slice_name]
            if not slice_rows:
                codes.append(f"MISSING_SLICE:{slice_name}")
                critical_pass = False
                continue
            s_correct = sum(1 for r in slice_rows if r["label"] == r["prediction"])
            s_acc = round(s_correct / len(slice_rows), 12)
            if s_acc < floor:
                codes.append(f"SLICE_FLOOR:{slice_name}")
                critical_pass = False

    codes = sort_dedupe_codes(codes)
    decision = "admit" if len(codes) == 0 else "reject"

    return 200, {
        "runId": run_id, "selectedTrialId": selected_trial_id, "datasetDigest": dataset_digest,
        "testMetric": test_metric, "criticalSlicePass": critical_pass,
        "decision": decision, "bytesProcessed": bytes_processed, "reasonCodes": codes,
    }


def bqml(body):
    if not isinstance(body, dict):
        return 400, {"error": "INVALID_INPUT"}
    phase = body.get("phase")
    if phase == "select":
        return _select(body)
    elif phase == "evaluate":
        return _evaluate(body)
    return 400, {"error": "INVALID_INPUT"}
