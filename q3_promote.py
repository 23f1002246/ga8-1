import re
from mlops_utils import parse_instant, is_finite_number, is_safe_nonneg_int, sort_dedupe_codes, utf8_sort_key

# Global alias state (simple single-pointer persistence across calls).
_ALIAS_STATE = {"champion": None}

CANON_VERSION_RE = re.compile(r"^[1-9]\d*$")


def _version_eligible(v_entry, policy, as_of_dt):
    codes = []
    ev = v_entry.get("evaluation")
    if not isinstance(ev, dict):
        codes.append("MISSING_EVALUATION")
        return codes

    created_dt = parse_instant(ev.get("createdAt"))
    if created_dt is None:
        codes.append("INVALID_TIMESTAMP")
    else:
        if created_dt > as_of_dt:
            codes.append("FUTURE_EVALUATION")
        else:
            max_age = policy.get("maxAgeSeconds")
            if is_finite_number(max_age) and (as_of_dt - created_dt).total_seconds() > max_age:
                codes.append("STALE_EVALUATION")

    for f in ("accuracy", "latencyMs", "sizeBytes"):
        val = ev.get(f)
        if not is_finite_number(val):
            codes.append("NON_FINITE")
            break
    acc = ev.get("accuracy")
    if is_finite_number(acc) and not (0 <= acc <= 1):
        codes.append("METRIC_RANGE")

    if ev.get("artifactDigest") != v_entry.get("artifactDigest"):
        codes.append("ARTIFACT_MISMATCH")
    if ev.get("datasetDigest") != policy.get("datasetDigest"):
        codes.append("DATASET_MISMATCH")
    if ev.get("schemaDigest") != policy.get("schemaDigest"):
        codes.append("SCHEMA_MISMATCH")

    if is_finite_number(acc) and 0 <= acc <= 1 and acc < policy.get("accuracyFloor", 0):
        codes.append("ACCURACY_FLOOR")
    lat = ev.get("latencyMs")
    if is_finite_number(lat) and lat > policy.get("maxLatencyMs", float("inf")):
        codes.append("LATENCY_LIMIT")
    size = ev.get("sizeBytes")
    if is_finite_number(size) and size > policy.get("maxSizeBytes", float("inf")):
        codes.append("SIZE_LIMIT")

    slices = ev.get("slices", {})
    if not isinstance(slices, dict):
        slices = {}
    for name, floor in (policy.get("requiredSlices") or {}).items():
        if name not in slices:
            codes.append(f"MISSING_SLICE:{name}")
        else:
            v = slices[name]
            if not is_finite_number(v):
                codes.append(f"SLICE_RANGE:{name}")
            elif not (0 <= v <= 1):
                codes.append(f"SLICE_RANGE:{name}")
            elif v < floor:
                codes.append(f"SLICE_FLOOR:{name}")

    return codes


def promote(body):
    if not isinstance(body, dict) or not isinstance(body.get("policy"), dict) \
            or not isinstance(body.get("versions"), list) or not isinstance(body.get("championVersion"), str):
        return 400, {"error": "INVALID_INPUT"}

    policy = body["policy"]
    versions = body["versions"]
    champion_version = body["championVersion"]
    as_of_dt = parse_instant(body.get("asOf"))

    policy_valid = (
        isinstance(policy.get("datasetDigest"), str) and policy["datasetDigest"] != ""
        and isinstance(policy.get("schemaDigest"), str) and policy["schemaDigest"] != ""
        and is_finite_number(policy.get("accuracyFloor")) and 0 <= policy["accuracyFloor"] <= 1
        and is_finite_number(policy.get("minImprovement"))
        and as_of_dt is not None
    )

    failed_gates = {}
    seen_versions = {}
    duplicates = set()
    for v in versions:
        vid = v.get("version") if isinstance(v, dict) else None
        if not (isinstance(vid, str) and CANON_VERSION_RE.match(vid)):
            failed_gates.setdefault(vid if vid is not None else "?", []).append("INVALID_VERSION")
            continue
        if vid in seen_versions:
            duplicates.add(vid)
        seen_versions.setdefault(vid, []).append(v)

    lookup = {}
    for vid, entries in seen_versions.items():
        if vid in duplicates:
            failed_gates.setdefault(vid, []).append("DUPLICATE_VERSION")
            continue
        lookup[vid] = entries[0]

    if not policy_valid:
        for vid in lookup:
            failed_gates.setdefault(vid, []).append("INVALID_POLICY")

    eligible = []
    for vid, v_entry in lookup.items():
        codes = _version_eligible(v_entry, policy, as_of_dt) if policy_valid and as_of_dt else ["INVALID_POLICY"]
        if codes:
            failed_gates.setdefault(vid, []).extend(codes)
        else:
            eligible.append(v_entry)

    eligible.sort(key=lambda v: (
        -v["evaluation"]["accuracy"], v["evaluation"]["latencyMs"], v["evaluation"]["sizeBytes"], int(v["version"])
    ))

    champion_entry = lookup.get(champion_version)
    champion_eligible = champion_entry is not None and champion_version not in failed_gates

    for vid in failed_gates:
        failed_gates[vid] = sort_dedupe_codes(failed_gates[vid])

    if not champion_eligible:
        return 200, {
            "action": "block", "championVersion": champion_version, "selectedVersion": None,
            "eligibleVersions": sorted([v["version"] for v in eligible], key=utf8_sort_key),
            "failedGates": failed_gates, "aliasMutation": None, "evidence": None,
        }

    champion_acc = champion_entry["evaluation"]["accuracy"]
    best_challenger = None
    for v in eligible:
        if v["version"] == champion_version:
            continue
        if best_challenger is None:
            best_challenger = v
            break  # eligible already sorted best-first

    action = "retain"
    selected = champion_entry
    if best_challenger is not None:
        diff = round(best_challenger["evaluation"]["accuracy"] - champion_acc, 12)
        if diff >= policy.get("minImprovement", 0):
            action = "promote"
            selected = best_challenger

    alias_mutation = None
    if action == "promote":
        alias_mutation = {"alias": "champion", "version": selected["version"]}
        _ALIAS_STATE["champion"] = selected["version"]

    return 200, {
        "action": action,
        "championVersion": champion_version,
        "selectedVersion": selected["version"],
        "eligibleVersions": sorted([v["version"] for v in eligible], key=utf8_sort_key),
        "failedGates": failed_gates,
        "aliasMutation": alias_mutation,
        "evidence": selected["evaluation"],
    }
