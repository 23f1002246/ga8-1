import re
import json
import hashlib
from mlops_utils import (
    parse_instant, format_instant_utc, sha256_hex, compact_json,
    canonicalize_text, word_set, jaccard, is_decimal_string, is_hex,
    is_safe_nonneg_int, is_finite_number, sort_dedupe_codes, utf8_sort_key,
)
import crc32c as crc32c_lib

URI_RE = re.compile(r"^gs://([^/]+)/(.+)$")
ROW_KEYS = {"id", "entity", "eventTime", "revision", "text"}


def compute_crc32c_hex(content_bytes: bytes) -> str:
    val = crc32c_lib.crc32c(content_bytes)
    return format(val, "08x")


def validate_row_shape(row):
    if not isinstance(row, dict):
        return None
    if set(row.keys()) != ROW_KEYS:
        return None
    if not all(isinstance(row.get(k), str) for k in ("id", "entity", "eventTime", "text")):
        return None
    if not is_safe_nonneg_int(row.get("revision")):
        return None
    if parse_instant(row["eventTime"]) is None:
        return None
    return row


def process_object(obj, policy_valid, min_dt, max_dt, contamination_threshold):
    """Returns (accepted: bool, object_codes: list[str], rows: list[dict] or None)."""
    codes = []
    uri = obj.get("uri")
    uri_ok = isinstance(uri, str) and URI_RE.match(uri) is not None
    if not uri_ok:
        codes.append("URI_INVALID")

    gen = obj.get("generation")
    fgen = obj.get("fetchedGeneration")
    gen_ok = is_decimal_string(gen)
    fgen_ok = is_decimal_string(fgen)
    if not (gen_ok and fgen_ok):
        codes.append("GENERATION_INVALID")
    elif gen != fgen:
        codes.append("GENERATION_MISMATCH")

    crc = obj.get("crc32c")
    crc_ok = is_hex(crc, 8)
    if not crc_ok:
        codes.append("CRC32C_INVALID")

    content = obj.get("content")
    content_is_str = isinstance(content, str)
    schema_id = obj.get("schemaId")

    if crc_ok and content_is_str:
        content_bytes = content.encode("utf-8")
        actual_crc = compute_crc32c_hex(content_bytes)
        if actual_crc != crc:
            codes.append("CRC32C_MISMATCH")

    schema_invalid = False
    parsed_rows = []
    jsonl_invalid = False

    if not content_is_str:
        schema_invalid = True
    if schema_id != "training-v1":
        schema_invalid = True

    if content_is_str:
        lines = content.split("\n")
        nonblank = [ln for ln in lines if ln.strip() != ""]
        if len(nonblank) == 0:
            schema_invalid = True
        for ln in nonblank:
            try:
                row = json.loads(ln)
            except Exception:
                jsonl_invalid = True
                continue
            valid_row = validate_row_shape(row)
            if valid_row is None:
                schema_invalid = True
            else:
                parsed_rows.append(valid_row)

    if jsonl_invalid:
        codes.append("JSONL_INVALID")
    if schema_invalid:
        codes.append("SCHEMA_INVALID")

    accepted = len(codes) == 0
    return accepted, codes, (parsed_rows if accepted else None)


def build_corpus(body):
    if not isinstance(body, dict) or not isinstance(body.get("policy"), dict) \
            or not isinstance(body.get("objects"), list):
        return 400, {"error": "INVALID_INPUT"}

    policy = body["policy"]
    objects = body["objects"]

    min_dt = parse_instant(policy.get("minTime"))
    max_dt = parse_instant(policy.get("maxTime"))
    threshold = policy.get("contaminationThreshold")
    threshold_ok = is_finite_number(threshold) and 0 <= threshold <= 1
    policy_valid = (min_dt is not None) and (max_dt is not None) and threshold_ok

    rejected_objects = []
    lineage = []
    all_rows = []  # each: dict row (canonicalized) + source info

    for obj in objects:
        accepted, codes, rows = process_object(obj, policy_valid, min_dt, max_dt, threshold)
        uri_val = obj.get("uri") if isinstance(obj, dict) else None
        uri_out = uri_val if isinstance(uri_val, str) else None
        if not accepted:
            rejected_objects.append({"uri": uri_out, "reasonCodes": sort_dedupe_codes(codes)})
            continue
        lineage.append({
            "uri": obj["uri"], "generation": obj["generation"],
            "crc32c": obj["crc32c"], "schemaId": obj["schemaId"],
        })
        for row in rows:
            entity_c = canonicalize_text(row["entity"])
            text_c = canonicalize_text(row["text"])
            dt = parse_instant(row["eventTime"])
            event_utc = format_instant_utc(dt)
            all_rows.append({
                "id": row["id"], "entity": entity_c, "eventTime": event_utc,
                "revision": row["revision"], "text": text_c,
                "_dt": dt,
            })

    # Deduplicate by [entity, eventTime, text]; keep highest revision, then
    # UTF-8-byte-smallest id.
    groups = {}
    for r in all_rows:
        key = (r["entity"], r["eventTime"], r["text"])
        groups.setdefault(key, []).append(r)

    rejected_rows = []
    retained = []
    for key, group in groups.items():
        best = None
        for r in group:
            if best is None:
                best = r
            else:
                if r["revision"] > best["revision"]:
                    best = r
                elif r["revision"] == best["revision"] and utf8_sort_key(r["id"]) < utf8_sort_key(best["id"]):
                    best = r
        for r in group:
            if r is not best:
                rejected_rows.append({"id": r["id"], "reasonCodes": ["DUPLICATE"]})
        retained.append(best)

    # Policy validity gate
    if not policy_valid:
        for r in retained:
            rejected_rows.append({"id": r["id"], "reasonCodes": ["POLICY_INVALID"]})
        retained = []
    else:
        still = []
        for r in retained:
            if not (min_dt <= r["_dt"] <= max_dt):
                rejected_rows.append({"id": r["id"], "reasonCodes": ["OUT_OF_WINDOW"]})
            else:
                still.append(r)
        retained = still

    # Bucketing
    buckets = {"train": [], "validation": [], "test": []}
    for r in retained:
        first_byte = hashlib.sha256(r["entity"].encode("utf-8")).digest()[0]
        b = first_byte % 10
        if b <= 5:
            split = "train"
        elif b <= 7:
            split = "validation"
        else:
            split = "test"
        r["_split"] = split
        buckets[split].append(r)

    # Contamination check (validation/test rows vs train rows)
    train_word_sets = [word_set(r["text"]) for r in buckets["train"]]
    final_rows = []
    for split in ("validation", "test"):
        kept = []
        for r in buckets[split]:
            ws = word_set(r["text"])
            contaminated = any(jaccard(ws, tws) >= threshold for tws in train_word_sets)
            if contaminated:
                rejected_rows.append({"id": r["id"], "reasonCodes": ["TRAIN_CONTAMINATION"]})
            else:
                kept.append(r)
        buckets[split] = kept

    def clean_row(r):
        return {"id": r["id"], "entity": r["entity"], "eventTime": r["eventTime"],
                "revision": r["revision"], "text": r["text"]}

    splits_out = {}
    digests = {}
    for split in ("train", "validation", "test"):
        rows_sorted = sorted(buckets[split],
                              key=lambda r: (utf8_sort_key(r["id"]), compact_json(clean_row(r))))
        clean_rows = [clean_row(r) for r in rows_sorted]
        splits_out[split] = clean_rows
        lines = []
        for cr in clean_rows:
            ordered = {"id": cr["id"], "entity": cr["entity"], "eventTime": cr["eventTime"],
                       "revision": cr["revision"], "text": cr["text"]}
            lines.append(compact_json(ordered))
        blob = "".join(line + "\n" for line in lines).encode("utf-8")
        digests[split] = sha256_hex(blob)

    rejected_objects.sort(key=lambda o: (utf8_sort_key(o["uri"] or ""), compact_json(o)))
    rejected_rows.sort(key=lambda o: (utf8_sort_key(o["id"]), compact_json(o)))
    lineage.sort(key=lambda o: (utf8_sort_key(o["uri"]), compact_json(o)))

    response = {
        "splits": splits_out,
        "rejectedObjects": rejected_objects,
        "rejectedRows": rejected_rows,
        "digests": digests,
        "lineage": lineage,
    }
    return 200, response
