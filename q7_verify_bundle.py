import re
import json
import hashlib
from mlops_utils import sha256_hex, compact_json, is_positive_safe_int, is_finite_number, sort_dedupe_codes, utf8_sort_key

REQUIRED_FILES = ["README.md", "training_manifest.json", "evaluation.json", "inventory.json",
                   "adapter_model.safetensors", "adapter_config.json"]
UNSAFE_EXTS = (".bin", ".pt", ".pth", ".pkl", ".pickle")
MANIFEST_REQUIRED_FIELDS = ["task", "datasetDigest", "codeDigest", "trainingConfigDigest",
                            "modelArtifactDigest", "evaluationArtifactDigest"]
CARD_MARKER_PREFIX = "<!-- tds-model-card "
CARD_MARKER_SUFFIX = "-->"


def verify_bundle(body):
    if not isinstance(body, dict) or not isinstance(body.get("files"), dict):
        return 400, {"error": "INVALID_INPUT"}
    policy = body.get("policy")
    if not isinstance(policy, dict):
        return 400, {"error": "INVALID_INPUT"}

    required_slices = policy.get("requiredSlices")
    license_ = policy.get("license")
    intended_use = policy.get("intendedUse")
    limitations = policy.get("limitations")
    policy_ok = (
        isinstance(required_slices, list) and len(required_slices) > 0
        and all(isinstance(s, str) and s != "" for s in required_slices)
        and len(set(required_slices)) == len(required_slices)
        and isinstance(license_, str) and license_ != ""
        and isinstance(intended_use, str) and intended_use != ""
        and isinstance(limitations, str) and limitations != ""
    )

    files = body["files"]
    violations = []

    if not policy_ok:
        violations.append("INVALID_POLICY")

    # Any file whose value is not a string is INVALID_FILE:<name>. Track which
    # files are usable (string-valued) so later steps never touch a non-string.
    string_files = {}
    for fname, fval in files.items():
        if isinstance(fval, str):
            string_files[fname] = fval
        else:
            violations.append(f"INVALID_FILE:{fname}")

    for f in REQUIRED_FILES:
        if f not in files:
            violations.append(f"MISSING_FILE:{f}")

    # unsafe weight files + untracked files (only considering files that exist)
    for fname in files:
        if fname not in REQUIRED_FILES and any(fname.endswith(ext) for ext in UNSAFE_EXTS):
            violations.append("UNSAFE_WEIGHTS")
        if fname not in REQUIRED_FILES:
            violations.append("UNTRACKED_FILE")

    def file_bytes(name):
        return string_files[name].encode("utf-8")

    # "have all required" means present AND string-valued
    have_all_required = all(f in string_files for f in REQUIRED_FILES)

    inventory_digest = None
    recomputed_inventory = []
    if have_all_required:
        other_files = [f for f in string_files if f != "inventory.json"]
        for f in sorted(other_files, key=utf8_sort_key):
            b = file_bytes(f)
            recomputed_inventory.append({"name": f, "bytes": len(b), "sha256": sha256_hex(b)})
        inv_json_str = compact_json(recomputed_inventory)
        inventory_digest = sha256_hex(inv_json_str.encode("utf-8"))

        try:
            submitted_inventory = json.loads(string_files["inventory.json"])
        except Exception:
            submitted_inventory = None
            violations.append(f"INVALID_JSON:inventory.json")

        if submitted_inventory is not None:
            if submitted_inventory != recomputed_inventory:
                violations.append("INVENTORY_MISMATCH")

    # adapter_config.json
    adapter_cfg = None
    if "adapter_config.json" in string_files:
        try:
            adapter_cfg = json.loads(string_files["adapter_config.json"])
        except Exception:
            violations.append("INVALID_JSON:adapter_config.json")
            adapter_cfg = None
        if adapter_cfg is not None:
            r_ok = is_positive_safe_int(adapter_cfg.get("r")) if isinstance(adapter_cfg, dict) else False
            tm = adapter_cfg.get("target_modules") if isinstance(adapter_cfg, dict) else None
            tm_ok = (isinstance(tm, list) and len(tm) > 0
                     and all(isinstance(x, str) and x != "" for x in tm)
                     and len(set(tm)) == len(tm))
            if not (isinstance(adapter_cfg, dict) and r_ok and tm_ok):
                violations.append("INVALID_ADAPTER_CONFIG")

    # training_manifest.json
    manifest = None
    base_revision = None
    if "training_manifest.json" in string_files:
        try:
            manifest = json.loads(string_files["training_manifest.json"])
        except Exception:
            violations.append("INVALID_JSON:training_manifest.json")
            manifest = None
        if manifest is not None:
            if not isinstance(manifest, dict):
                violations.append("INVALID_TRAINING_MANIFEST")
                manifest = None
            else:
                base_revision = manifest.get("baseRevision")
                if not (isinstance(base_revision, str) and re.fullmatch(r"[0-9a-f]{40}", base_revision)):
                    violations.append("MUTABLE_BASE_REVISION")
                for field in MANIFEST_REQUIRED_FIELDS:
                    v = manifest.get(field)
                    if not (isinstance(v, str) and v != ""):
                        violations.append(f"MISSING_MANIFEST_FIELD:{field}")

    # recompute model artifact digest + evaluation digest
    model_digest_actual = None
    eval_digest_actual = None
    if "adapter_model.safetensors" in string_files:
        model_digest_actual = sha256_hex(file_bytes("adapter_model.safetensors"))
    if "evaluation.json" in string_files:
        eval_digest_actual = sha256_hex(file_bytes("evaluation.json"))

    if manifest is not None and isinstance(manifest, dict):
        expected_model_digest = manifest.get("modelArtifactDigest")
        expected_eval_digest = manifest.get("evaluationArtifactDigest")
        if model_digest_actual is not None and expected_model_digest != model_digest_actual:
            violations.append("MODEL_ARTIFACT_MISMATCH")
        if eval_digest_actual is not None and expected_eval_digest != eval_digest_actual:
            violations.append("EVALUATION_ARTIFACT_MISMATCH")

    # evaluation.json content
    evaluation = None
    if "evaluation.json" in string_files:
        try:
            evaluation = json.loads(string_files["evaluation.json"])
        except Exception:
            violations.append("INVALID_JSON:evaluation.json")
            evaluation = None
        if evaluation is not None:
            if not isinstance(evaluation, dict):
                violations.append("INVALID_EVALUATION")
                evaluation = None
            else:
                eval_model_digest = evaluation.get("modelArtifactDigest") or evaluation.get("artifactDigest")
                if model_digest_actual is not None and eval_model_digest != model_digest_actual:
                    violations.append("EVALUATION_DIGEST_MISMATCH")
                agg = evaluation.get("aggregate")
                if not (is_finite_number(agg) and 0 <= agg <= 1):
                    violations.append("INVALID_AGGREGATE")
                if policy_ok:
                    slices = evaluation.get("slices", {})
                    if not isinstance(slices, dict):
                        slices = {}
                    for sl in required_slices:
                        if sl not in slices:
                            violations.append(f"MISSING_SLICE:{sl}")
                        else:
                            v = slices[sl]
                            if not (is_finite_number(v) and 0 <= v <= 1):
                                violations.append(f"SLICE_RANGE:{sl}")

    # Model card marker in README.md
    if "README.md" in string_files:
        readme = string_files["README.md"]
        markers = []
        search_from = 0
        while True:
            idx = readme.find(CARD_MARKER_PREFIX, search_from)
            if idx == -1:
                break
            end = readme.find(CARD_MARKER_SUFFIX, idx + len(CARD_MARKER_PREFIX))
            if end == -1:
                break
            markers.append(readme[idx + len(CARD_MARKER_PREFIX):end])
            search_from = end + len(CARD_MARKER_SUFFIX)

        if len(markers) == 0:
            violations.append("MODEL_CARD_COUNT")
            violations.append("MISSING_MODEL_CARD")
        elif len(markers) > 1:
            violations.append("MODEL_CARD_COUNT")
        else:
            raw = markers[0].strip()
            try:
                card = json.loads(raw)
            except Exception:
                card = None
            if not isinstance(card, dict):
                violations.append("INVALID_MODEL_CARD")
            else:
                expected = {
                    "task": manifest.get("task") if manifest else None,
                    "baseRevision": base_revision,
                    "datasetDigest": manifest.get("datasetDigest") if manifest else None,
                    "modelArtifactDigest": model_digest_actual,
                    "license": license_ if policy_ok else None,
                    "intendedUse": intended_use if policy_ok else None,
                    "limitations": limitations if policy_ok else None,
                }
                mismatch = False
                for k, v in expected.items():
                    if card.get(k) != v:
                        mismatch = True
                if mismatch:
                    violations.append("MODEL_CARD_MISMATCH")

    final_codes = sort_dedupe_codes(violations)
    decision = "admit" if len(final_codes) == 0 else "reject"

    return 200, {
        "decision": decision,
        "violations": final_codes,
        "inventoryDigest": inventory_digest,
    }
