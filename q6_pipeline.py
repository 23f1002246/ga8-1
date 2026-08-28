from mlops_utils import sha256_hex, compact_json, is_positive_safe_int

_SESSIONS = {}  # session -> state dict

DAG = ["verify_data", "prepare", "train", "evaluate", "register", "publish"]
PARENT = {"verify_data": None, "prepare": "verify_data", "train": "prepare",
          "evaluate": "train", "register": "evaluate", "publish": "register"}

REQUIRED_INPUTS = ["generation", "checksum", "canonicalData", "prepareCode", "prepareConfig",
                   "trainCode", "trainConfig", "runtime", "evaluateCode", "evaluateConfig",
                   "schemaDigest", "publishConfig"]


def _new_state():
    return {
        "revision": None,
        "inputs": None,
        "event_ids_seen": set(),
        "event_id_bodies": {},  # eventId -> canonical json of accepted event (for replay/conflict check)
        # trig_events holds ONLY the event(s) that define the node's CURRENT state:
        #   succeeded -> [first successful event id] (immutable, bound forever)
        #   started -> [the start event id for the current attempt]
        #   terminal_failed -> [the terminal event id]
        #   none / retryable_failed -> []
        "nodes": {n: {"state": "none", "artifact": None, "attempt": 0, "trig_events": [], "first_event_id": None}
                  for n in DAG},
    }


def _compute_keys(state):
    inputs = state["inputs"]
    keys = {}
    keys["verify_data"] = sha256_hex(compact_json([inputs["generation"], inputs["checksum"]]).encode())

    def reusable(node):
        return state["nodes"][node]["state"] == "succeeded"

    keys["prepare"] = sha256_hex(compact_json(
        [inputs["canonicalData"], inputs["prepareCode"], inputs["prepareConfig"]]).encode()) \
        if reusable("verify_data") else None

    keys["train"] = None
    if reusable("prepare") and keys["prepare"] is not None:
        prep_artifact = state["nodes"]["prepare"]["artifact"]
        keys["train"] = sha256_hex(compact_json(
            [prep_artifact, inputs["trainCode"], inputs["trainConfig"], inputs["runtime"]]).encode())

    keys["evaluate"] = None
    if reusable("train") and keys["train"] is not None:
        train_artifact = state["nodes"]["train"]["artifact"]
        keys["evaluate"] = sha256_hex(compact_json(
            [train_artifact, inputs["canonicalData"], inputs["evaluateCode"], inputs["evaluateConfig"]]).encode())

    keys["register"] = None
    if reusable("evaluate") and keys["evaluate"] is not None:
        eval_artifact = state["nodes"]["evaluate"]["artifact"]
        keys["register"] = sha256_hex(compact_json([eval_artifact, inputs["schemaDigest"]]).encode())

    keys["publish"] = None
    if reusable("register") and keys["register"] is not None:
        reg_artifact = state["nodes"]["register"]["artifact"]
        keys["publish"] = sha256_hex(compact_json([reg_artifact, inputs["publishConfig"]]).encode())

    return keys


def _process_event(state, ev, keys):
    node = ev.get("node")
    if node not in DAG:
        return False  # ignore
    if ev.get("revision") != state["revision"]:
        return False
    key = ev.get("key")
    if key != keys.get(node):
        return False  # unavailable parent / wrong key

    status = ev.get("status")
    if status not in ("started", "succeeded", "retryable_failed", "terminal_failed"):
        return False
    attempt = ev.get("attempt")
    if not is_positive_safe_int(attempt):
        return False
    artifact = ev.get("artifactDigest")
    receipt = ev.get("receiptId")

    if status == "succeeded":
        if not (isinstance(artifact, str) and artifact != ""):
            return False
    else:
        if artifact is not None:
            return False

    needs_receipt = status == "succeeded" and node in ("register", "publish")
    if needs_receipt:
        expected_receipt = f"receipt:{node}:{key}"
        if receipt != expected_receipt:
            return False
    else:
        if receipt is not None:
            return False

    ns = state["nodes"][node]
    cur = ns["state"]
    cur_attempt = ns["attempt"]

    def accept():
        return True

    if cur == "none":
        if status == "started" and attempt == 1:
            ns["state"] = "started"
            ns["attempt"] = 1
            ns["trig_events"] = [ev["eventId"]]
            return accept()
        return False  # ignore (completion or attempt>1 with no prior start)

    if cur == "started":
        if status in ("succeeded", "retryable_failed", "terminal_failed") and attempt == cur_attempt:
            if status == "succeeded":
                ns["state"] = "succeeded"
                ns["artifact"] = artifact
                ns["first_event_id"] = ev["eventId"]
                ns["trig_events"] = [ev["eventId"]]
            elif status == "retryable_failed":
                ns["state"] = "retryable_failed"
                ns["trig_events"] = []
            else:
                ns["state"] = "terminal_failed"
                ns["trig_events"] = [ev["eventId"]]
            return accept()
        if attempt < cur_attempt:
            return False  # ignore lower attempt
        raise _StatusConflict()

    if cur == "retryable_failed":
        if status == "started" and attempt == cur_attempt + 1:
            ns["state"] = "started"
            ns["attempt"] = attempt
            ns["trig_events"] = [ev["eventId"]]
            return accept()
        if attempt < cur_attempt:
            return False
        raise _StatusConflict()

    if cur == "succeeded":
        if status == "succeeded":
            if artifact != ns["artifact"]:
                raise _EvidenceConflict()
            return False  # exact replay of success info; ignored (already succeeded)
        raise _StatusConflict()

    if cur == "terminal_failed":
        raise _StatusConflict()

    return False


class _StatusConflict(Exception):
    pass


class _EvidenceConflict(Exception):
    pass


def pipeline(body):
    if not isinstance(body, dict):
        return 409, {"error": "INVALID_REQUEST"}
    session = body.get("session")
    revision = body.get("revision")
    inputs = body.get("inputs")
    events = body.get("events", [])

    if not (isinstance(session, str) and session != "") or not is_positive_safe_int(revision) \
            or not isinstance(inputs, dict) or not isinstance(events, list):
        return 409, {"error": "INVALID_REQUEST"}

    for f in REQUIRED_INPUTS:
        if not (isinstance(inputs.get(f), str) and inputs[f] != ""):
            return 409, {"error": "INVALID_REQUEST"}

    if session not in _SESSIONS:
        _SESSIONS[session] = _new_state()
    state = _SESSIONS[session]

    if state["revision"] is not None and state["revision"] == revision:
        if state["inputs"] != inputs:
            return 409, {"error": "REVISION_CONFLICT"}
    elif state["revision"] is not None and revision < state["revision"]:
        pass  # ignore stale revision events per spec; but a whole new lower revision request is unusual
    else:
        # new (higher) revision: replace inputs, clear attempt/terminal state, keep successful cache
        for node in DAG:
            ns = state["nodes"][node]
            if ns["state"] != "succeeded":
                ns["state"] = "none"
                ns["attempt"] = 0
                ns["trig_events"] = []
        state["revision"] = revision
        state["inputs"] = inputs

    accepted_ids = []
    ignored_ids = []

    # Validate + process events as one atomic batch
    import copy
    snapshot = copy.deepcopy(state)
    try:
        for ev in events:
            # Structural validity: must be an object with exactly the 8 named fields
            # and a non-empty string eventId. A structurally malformed event is an
            # INVALID_EVENT 409 (rolls back the whole batch).
            if not isinstance(ev, dict) or set(ev.keys()) != {
                "eventId", "revision", "node", "attempt", "status", "key", "artifactDigest", "receiptId"
            }:
                raise _ConflictErr("INVALID_EVENT")
            eid = ev.get("eventId")
            if not isinstance(eid, str) or eid == "":
                raise _ConflictErr("INVALID_EVENT")

            if ev.get("revision") != state["revision"]:
                ignored_ids.append(eid)
                continue

            if eid in state["event_id_bodies"]:
                if state["event_id_bodies"][eid] == compact_json(ev):
                    continue  # exact replay ignored, does not go in either list
                else:
                    raise _ConflictErr("EVENT_ID_CONFLICT")

            keys = _compute_keys(state)
            try:
                ok = _process_event(state, ev, keys)
            except _StatusConflict:
                raise _ConflictErr("STATUS_CONFLICT")
            except _EvidenceConflict:
                raise _ConflictErr("EVIDENCE_CONFLICT")

            if ok:
                accepted_ids.append(eid)
                state["event_id_bodies"][eid] = compact_json(ev)
            else:
                ignored_ids.append(eid)
    except _ConflictErr as ce:
        _SESSIONS[session] = snapshot
        return 409, {"error": ce.args[0]}

    keys = _compute_keys(state)
    nodes_out = []
    upstream_terminal = False
    upstream_pending = False
    for node in DAG:
        ns = state["nodes"][node]
        parent = PARENT[node]
        # dependencyDigests: the named inputs that feed this node's cache key,
        # plus the computed cacheKey itself. The named-input sets mirror the
        # SHA-256 array definitions in the spec.
        if node == "verify_data":
            dep_digests = {"generation": inputs["generation"], "checksum": inputs["checksum"],
                           "cacheKey": keys.get(node)}
        elif node == "prepare":
            dep_digests = {"canonicalData": inputs["canonicalData"], "prepareCode": inputs["prepareCode"],
                           "prepareConfig": inputs["prepareConfig"], "cacheKey": keys.get(node)}
        elif node == "train":
            dep_digests = {"prepareArtifact": state["nodes"]["prepare"]["artifact"],
                           "trainCode": inputs["trainCode"], "trainConfig": inputs["trainConfig"],
                           "runtime": inputs["runtime"], "cacheKey": keys.get(node)}
        elif node == "evaluate":
            dep_digests = {"trainArtifact": state["nodes"]["train"]["artifact"],
                           "canonicalData": inputs["canonicalData"], "evaluateCode": inputs["evaluateCode"],
                           "evaluateConfig": inputs["evaluateConfig"], "cacheKey": keys.get(node)}
        elif node == "register":
            dep_digests = {"evaluateArtifact": state["nodes"]["evaluate"]["artifact"],
                           "schemaDigest": inputs["schemaDigest"], "cacheKey": keys.get(node)}
        else:  # publish
            dep_digests = {"registerArtifact": state["nodes"]["register"]["artifact"],
                           "publishConfig": inputs["publishConfig"], "cacheKey": keys.get(node)}

        if upstream_terminal:
            action, reason = "block", "UPSTREAM_TERMINAL"
        elif ns["state"] == "succeeded":
            action, reason = "reuse", "CACHE_HIT"
        elif ns["state"] == "started":
            action, reason = "block", "RUNNING"
        elif ns["state"] == "terminal_failed":
            action, reason = "block", "TERMINAL_FAILURE"
            upstream_terminal = True
        elif keys.get(node) is None:
            action, reason = "block", "UPSTREAM_PENDING"
            upstream_pending = True
        elif ns["state"] == "retryable_failed":
            action, reason = "rerun", "RETRYABLE_FAILURE"
        else:
            action, reason = "rerun", "CACHE_MISS"

        nodes_out.append({
            "node": node, "action": action, "reasonCodes": [reason],
            "dependencyDigests": dep_digests,
            "triggeringEventIds": ns["trig_events"],
        })

    return 200, {
        "revision": state["revision"],
        "acceptedEventIds": accepted_ids,
        "ignoredEventIds": ignored_ids,
        "nodes": nodes_out,
    }


class _ConflictErr(Exception):
    pass
