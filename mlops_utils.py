"""Shared helpers: timestamp validation/normalization, digests, safe-int checks,
canonicalization, compact JSON. Imported by main.py."""
import re
import json
import hashlib
import unicodedata
from datetime import datetime, timezone, timedelta

MAX_SAFE_INT = 2**53 - 1

TS_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(\.\d{1,3})?(Z|[+-]\d{2}:\d{2})$"
)

def is_safe_nonneg_int(v):
    return isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= MAX_SAFE_INT

def is_positive_safe_int(v):
    return isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= MAX_SAFE_INT

def is_finite_number(v):
    if isinstance(v, bool):
        return False
    if not isinstance(v, (int, float)):
        return False
    import math
    return math.isfinite(v)

def parse_instant(s):
    """Validate and parse a YYYY-MM-DDTHH:mm:ss[.sss](Z|±HH:mm) instant.
    Returns an aware datetime in UTC, or None if invalid."""
    if not isinstance(s, str):
        return None
    m = TS_RE.match(s)
    if not m:
        return None
    year, month, day, hh, mm, ss, frac, off = m.groups()
    year, month, day, hh, mm, ss = map(int, (year, month, day, hh, mm, ss))
    try:
        base = datetime(year, month, day, hh, mm, ss, tzinfo=timezone.utc)
    except ValueError:
        return None  # invalid calendar values (e.g. Feb 30)
    micros = 0
    if frac:
        digits = frac[1:]
        micros = int((digits + "000000")[:6])
    if off == "Z":
        offset = timedelta(0)
    else:
        sign = 1 if off[0] == "+" else -1
        oh, om = int(off[1:3]), int(off[4:6])
        if oh > 14 or om > 59:
            return None
        if oh == 14 and om != 0:
            return None
        offset = sign * timedelta(hours=oh, minutes=om)
    # base is constructed as if it were UTC clock time; the real instant is
    # base_clock_time - offset (since local = UTC + offset  =>  UTC = local - offset)
    dt_utc = base.replace(microsecond=micros) - offset
    return dt_utc

def format_instant_utc(dt):
    """Format an aware UTC datetime as YYYY-MM-DDTHH:mm:ss.sssZ (always 3 fraction digits)."""
    ms = dt.microsecond // 1000
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{ms:03d}Z"

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def compact_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, sort_keys=False)

def canonicalize_text(s: str) -> str:
    """NFKC, lowercase, trim, collapse Unicode whitespace runs to one ASCII space."""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s, flags=re.UNICODE)
    s = s.strip()
    return s

WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)  # runs of unicode letters/digits

def word_set(text: str):
    return set(m.group(0).lower() for m in WORD_RE.finditer(text))

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 1.0

def is_decimal_string(s):
    return isinstance(s, str) and re.fullmatch(r"\d+", s) is not None

def is_hex(s, length):
    return isinstance(s, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", s) is not None

def sort_dedupe_codes(codes):
    return sorted(set(codes), key=lambda c: c.encode("utf-8"))

def utf8_sort_key(s):
    return s.encode("utf-8") if isinstance(s, str) else b""
