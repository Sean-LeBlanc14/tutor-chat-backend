# single_word_registry.py
import os, re, json
from collections import Counter, defaultdict
from typing import Dict, Tuple, List

try:
    from wordfreq import zipf_frequency
    HAS_WORDFREQ = True
except Exception:
    HAS_WORDFREQ = False

DEFAULT_MIN_ZIPF = 3.25  # "common enough" English threshold

def _safe_load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _safe_write_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _preserve_case(src: str, repl: str) -> str:
    if src.isupper():
        return repl.upper()
    if src.islower():
        return repl.lower()
    if src[:1].isupper() and src[1:].islower():
        return repl[:1].upper() + repl[1:].lower()
    return repl

class SingleWordRegistry:
    """
    Logs split single-word errors (e.g., 'di ff erent', 'Con tents', 'jour nal')
    to metadata/word_registry/pending.json with counts & samples.

    Applies only APPROVED corrections from metadata/word_registry/corrections.json
    across all text, preserving case per occurrence.
    """

    def __init__(self,
                 meta_dir: str,
                 min_zipf: float = DEFAULT_MIN_ZIPF,
                 context_radius: int = 25,
                 sample_limit: int = 30):
        self.meta_dir = meta_dir
        self.store_dir = os.path.join(meta_dir, "word_registry")
        self.corrections_path = os.path.join(self.store_dir, "corrections.json")
        self.pending_path = os.path.join(self.store_dir, "pending.json")
        self.min_zipf = min_zipf
        self.context_radius = context_radius
        self.sample_limit = sample_limit

        self.corrections = _safe_load_json(self.corrections_path, {"version": 1, "rules": {}})
        self.pending = _safe_load_json(self.pending_path, {"version": 1, "candidates": {}})

        # detection patterns
        self.split2 = re.compile(r"\b([A-Za-z]{2,})\s+([A-Za-z]{2,})\b")
        self.intra_ff = re.compile(r"(?<=\w)\s*f\s*f\s*(?=\w)", re.IGNORECASE)

    # ---------- public API ----------

    def scan(self, text: str, source_name: str) -> None:
        """Detect candidates, append to pending.json (no mutation)."""
        if not text:
            return
        cands: Counter = Counter()
        samples: dict = defaultdict(list)

        # generic splits (two words that might be one)
        for m in self.split2.finditer(text):
            a, b = m.group(1), m.group(2)
            if a.isupper() or b.isupper():
                continue
            if len(a) == 1 or len(b) == 1:
                continue
            merged = a + b
            if HAS_WORDFREQ and zipf_frequency(merged, "en") < self.min_zipf:
                continue
            key = f"{a} {b}"
            cands[key] += 1
            if len(samples[key]) < self.sample_limit:
                s, e = m.span()
                samples[key].append(self._context(text, s, e))

        # ff artifacts: record the whole token like "di ff erent"
        for m in self.intra_ff.finditer(text):
            s, e = m.span()
            key = self._token_within(text, s, e)
            if key:
                cands[key] += 1
                if len(samples[key]) < self.sample_limit:
                    samples[key].append(self._context(text, s, e))

        if not cands:
            return

        pend = self.pending["candidates"]
        for split, cnt in cands.items():
            entry = pend.get(split) or {
                "status": "pending",
                "replacement": None,
                "total_count": 0,
                "sources": {},
                "samples": []
            }
            entry["total_count"] += cnt
            entry["sources"][source_name] = entry["sources"].get(source_name, 0) + cnt

            sugg = split.replace(" ", "")
            if HAS_WORDFREQ and zipf_frequency(sugg, "en") < self.min_zipf:
                pass
            else:
                entry["replacement"] = entry["replacement"] or sugg

            for sm in samples[split]:
                if len(entry["samples"]) >= self.sample_limit:
                    break
                if sm not in entry["samples"]:
                    entry["samples"].append(sm)

            pend[split] = entry

        _safe_write_json(self.pending_path, self.pending)

    def apply_approved(self, text: str) -> Tuple[str, Dict]:
        """Apply only APPROVED corrections from corrections.json."""
        rules = self.corrections.get("rules", {})
        if not rules or not text:
            return text, {"applied": 0}

        items = sorted(rules.items(), key=lambda kv: len(kv[0]), reverse=True)
        applied = 0
        out = text

        for erroneous, spec in items:
            if not isinstance(spec, dict):
                continue
            if spec.get("status") != "approved":
                continue
            repl = spec.get("replacement")
            if not repl:
                continue

            pat = re.compile(rf"(?<!\w){re.escape(erroneous)}(?!\w)")
            def _repl(m):
                nonlocal applied
                applied += 1
                src = m.group(0)
                return _preserve_case(src, repl)
            out = pat.sub(_repl, out)

        return out, {"applied": applied}

    # ---------- utils ----------

    def _context(self, text: str, s: int, e: int) -> str:
        a = max(0, s - self.context_radius)
        b = min(len(text), e + self.context_radius)
        return text[a:b].replace("\n", "\\n")

    def _token_within(self, text: str, s: int, e: int) -> str:
        L = s
        while L > 0 and text[L-1].isalpha():
            L -= 1
        R = e
        while R < len(text) and text[R].isalpha():
            R += 1
        token = text[L:R]
        if re.search(r"[A-Za-z]\s+[A-Za-z]", token):
            return token
        return ""
