# tools/spell_format.py
# Conservative formatter/spell-fixer used by extract_box_texts.py
# - symspellpy is OPTIONAL. If unavailable, this becomes a no-op spell fixer.
# - domain_terms.txt (optional) lets you whitelist terms you don't want "fixed".

from __future__ import annotations
import os
import re
from typing import Optional

try:
    # Do NOT import 'utils' (not exported in newer versions)
    from symspellpy import SymSpell, Verbosity  # type: ignore
except Exception:
    SymSpell = None
    Verbosity = None


def _read_lines(path: str) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    except Exception:
        return []


class SpellFormatter:
    """
    clean(text) does:
      1) light punctuation/spacing tidy
      2) OPTIONAL word-level spell correction via SymSpell (if installed)
         * respects domain_terms.txt (whitelist)
    """
    def __init__(
        self,
        domain_terms_path: Optional[str] = None,
        max_edit_distance: int = 2,
        prefix_length: int = 7,
    ):
        self.domain_terms = set()
        if domain_terms_path and os.path.exists(domain_terms_path):
            self.domain_terms.update(_read_lines(domain_terms_path))

        self.sym: Optional[SymSpell] = None
        if SymSpell is not None:
            try:
                self.sym = SymSpell(max_dictionary_edit_distance=max_edit_distance,
                                    prefix_length=prefix_length)

                # Try to load a bundled frequency dictionary if present.
                # If you have a file like "frequency_dictionary_en_82_765.txt" put it in resources/
                # Format: term <tab> count
                here = os.path.dirname(os.path.dirname(__file__))  # project root
                freq_path = os.path.join(here, "resources", "frequency_dictionary_en_82_765.txt")
                if os.path.exists(freq_path):
                    # term_index=0, count_index=1 is the typical format
                    self.sym.load_dictionary(freq_path, term_index=0, count_index=1)

                # Inject domain terms with high counts so they won't be "corrected"
                for term in self.domain_terms:
                    try:
                        self.sym.create_dictionary_entry(term, 1_000_000)
                    except Exception:
                        pass
            except Exception:
                # If anything goes wrong, degrade gracefully
                self.sym = None

    # ---------------- tiny helpers ----------------
    _token_re = re.compile(r"\S+")

    @staticmethod
    def _tidy_punctuation(s: str) -> str:
        # remove space before punctuation
        s = re.sub(r"\s+([,.;:?!])", r"\1", s)
        # collapse 3+ spaces
        s = re.sub(r"[ \t]{3,}", "  ", s)
        # trim spaces inside brackets/quotes
        s = re.sub(r"([\(\[\{])\s+", r"\1", s)
        s = re.sub(r"\s+([\)\]\}])", r"\1", s)
        # normalize ellipses and double periods
        s = re.sub(r"\.{3,}", "...", s)
        s = re.sub(r"\.(\s*\.)+", ".", s)
        return s

    def _should_skip_spell(self, word: str) -> bool:
        if len(word) < 4:
            return True
        if any(ch.isdigit() for ch in word):
            return True
        lw = word.lower()
        if lw in self.domain_terms:
            return True
        # skip obvious abbreviations/acronyms/hyphenated terms
        if "-" in word or word.isupper():
            return True
        return False

    def _correct_word(self, word: str) -> str:
        if not self.sym or self._should_skip_spell(word):
            return word

        # Preserve punctuation wrappers like quotes/brackets
        m = re.match(r"^([\"'(\[]?)([A-Za-z\-]+)([\"')\],.:;!?]*)$", word)
        if not m:
            return word
        pre, core, post = m.groups()
        orig_core = core

        # Query SymSpell in lowercase
        lw = core.lower()
        try:
            from symspellpy import Verbosity as _Verbosity  # lazy local import for safety
            cand = self.sym.lookup(lw, _Verbosity.CLOSEST, max_edit_distance=2)
        except Exception:
            cand = []

        if not cand:
            return word

        fixed = cand[0].term
        # restore case
        if orig_core.istitle():
            fixed = fixed.capitalize()
        elif orig_core.isupper():
            fixed = fixed.upper()

        return f"{pre}{fixed}{post}"

    # ---------------- public API ----------------
    def clean(self, text: str) -> str:
        if not text:
            return text
        s = text

        # punctuation/spacing tidy first
        s = self._tidy_punctuation(s)

        # word-level correction pass (only if symspell available)
        if self.sym:
            s = self._token_re.sub(lambda m: self._correct_word(m.group(0)), s)

        return s
