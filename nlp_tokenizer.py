"""
Modulo di tokenizzazione del testo.
Supporta: pulizia, stop-word removal, stemming opzionale.
"""

import re
import string
from typing import List, Dict, Tuple


# Stop words italiane + inglesi (leggere, senza dipendenze esterne)
STOP_WORDS_IT = {
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "e", "ed",
    "o", "ma", "se", "che", "chi", "cui", "non", "più", "per", "su",
    "con", "tra", "fra", "di", "del", "dello", "della", "dei", "degli",
    "delle", "da", "dal", "dallo", "dalla", "dai", "dagli", "dalle",
    "in", "nel", "nello", "nella", "nei", "negli", "nelle", "a", "al",
    "allo", "alla", "ai", "agli", "alle", "è", "sono", "era", "erano",
    "ho", "ha", "hanno", "questo", "questa", "questi", "queste", "quello",
    "quella", "quelli", "quelle", "anche", "come", "quando", "dove",
    "perché", "poi", "così", "tutto", "tutti", "tutta", "tutte",
}

STOP_WORDS_EN = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "that", "this",
    "these", "those", "it", "its", "they", "them", "their", "we", "our",
    "i", "my", "you", "your", "he", "she", "his", "her",
}

ALL_STOP_WORDS = STOP_WORDS_IT | STOP_WORDS_EN


class Tokenizer:
    """
    Tokenizzatore testuale con pipeline configurabile:
    1. Normalizzazione (lowercase, punteggiatura)
    2. Splitting in token
    3. Filtraggio stop-word
    4. Filtraggio token troppo corti
    """

    def __init__(
        self,
        remove_stopwords: bool = True,
        min_length: int = 2,
        language: str = "auto",
    ):
        self.remove_stopwords = remove_stopwords
        self.min_length = min_length
        self.language = language

    # ── API pubblica ─────────────────────────────────────────────────────────

    def tokenize(self, text: str) -> List[str]:
        """Restituisce lista di token puliti."""
        text = self._normalize(text)
        tokens = self._split(text)
        tokens = self._filter(tokens)
        return tokens

    def tokenize_with_positions(self, text: str) -> List[Tuple[str, int]]:
        """Restituisce (token, posizione_nel_testo_originale)."""
        tokens_raw = self._split(self._normalize(text))
        result = []
        pos = 0
        for tok in tokens_raw:
            idx = text.lower().find(tok, pos)
            if idx != -1:
                pos = idx + len(tok)
            result.append((tok, idx))
        # Filtra stop-word mantenendo le posizioni
        if self.remove_stopwords:
            result = [(t, p) for t, p in result if t not in ALL_STOP_WORDS]
        result = [(t, p) for t, p in result if len(t) >= self.min_length]
        return result

    def build_vocabulary(self, texts: List[str]) -> Dict[str, int]:
        """Costruisce vocabolario {token: indice} su una lista di testi."""
        vocab: Dict[str, int] = {}
        idx = 0
        for text in texts:
            for token in self.tokenize(text):
                if token not in vocab:
                    vocab[token] = idx
                    idx += 1
        return vocab

    def get_ngrams(self, tokens: List[str], n: int = 2) -> List[Tuple[str, ...]]:
        """Produce n-grammi dalla lista di token."""
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    # ── Interno ──────────────────────────────────────────────────────────────

    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[''`]", "'", text)            # normalizza apostrofi
        text = re.sub(r"[^\w\s']", " ", text)         # rimuovi punteggiatura
        text = re.sub(r"\d+", "", text)               # rimuovi numeri
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _split(self, text: str) -> List[str]:
        return text.split()

    def _filter(self, tokens: List[str]) -> List[str]:
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in ALL_STOP_WORDS]
        tokens = [t for t in tokens if len(t) >= self.min_length]
        return tokens