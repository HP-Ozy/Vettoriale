"""
Modulo di embedding: mappa token → vettori ad alta dimensione
e aggiorna i vettori in base al contesto (finestra scorrevole).

Strategia:
  - Vettori iniziali: TF-IDF sul corpus accumulato
  - Aggiornamento contestuale: media pesata con i vettori
    dei token vicini (co-occurrence window)
  - Il peso del contesto è regolabile (alpha)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from nlp_tokenizer import Tokenizer


class TokenEmbedder:
    """
    Gestisce la mappatura token → vettore e il suo aggiornamento contestuale.

    Attributi pubblici:
        embeddings  : Dict[token, np.ndarray]  — vettori correnti
        history     : List[Dict]               — storico degli aggiornamenti
        vocabulary  : Dict[token, int]         — indice vocabolario
    """

    def __init__(self, context_window: int = 2, alpha: float = 0.3):
        """
        Args:
            context_window: quanti token a sinistra/destra formano il contesto
            alpha: peso dell'aggiornamento contestuale (0=nessun aggiornamento, 1=solo contesto)
        """
        self.context_window = context_window
        self.alpha = alpha
        self.tokenizer = Tokenizer()

        self.embeddings: Dict[str, np.ndarray] = {}
        self.vocabulary: Dict[str, int] = {}
        self.history: List[Dict] = []          # storico per animazione
        self._corpus: List[str] = []           # testi accumulati
        self._vectorizer: Optional[TfidfVectorizer] = None

    # ── API pubblica ─────────────────────────────────────────────────────────

    def fit(self, texts: List[str]) -> "TokenEmbedder":
        """
        Costruisce i vettori TF-IDF dal corpus.
        Ogni token ottiene un vettore = riga della matrice TF-IDF.
        """
        self._corpus = texts

        # TF-IDF: ogni documento è un token (per avere vettori per token)
        # Usiamo i testi puliti come documenti
        cleaned = [" ".join(self.tokenizer.tokenize(t)) for t in texts]
        cleaned = [c for c in cleaned if c.strip()]

        if not cleaned:
            return self

        self._vectorizer = TfidfVectorizer(
            max_features=500,
            sublinear_tf=True,
            analyzer="word",
            token_pattern=r"[a-zA-ZàèìòùáéíóúÀÈÌÒÙ]{2,}",
        )
        tfidf_matrix = self._vectorizer.fit_transform(cleaned)
        feature_names = self._vectorizer.get_feature_names_out()

        # Vettore per ogni token = media delle sue righe TF-IDF
        # (come se ogni frase in cui appare contribuisse al suo significato)
        all_tokens: set = set()
        for text in texts:
            all_tokens.update(self.tokenizer.tokenize(text))

        for i, token in enumerate(all_tokens):
            self.vocabulary[token] = i

        for token in all_tokens:
            if token in feature_names:
                token_idx = list(feature_names).index(token)
                # Vettore = colonna TF-IDF del token su tutti i documenti
                vec = np.array(tfidf_matrix[:, token_idx].todense()).flatten()
                # Pad/trim a lunghezza fissa (n_docs)
                self.embeddings[token] = vec.astype(np.float32)
            else:
                # Token non nel vocabolario TF-IDF → vettore casuale piccolo
                n = tfidf_matrix.shape[0]
                self.embeddings[token] = np.random.randn(n).astype(np.float32) * 0.01

        self._pad_embeddings()
        self._save_snapshot(label="iniziale (TF-IDF)")
        return self

    def update_context(self, text: str) -> Dict[str, np.ndarray]:
        """
        Aggiorna i vettori dei token nel testo in base al loro contesto locale.
        Usa una finestra scorrevole di ampiezza ±context_window.

        Returns:
            dizionario {token: vettore_aggiornato}
        """
        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            return {}

        # Aggiungi eventuali token nuovi
        for tok in tokens:
            if tok not in self.embeddings:
                dim = self._embedding_dim()
                self.embeddings[tok] = np.random.randn(dim).astype(np.float32) * 0.01
                self.vocabulary[tok] = len(self.vocabulary)

        updated: Dict[str, np.ndarray] = {}

        for i, token in enumerate(tokens):
            # Raccoglie token nel contesto
            left  = max(0, i - self.context_window)
            right = min(len(tokens), i + self.context_window + 1)
            context_tokens = tokens[left:i] + tokens[i+1:right]

            if not context_tokens:
                continue

            # Media dei vettori contestuali
            context_vecs = [self.embeddings[ct] for ct in context_tokens
                            if ct in self.embeddings]
            if not context_vecs:
                continue

            context_mean = np.mean(context_vecs, axis=0)

            # Aggiornamento: blend tra vettore corrente e contesto
            old_vec = self.embeddings[token].copy()
            new_vec = (1 - self.alpha) * old_vec + self.alpha * context_mean
            self.embeddings[token] = new_vec.astype(np.float32)
            updated[token] = new_vec

        if updated:
            self._save_snapshot(label=f"dopo: \"{text[:40]}\"")

        return updated

    def get_similar(self, token: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """Restituisce i top_n token più simili per coseno."""
        if token not in self.embeddings:
            return []

        query = self.embeddings[token]
        scores = []
        for tok, vec in self.embeddings.items():
            if tok == token:
                continue
            sim = self._cosine(query, vec)
            scores.append((tok, float(sim)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

    def snapshot_tokens(self) -> List[str]:
        """Token disponibili nello storico."""
        return list(self.embeddings.keys())

    # ── Interno ──────────────────────────────────────────────────────────────

    def _pad_embeddings(self):
        """Assicura che tutti i vettori abbiano la stessa dimensione."""
        if not self.embeddings:
            return
        max_dim = max(v.shape[0] for v in self.embeddings.values())
        for tok in self.embeddings:
            vec = self.embeddings[tok]
            if vec.shape[0] < max_dim:
                self.embeddings[tok] = np.pad(vec, (0, max_dim - vec.shape[0]))

    def _embedding_dim(self) -> int:
        if self.embeddings:
            return next(iter(self.embeddings.values())).shape[0]
        return 50

    def _save_snapshot(self, label: str):
        """Salva uno snapshot dei vettori correnti per l'animazione."""
        self.history.append({
            "label": label,
            "embeddings": {tok: vec.copy() for tok, vec in self.embeddings.items()},
        })

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))