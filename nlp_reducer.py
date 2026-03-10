"""
Riduzione dimensionale degli embedding → 2D / 3D.

Algoritmi disponibili:
  - PCA  : veloce, deterministico, preserva varianza globale
  - t-SNE: lento ma ottimo per cluster locali, non deterministico
"""

import numpy as np
from typing import Dict, List, Literal, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


ReductionMethod = Literal["PCA", "t-SNE"]


class DimensionReducer:
    """
    Prende un dizionario {token: vettore_nd} e lo proietta in 2D o 3D.
    Mantiene la stessa istanza del modello per coerenza tra aggiornamenti.
    """

    def __init__(self, method: ReductionMethod = "PCA", n_components: int = 2):
        self.method = method
        self.n_components = n_components
        self._scaler = StandardScaler()
        self._model = None

    # ── API pubblica ─────────────────────────────────────────────────────────

    def fit_transform(
        self, embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Adatta il modello agli embeddings e restituisce le coordinate ridotte.
        
        Returns:
            {token: np.ndarray di shape (n_components,)}
        """
        tokens, matrix = self._to_matrix(embeddings)
        if matrix.shape[0] < 2:
            # Caso degenere: un solo token
            return {tokens[0]: np.zeros(self.n_components)}

        matrix_scaled = self._scaler.fit_transform(matrix)
        reduced = self._run_reduction(matrix_scaled, fit=True)

        return {tok: reduced[i] for i, tok in enumerate(tokens)}

    def transform(
        self, embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Proietta nuovi embeddings usando il modello già addestrato.
        (Solo per PCA; t-SNE richiede sempre fit_transform)
        """
        if self.method == "t-SNE" or self._model is None:
            return self.fit_transform(embeddings)

        tokens, matrix = self._to_matrix(embeddings)
        matrix_scaled = self._scaler.transform(matrix)
        reduced = self._model.transform(matrix_scaled)
        return {tok: reduced[i] for i, tok in enumerate(tokens)}

    def reduce_snapshots(
        self,
        history: List[Dict],
        tokens_to_show: List[str],
    ) -> List[Dict]:
        """
        Riduce tutti gli snapshot dello storico in modo coerente.
        Usa PCA fittato sul primo snapshot per mantenere lo spazio stabile.

        Returns:
            Lista di {label, coords: {token: np.ndarray}}
        """
        if not history:
            return []

        # Fit sul primo snapshot (baseline)
        first_emb = {
            t: v for t, v in history[0]["embeddings"].items()
            if t in tokens_to_show
        }
        if len(first_emb) < 2:
            return []

        tokens_ref, matrix_ref = self._to_matrix(first_emb)
        matrix_scaled_ref = self._scaler.fit_transform(matrix_ref)

        # Usa PCA per snapshot (t-SNE non è stabile tra frame)
        pca = PCA(n_components=min(self.n_components, matrix_scaled_ref.shape[1],
                                   matrix_scaled_ref.shape[0]))
        pca.fit(matrix_scaled_ref)

        result = []
        for snap in history:
            emb = {t: v for t, v in snap["embeddings"].items() if t in tokens_to_show}
            if not emb:
                continue
            tokens_s, matrix_s = self._to_matrix(emb)
            # Pad/trim alla stessa dimensione del riferimento
            dim_ref = matrix_scaled_ref.shape[1]
            matrix_s = self._pad_or_trim(matrix_s, dim_ref)
            matrix_s_scaled = self._scaler.transform(matrix_s)
            try:
                reduced = pca.transform(matrix_s_scaled)
            except Exception:
                reduced = np.zeros((len(tokens_s), self.n_components))
            # Pad componenti se necessario
            if reduced.shape[1] < self.n_components:
                reduced = np.pad(reduced, ((0,0),(0, self.n_components - reduced.shape[1])))
            result.append({
                "label": snap["label"],
                "coords": {tok: reduced[i] for i, tok in enumerate(tokens_s)},
            })

        return result

    # ── Interno ──────────────────────────────────────────────────────────────

    def _to_matrix(
        self, embeddings: Dict[str, np.ndarray]
    ) -> Tuple[List[str], np.ndarray]:
        tokens = list(embeddings.keys())
        vecs = [embeddings[t] for t in tokens]
        # Uniforma le dimensioni
        max_dim = max(v.shape[0] for v in vecs)
        padded = [np.pad(v, (0, max_dim - v.shape[0])) for v in vecs]
        return tokens, np.array(padded, dtype=np.float32)

    def _pad_or_trim(self, matrix: np.ndarray, target_dim: int) -> np.ndarray:
        current = matrix.shape[1]
        if current < target_dim:
            return np.pad(matrix, ((0, 0), (0, target_dim - current)))
        return matrix[:, :target_dim]

    def _run_reduction(self, matrix: np.ndarray, fit: bool = True) -> np.ndarray:
        n_samples = matrix.shape[0]
        n_comp = min(self.n_components, matrix.shape[1], n_samples)

        if self.method == "PCA":
            if fit or self._model is None:
                self._model = PCA(n_components=n_comp)
                reduced = self._model.fit_transform(matrix)
            else:
                reduced = self._model.transform(matrix)

        else:  # t-SNE
            perplexity = min(30, max(2, n_samples - 1))
            self._model = TSNE(
                n_components=n_comp,
                perplexity=perplexity,
                random_state=42,
                max_iter=500,
            )
            reduced = self._model.fit_transform(matrix)

        # Pad se n_comp < n_components richiesto
        if reduced.shape[1] < self.n_components:
            reduced = np.pad(reduced, ((0, 0), (0, self.n_components - reduced.shape[1])))

        return reduced