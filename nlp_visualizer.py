"""
Modulo di visualizzazione degli embedding token.
Genera figure Plotly statiche e animate (frame per frame).
"""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional


# Palette colori per i token
PALETTE = [
    "#00d4ff", "#ff6b6b", "#ffd93d", "#6bcb77", "#c77dff",
    "#ff9f43", "#48dbfb", "#ff6b9d", "#a29bfe", "#55efc4",
    "#fd79a8", "#74b9ff", "#00cec9", "#e17055", "#fdcb6e",
]


class EmbeddingVisualizer:
    """Genera visualizzazioni Plotly per gli embedding token."""

    def __init__(self):
        self._color_map: Dict[str, str] = {}
        self._color_idx = 0

    # ── API pubblica ─────────────────────────────────────────────────────────

    def plot_static(
        self,
        coords: Dict[str, np.ndarray],
        mode: str = "2D",
        title: str = "Embedding Token",
        highlight: Optional[List[str]] = None,
        similar_to: Optional[str] = None,
        similarity_scores: Optional[Dict[str, float]] = None,
    ) -> go.Figure:
        """
        Scatter statico 2D o 3D dei token.
        
        Args:
            coords: {token: array [x,y] o [x,y,z]}
            highlight: token da evidenziare con bordo
            similar_to: token centrale per visualizzare similarità
        """
        fig = go.Figure()

        tokens = list(coords.keys())
        for tok in tokens:
            if tok not in self._color_map:
                self._color_map[tok] = PALETTE[self._color_idx % len(PALETTE)]
                self._color_idx += 1

        if mode == "2D":
            self._scatter_2d(fig, coords, tokens, highlight, similarity_scores)
        else:
            self._scatter_3d(fig, coords, tokens, highlight, similarity_scores)

        fig.update_layout(**self._base_layout(title, mode))
        return fig

    def plot_animated(
        self,
        snapshots: List[Dict],
        mode: str = "2D",
    ) -> go.Figure:
        """
        Animazione frame-per-frame dell'evoluzione degli embedding.

        Args:
            snapshots: lista di {label, coords: {token: array}}
        """
        if not snapshots:
            return go.Figure()

        # Prepara i colori
        all_tokens = set()
        for snap in snapshots:
            all_tokens.update(snap["coords"].keys())
        for tok in all_tokens:
            if tok not in self._color_map:
                self._color_map[tok] = PALETTE[self._color_idx % len(PALETTE)]
                self._color_idx += 1

        frames = []
        for snap in snapshots:
            frame_data = self._make_frame_data(snap["coords"], mode)
            frames.append(go.Frame(data=frame_data, name=snap["label"]))

        # Figura iniziale = primo frame
        fig = go.Figure(
            data=self._make_frame_data(snapshots[0]["coords"], mode),
            frames=frames,
        )

        # Slider + pulsanti play
        slider_steps = [
            dict(
                args=[[f.name], {"frame": {"duration": 800, "redraw": True},
                                  "mode": "immediate"}],
                label=snap["label"][:25],
                method="animate",
            )
            for f, snap in zip(frames, snapshots)
        ]

        fig.update_layout(
            **self._base_layout("Evoluzione Embedding nel Contesto", mode),
            updatemenus=[dict(
                type="buttons", showactive=False,
                y=1.08, x=0.5, xanchor="center",
                buttons=[
                    dict(label="▶ Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 900, "redraw": True},
                                      "fromcurrent": True, "transition": {"duration": 400}}]),
                    dict(label="⏸ Pausa",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}]),
                ],
            )],
            sliders=[dict(
                steps=slider_steps,
                active=0,
                y=0, x=0.05, len=0.9,
                currentvalue=dict(prefix="Fase: ", font=dict(color="#aaa", size=11)),
                pad=dict(t=40),
                font=dict(color="#aaa"),
                bgcolor="#1e2130",
                bordercolor="#333",
            )],
        )
        return fig

    # ── Scatter 2D ───────────────────────────────────────────────────────────

    def _scatter_2d(self, fig, coords, tokens, highlight, similarity_scores):
        # Linee di similarità (ragno)
        if highlight and similarity_scores:
            for tok, score in similarity_scores.items():
                if tok in coords and highlight[0] in coords:
                    cx, cy = coords[highlight[0]][:2]
                    tx, ty = coords[tok][:2]
                    alpha = int(40 + score * 160)
                    fig.add_trace(go.Scatter(
                        x=[cx, tx], y=[cy, ty],
                        mode="lines",
                        line=dict(color=f"rgba(255,255,100,{score:.2f})", width=1.5, dash="dot"),
                        showlegend=False, hoverinfo="skip",
                    ))

        for tok in tokens:
            pt = coords[tok]
            x, y = float(pt[0]), float(pt[1]) if len(pt) > 1 else 0.0
            color = self._color_map.get(tok, "#aaa")
            is_highlight = highlight and tok in highlight
            sim_score = similarity_scores.get(tok, None) if similarity_scores else None

            marker = dict(
                color=color,
                size=18 if is_highlight else 12,
                line=dict(color="white" if is_highlight else color, width=2 if is_highlight else 0),
                symbol="star" if is_highlight else "circle",
                opacity=1.0,
            )

            hover = f"<b>{tok}</b>"
            if sim_score is not None:
                hover += f"<br>cosine sim: {sim_score:.3f}"

            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                marker=marker,
                text=[f"<b>{tok}</b>"],
                textposition="top center",
                textfont=dict(color=color, size=11),
                name=tok,
                hovertemplate=hover + "<extra></extra>",
            ))

    # ── Scatter 3D ───────────────────────────────────────────────────────────

    def _scatter_3d(self, fig, coords, tokens, highlight, similarity_scores):
        for tok in tokens:
            pt = coords[tok]
            x = float(pt[0])
            y = float(pt[1]) if len(pt) > 1 else 0.0
            z = float(pt[2]) if len(pt) > 2 else 0.0
            color = self._color_map.get(tok, "#aaa")
            is_highlight = highlight and tok in highlight
            sim_score = similarity_scores.get(tok, None) if similarity_scores else None

            hover = f"<b>{tok}</b><br>({x:.3f}, {y:.3f}, {z:.3f})"
            if sim_score is not None:
                hover += f"<br>cosine sim: {sim_score:.3f}"

            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode="markers+text",
                marker=dict(
                    color=color,
                    size=10 if is_highlight else 6,
                    line=dict(color="white", width=1 if is_highlight else 0),
                    symbol="diamond" if is_highlight else "circle",
                ),
                text=[f"<b>{tok}</b>"],
                textfont=dict(color=color, size=10),
                name=tok,
                hovertemplate=hover + "<extra></extra>",
            ))

    # ── Frame animazione ─────────────────────────────────────────────────────

    def _make_frame_data(self, coords: Dict[str, np.ndarray], mode: str) -> list:
        data = []
        for tok, pt in coords.items():
            color = self._color_map.get(tok, "#aaa")
            if mode == "2D":
                x, y = float(pt[0]), float(pt[1]) if len(pt) > 1 else 0.0
                data.append(go.Scatter(
                    x=[x], y=[y],
                    mode="markers+text",
                    marker=dict(color=color, size=12),
                    text=[f"<b>{tok}</b>"],
                    textposition="top center",
                    textfont=dict(color=color, size=11),
                    name=tok,
                    hovertemplate=f"<b>{tok}</b><br>({x:.3f}, {y:.3f})<extra></extra>",
                ))
            else:
                x = float(pt[0])
                y = float(pt[1]) if len(pt) > 1 else 0.0
                z = float(pt[2]) if len(pt) > 2 else 0.0
                data.append(go.Scatter3d(
                    x=[x], y=[y], z=[z],
                    mode="markers+text",
                    marker=dict(color=color, size=6),
                    text=[f"<b>{tok}</b>"],
                    textfont=dict(color=color, size=10),
                    name=tok,
                ))
        return data

    # ── Layout ───────────────────────────────────────────────────────────────

    def _base_layout(self, title: str, mode: str) -> dict:
        base = dict(
            title=dict(text=title, font=dict(color="#e0e0e0", size=15)),
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            font=dict(color="#e0e0e0", family="monospace"),
            margin=dict(l=10, r=10, t=60, b=10),
            height=560,
            showlegend=True,
            legend=dict(
                bgcolor="#1e2130", bordercolor="#333", borderwidth=1,
                font=dict(color="#ccc", size=10),
            ),
            uirevision="stable",
        )
        if mode == "2D":
            base.update(dict(
                xaxis=dict(showgrid=True, gridcolor="#2a2f45", zeroline=True,
                           zerolinecolor="#555", tickfont=dict(color="#aaa")),
                yaxis=dict(showgrid=True, gridcolor="#2a2f45", zeroline=True,
                           zerolinecolor="#555", tickfont=dict(color="#aaa"),
                           scaleanchor="x", scaleratio=1),
            ))
        else:
            base["scene"] = dict(
                xaxis=dict(backgroundcolor="#0f1117", gridcolor="#2a2f45",
                           zerolinecolor="#555", tickfont=dict(color="#aaa")),
                yaxis=dict(backgroundcolor="#0f1117", gridcolor="#2a2f45",
                           zerolinecolor="#555", tickfont=dict(color="#aaa")),
                zaxis=dict(backgroundcolor="#0f1117", gridcolor="#2a2f45",
                           zerolinecolor="#555", tickfont=dict(color="#aaa")),
                bgcolor="#0f1117",
            )
        return base