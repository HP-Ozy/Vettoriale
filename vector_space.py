import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Any


class VectorSpace:
    """Gestisce lo spazio vettoriale e la generazione del grafico."""

    def __init__(self):
        self.vectors: List[Dict[str, Any]] = []

    # ── CRUD ────────────────────────────────────────────────────────────────

    def add_vector(self, name: str, coords: list, origin: list, color: str):
        self.vectors.append({
            "name": name,
            "coords": coords,
            "origin": origin,
            "color": color,
        })

    def remove_vector(self, index: int):
        if 0 <= index < len(self.vectors):
            self.vectors.pop(index)

    def clear(self):
        self.vectors = []

    def get_by_name(self, name: str) -> Dict[str, Any]:
        for v in self.vectors:
            if v["name"] == name:
                return v
        return self.vectors[0]

    # ── Figura Plotly ────────────────────────────────────────────────────────

    def build_figure(self, mode: str = "2D") -> go.Figure:
        if mode == "2D":
            return self._build_2d()
        else:
            return self._build_3d()

    # ── 2D ──────────────────────────────────────────────────────────────────

    def _build_2d(self) -> go.Figure:
        fig = go.Figure()

        # Calcola range dinamico
        all_x = [0.0]
        all_y = [0.0]
        for v in self.vectors:
            ox, oy = (v["origin"] + [0, 0])[:2]
            ex = ox + v["coords"][0]
            ey = oy + (v["coords"][1] if len(v["coords"]) > 1 else 0)
            all_x += [ox, ex]
            all_y += [oy, ey]

        margin = 1.5
        xr = max(abs(min(all_x)), abs(max(all_x))) + margin
        yr = max(abs(min(all_y)), abs(max(all_y))) + margin

        # Griglia
        self._add_grid_2d(fig, xr, yr)

        # Assi
        self._add_axes_2d(fig, xr, yr)

        # Vettori
        for v in self.vectors:
            ox, oy = (list(v["origin"]) + [0, 0])[:2]
            dx = v["coords"][0]
            dy = v["coords"][1] if len(v["coords"]) > 1 else 0
            self._add_arrow_2d(fig, ox, oy, dx, dy, v["color"], v["name"])

        fig.update_layout(
            **self._base_layout(),
            xaxis=dict(range=[-xr, xr], zeroline=False, showgrid=False,
                       tickfont=dict(color="#aaa"), color="#aaa"),
            yaxis=dict(range=[-yr, yr], zeroline=False, showgrid=False,
                       scaleanchor="x", scaleratio=1,
                       tickfont=dict(color="#aaa"), color="#aaa"),
        )
        return fig

    def _add_grid_2d(self, fig, xr, yr):
        step = 1
        for xi in np.arange(-xr, xr + step, step):
            fig.add_shape(type="line", x0=xi, y0=-yr, x1=xi, y1=yr,
                          line=dict(color="#2a2f45", width=1))
        for yi in np.arange(-yr, yr + step, step):
            fig.add_shape(type="line", x0=-xr, y0=yi, x1=xr, y1=yi,
                          line=dict(color="#2a2f45", width=1))

    def _add_axes_2d(self, fig, xr, yr):
        # Asse X
        fig.add_annotation(x=xr, y=0, ax=-xr, ay=0,
                            xref="x", yref="y", axref="x", ayref="y",
                            showarrow=True, arrowhead=3, arrowsize=1.5,
                            arrowwidth=2, arrowcolor="#555")
        fig.add_annotation(x=xr * 0.98, y=0.2, text="<b>x</b>",
                            showarrow=False, font=dict(color="#888", size=14))
        # Asse Y
        fig.add_annotation(x=0, y=yr, ax=0, ay=-yr,
                            xref="x", yref="y", axref="x", ayref="y",
                            showarrow=True, arrowhead=3, arrowsize=1.5,
                            arrowwidth=2, arrowcolor="#555")
        fig.add_annotation(x=0.2, y=yr * 0.98, text="<b>y</b>",
                            showarrow=False, font=dict(color="#888", size=14))
        # Origine
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers",
                                 marker=dict(color="white", size=6),
                                 showlegend=False, hoverinfo="skip"))

    def _add_arrow_2d(self, fig, ox, oy, dx, dy, color, name):
        ex, ey = ox + dx, oy + dy
        # Linea del vettore
        fig.add_trace(go.Scatter(
            x=[ox, ex], y=[oy, ey],
            mode="lines",
            line=dict(color=color, width=2.5),
            showlegend=False, hoverinfo="skip"
        ))
        # Punta freccia
        fig.add_annotation(
            x=ex, y=ey, ax=ox, ay=oy,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.5,
            arrowwidth=2.5, arrowcolor=color
        )
        # Etichetta
        fig.add_trace(go.Scatter(
            x=[ex + 0.15], y=[ey + 0.15],
            mode="text",
            text=[f"<b>{name}</b>"],
            textfont=dict(color=color, size=13),
            showlegend=True,
            name=f"{name} ({dx:.2f}, {dy:.2f})",
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"Origine: ({ox:.2f}, {oy:.2f})<br>"
                f"Vettore: ({dx:.2f}, {dy:.2f})<br>"
                f"Punta: ({ex:.2f}, {ey:.2f})<br>"
                f"‖v‖ = {np.sqrt(dx**2+dy**2):.3f}<extra></extra>"
            )
        ))

    # ── 3D ──────────────────────────────────────────────────────────────────

    def _build_3d(self) -> go.Figure:
        fig = go.Figure()

        all_pts = [[0, 0, 0]]
        for v in self.vectors:
            o = (list(v["origin"]) + [0, 0, 0])[:3]
            c = (list(v["coords"]) + [0, 0, 0])[:3]
            all_pts.append([o[i] + c[i] for i in range(3)])

        margin = 1.5
        maxval = max(max(abs(p[i]) for p in all_pts) for i in range(3)) + margin

        # Assi
        self._add_axes_3d(fig, maxval)

        # Piano XY semi-trasparente
        self._add_plane_3d(fig, maxval)

        # Vettori
        for v in self.vectors:
            o = (list(v["origin"]) + [0, 0, 0])[:3]
            c = (list(v["coords"]) + [0, 0, 0])[:3]
            self._add_arrow_3d(fig, o, c, v["color"], v["name"])

        fig.update_layout(
            **self._base_layout(),
            scene=dict(
                xaxis=dict(range=[-maxval, maxval], backgroundcolor="#0f1117",
                           gridcolor="#2a2f45", zerolinecolor="#555",
                           tickfont=dict(color="#aaa"), title="x"),
                yaxis=dict(range=[-maxval, maxval], backgroundcolor="#0f1117",
                           gridcolor="#2a2f45", zerolinecolor="#555",
                           tickfont=dict(color="#aaa"), title="y"),
                zaxis=dict(range=[-maxval, maxval], backgroundcolor="#0f1117",
                           gridcolor="#2a2f45", zerolinecolor="#555",
                           tickfont=dict(color="#aaa"), title="z"),
                bgcolor="#0f1117",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                aspectmode="cube",
            )
        )
        return fig

    def _add_axes_3d(self, fig, r):
        for axis, color, label in [
            ([[-r, 0, 0], [r, 0, 0]], "#e74c3c", "x"),
            ([[0, -r, 0], [0, r, 0]], "#2ecc71", "y"),
            ([[0, 0, -r], [0, 0, r]], "#3498db", "z"),
        ]:
            fig.add_trace(go.Scatter3d(
                x=[axis[0][0], axis[1][0]],
                y=[axis[0][1], axis[1][1]],
                z=[axis[0][2], axis[1][2]],
                mode="lines+text",
                line=dict(color=color, width=4),
                text=["", f"<b>{label}</b>"],
                textfont=dict(color=color, size=14),
                showlegend=False, hoverinfo="skip"
            ))

    def _add_plane_3d(self, fig, r):
        grid = np.linspace(-r, r, 3)
        x, y = np.meshgrid(grid, grid)
        z = np.zeros_like(x)
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            colorscale=[[0, "rgba(50,80,120,0.12)"], [1, "rgba(50,80,120,0.12)"]],
            showscale=False, hoverinfo="skip", name="Piano XY"
        ))

    def _add_arrow_3d(self, fig, origin, coords, color, name):
        ox, oy, oz = origin
        dx, dy, dz = coords
        ex, ey, ez = ox + dx, oy + dy, oz + dz
        norm = float(np.linalg.norm([dx, dy, dz]))

        # Linea
        fig.add_trace(go.Scatter3d(
            x=[ox, ex], y=[oy, ey], z=[oz, ez],
            mode="lines",
            line=dict(color=color, width=6),
            showlegend=False, hoverinfo="skip"
        ))
        # Punta (cono)
        fig.add_trace(go.Cone(
            x=[ex], y=[ey], z=[ez],
            u=[dx * 0.001], v=[dy * 0.001], w=[dz * 0.001],
            colorscale=[[0, color], [1, color]],
            showscale=False, sizemode="absolute", sizeref=0.25,
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"Origine: ({ox:.2f},{oy:.2f},{oz:.2f})<br>"
                f"Vettore: ({dx:.2f},{dy:.2f},{dz:.2f})<br>"
                f"‖v‖ = {norm:.3f}<extra></extra>"
            ),
            name=name
        ))
        # Etichetta
        fig.add_trace(go.Scatter3d(
            x=[ex + 0.1], y=[ey + 0.1], z=[ez + 0.1],
            mode="text",
            text=[f"<b>{name}</b>"],
            textfont=dict(color=color, size=13),
            showlegend=True,
            name=f"{name} ({dx:.2f},{dy:.2f},{dz:.2f})",
            hoverinfo="skip"
        ))

    # ── Layout comune ────────────────────────────────────────────────────────

    def _base_layout(self) -> dict:
        return dict(
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            font=dict(color="#e0e0e0", family="monospace"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=620,
            legend=dict(
                bgcolor="#1e2130", bordercolor="#333", borderwidth=1,
                font=dict(color="#ccc", size=11),
                orientation="v", x=1.01, y=1
            ),
            uirevision="stable",
        )
