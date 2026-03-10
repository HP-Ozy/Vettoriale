import streamlit as st
import numpy as np
import plotly.graph_objects as go
from vector_space import VectorSpace
from nlp_tokenizer import Tokenizer
from nlp_embedder import TokenEmbedder
from nlp_reducer import DimensionReducer
from nlp_visualizer import EmbeddingVisualizer

st.set_page_config(page_title="Piano degli Assi", layout="wide", page_icon="📐")

st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { color: #e0e0e0; }
    .vector-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
        border-left: 4px solid;
    }
    .token-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        margin: 3px;
        font-size: 0.8rem;
        font-family: monospace;
    }
    .info-box {
        background: #1e2130;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        border-left: 3px solid #00d4ff;
    }
</style>
""", unsafe_allow_html=True)

# ── Stato sessione ──────────────────────────────────────────────────────────
if "space" not in st.session_state:
    st.session_state.space = VectorSpace()
if "mode" not in st.session_state:
    st.session_state.mode = "2D"
if "embedder" not in st.session_state:
    st.session_state.embedder = None
if "reducer" not in st.session_state:
    st.session_state.reducer = None
if "visualizer" not in st.session_state:
    st.session_state.visualizer = EmbeddingVisualizer()
if "nlp_coords" not in st.session_state:
    st.session_state.nlp_coords = None
if "nlp_mode" not in st.session_state:
    st.session_state.nlp_mode = "2D"

st.title("📐 Piano degli Assi — Visualizzatore Vettoriale")

# ── Tab principali ──────────────────────────────────────────────────────────
tab_vettori, tab_nlp = st.tabs(["🗺️ Spazio Vettoriale", "🧠 NLP — Embedding Token"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SPAZIO VETTORIALE
# ══════════════════════════════════════════════════════════════════════════════
with tab_vettori:
    space: VectorSpace = st.session_state.space

    col_ctrl, col_plot = st.columns([1, 2.6])

    with col_ctrl:
        st.subheader("⚙️ Modalità")
        mode = st.radio("Dimensioni", ["2D", "3D"], horizontal=True, key="mode_radio",
                        index=0 if st.session_state.mode == "2D" else 1)
        if mode != st.session_state.mode:
            st.session_state.mode = mode
            st.rerun()

        st.divider()
        st.subheader("➕ Nuovo Vettore")
        v_name = st.text_input("Nome vettore", value=f"v{len(space.vectors)+1}")
        v_color = st.color_picker("Colore", value="#00d4ff")

        if st.session_state.mode == "2D":
            c1, c2 = st.columns(2)
            vx = c1.number_input("x", value=1.0, step=0.5, format="%.2f")
            vy = c2.number_input("y", value=1.0, step=0.5, format="%.2f")
            coords = [vx, vy]
        else:
            c1, c2, c3 = st.columns(3)
            vx = c1.number_input("x", value=1.0, step=0.5, format="%.2f")
            vy = c2.number_input("y", value=0.0, step=0.5, format="%.2f")
            vz = c3.number_input("z", value=1.0, step=0.5, format="%.2f")
            coords = [vx, vy, vz]

        with st.expander("📍 Origine personalizzata"):
            if st.session_state.mode == "2D":
                o1, o2 = st.columns(2)
                origin = [o1.number_input("ox", value=0.0, step=0.5, format="%.2f"),
                          o2.number_input("oy", value=0.0, step=0.5, format="%.2f")]
            else:
                o1, o2, o3 = st.columns(3)
                origin = [o1.number_input("ox", value=0.0, step=0.5, format="%.2f"),
                          o2.number_input("oy", value=0.0, step=0.5, format="%.2f"),
                          o3.number_input("oz", value=0.0, step=0.5, format="%.2f")]

        if st.button("➕ Aggiungi vettore", use_container_width=True, type="primary"):
            if v_name.strip():
                space.add_vector(v_name.strip(), coords, origin, v_color)
                st.success(f"Vettore **{v_name}** aggiunto!")
                st.rerun()
            else:
                st.warning("Inserisci un nome!")

        st.divider()
        st.subheader(f"📋 Vettori ({len(space.vectors)})")
        if not space.vectors:
            st.caption("Nessun vettore ancora. Aggiungine uno! 👆")
        else:
            for i, v in enumerate(space.vectors):
                arr = np.array(v["coords"])
                norm = float(np.linalg.norm(arr))
                coords_str = ", ".join(f"{c:.2f}" for c in v["coords"])
                st.markdown(f"""
                <div class="vector-card" style="border-color:{v['color']}">
                    <b style="color:{v['color']}">{v['name']}</b><br>
                    <small style="color:#aaa">({coords_str}) &nbsp;|&nbsp; ‖v‖ = {norm:.3f}</small>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"🗑 Rimuovi {v['name']}", key=f"del_{i}", use_container_width=True):
                    space.remove_vector(i)
                    st.rerun()

        if space.vectors:
            st.divider()
            if st.button("🗑 Rimuovi tutti", use_container_width=True):
                space.clear()
                st.rerun()

        if len(space.vectors) >= 2:
            st.divider()
            st.subheader("🔢 Operazioni")
            names = [v["name"] for v in space.vectors]
            va_name = st.selectbox("Vettore A", names, key="op_a")
            vb_name = st.selectbox("Vettore B", names, key="op_b", index=min(1, len(names)-1))
            op = st.selectbox("Operazione", ["Somma (A+B)", "Differenza (A−B)", "Prodotto scalare", "Angolo tra A e B"])

            va = np.array(space.get_by_name(va_name)["coords"])
            vb = np.array(space.get_by_name(vb_name)["coords"])
            n = max(len(va), len(vb))
            va = np.pad(va, (0, n - len(va)))
            vb = np.pad(vb, (0, n - len(vb)))

            if op == "Somma (A+B)":
                res = va + vb
                st.info(f"**A+B** = ({', '.join(f'{x:.3f}' for x in res)})")
                if st.button("📥 Aggiungi come vettore"):
                    dim = 3 if st.session_state.mode == "3D" else 2
                    space.add_vector(f"{va_name}+{vb_name}", res[:dim].tolist(), [0]*dim, "#ff9f43")
                    st.rerun()
            elif op == "Differenza (A−B)":
                res = va - vb
                st.info(f"**A−B** = ({', '.join(f'{x:.3f}' for x in res)})")
                if st.button("📥 Aggiungi come vettore"):
                    dim = 3 if st.session_state.mode == "3D" else 2
                    space.add_vector(f"{va_name}-{vb_name}", res[:dim].tolist(), [0]*dim, "#ff6b81")
                    st.rerun()
            elif op == "Prodotto scalare":
                st.metric("A · B", f"{float(np.dot(va, vb)):.4f}")
            elif op == "Angolo tra A e B":
                na, nb = np.linalg.norm(va), np.linalg.norm(vb)
                if na > 0 and nb > 0:
                    cos_a = np.clip(np.dot(va, vb) / (na * nb), -1, 1)
                    st.metric("Angolo", f"{float(np.degrees(np.arccos(cos_a))):.2f}°")
                else:
                    st.warning("Uno dei vettori è nullo.")

    with col_plot:
        st.subheader("🗺️ Piano degli Assi")
        fig = space.build_figure(mode=st.session_state.mode)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
        if space.vectors:
            st.caption("💡 Trascina per ruotare (3D) · Scroll per zoom · Doppio click per reset")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — NLP EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════
with tab_nlp:
    st.markdown("### 🧠 Tokenizzazione → Embedding → Riduzione → Visualizzazione")
    st.markdown(
        '<div class="info-box">Inserisci uno o più testi: il sistema li tokenizza, '
        'costruisce vettori TF-IDF per ogni token, li aggiorna con il contesto '
        'e li proietta in 2D/3D.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 2])

    # ── Pannello configurazione ─────────────────────────────────────────────
    with left:

        # --- Corpus ---
        st.subheader("📝 Corpus iniziale")
        default_corpus = (
            "Il gatto dorme sul divano vicino alla finestra\n"
            "Il cane corre nel parco con il suo padrone\n"
            "La macchina elettrica percorre la strada velocemente\n"
            "Il computer elabora grandi quantità di dati\n"
            "La rete neurale impara dai dati di addestramento"
        )
        corpus_text = st.text_area(
            "Un testo per riga (corpus di addestramento)",
            value=default_corpus,
            height=160,
            help="Ogni riga è un documento. Il sistema apprende i vettori da questi testi.",
        )

        st.divider()

        # --- Parametri ---
        st.subheader("⚙️ Parametri")
        c1, c2 = st.columns(2)
        reduction_method = c1.selectbox("Riduzione", ["PCA", "t-SNE"], key="red_method")
        nlp_mode = c2.radio("Visualizza", ["2D", "3D"], horizontal=True, key="nlp_mode_radio")
        context_window = st.slider("Finestra contesto", 1, 5, 2,
                                   help="Quanti token a sinistra/destra influenzano l'aggiornamento")
        alpha = st.slider("Peso contesto (alpha)", 0.0, 1.0, 0.3, 0.05,
                          help="0 = nessun aggiornamento | 1 = solo contesto")
        remove_sw = st.checkbox("Rimuovi stop-word", value=True)

        if st.button("🚀 Addestra embedding", use_container_width=True, type="primary"):
            corpus_lines = [l.strip() for l in corpus_text.split("\n") if l.strip()]
            if not corpus_lines:
                st.warning("Inserisci almeno un testo nel corpus.")
            else:
                with st.spinner("Addestramento in corso..."):
                    tokenizer = Tokenizer(remove_stopwords=remove_sw)
                    embedder = TokenEmbedder(context_window=context_window, alpha=alpha)
                    embedder.tokenizer = tokenizer
                    embedder.fit(corpus_lines)

                    reducer = DimensionReducer(
                        method=reduction_method,
                        n_components=3 if nlp_mode == "3D" else 2,
                    )

                    if len(embedder.embeddings) >= 2:
                        coords = reducer.fit_transform(embedder.embeddings)
                    else:
                        coords = {}

                    st.session_state.embedder = embedder
                    st.session_state.reducer = reducer
                    st.session_state.nlp_coords = coords
                    st.session_state.nlp_mode = nlp_mode
                    st.session_state.visualizer = EmbeddingVisualizer()
                st.success(f"✅ {len(embedder.embeddings)} token trovati!")
                st.rerun()

        st.divider()

        # --- Aggiornamento contestuale ---
        st.subheader("🔄 Aggiornamento contestuale")
        context_text = st.text_input(
            "Nuova frase di contesto",
            placeholder="es: il gatto veloce impara dai dati",
            help="I token di questa frase aggiornano i loro vettori in base al vicinato.",
        )
        if st.button("🔄 Aggiorna vettori", use_container_width=True,
                     disabled=st.session_state.embedder is None):
            if context_text.strip():
                with st.spinner("Aggiornamento..."):
                    emb: TokenEmbedder = st.session_state.embedder
                    red: DimensionReducer = st.session_state.reducer
                    emb.update_context(context_text)
                    coords = red.transform(emb.embeddings)
                    st.session_state.nlp_coords = coords
                st.success("Vettori aggiornati!")
                st.rerun()
            else:
                st.warning("Inserisci una frase.")

        st.divider()

        # --- Similarità ---
        st.subheader("🔍 Token simili")
        if st.session_state.embedder and st.session_state.embedder.embeddings:
            tokens_list = sorted(st.session_state.embedder.embeddings.keys())
            query_tok = st.selectbox("Token", tokens_list, key="sim_query")
            top_n = st.slider("Top N", 1, 10, 5)
            if st.button("🔍 Cerca simili", use_container_width=True):
                similar = st.session_state.embedder.get_similar(query_tok, top_n)
                st.markdown(f"**Token simili a «{query_tok}»:**")
                for tok, score in similar:
                    bar = "█" * int(score * 20)
                    st.markdown(
                        f'<div class="token-pill" style="background:#1e2130;color:#aaa">'
                        f'<span style="color:#00d4ff">{tok}</span> '
                        f'<span style="color:#666">{bar}</span> '
                        f'<span style="color:#ffd93d">{score:.3f}</span></div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("Addestra prima gli embedding 👆")

    # ── Pannello visualizzazione ────────────────────────────────────────────
    with right:
        viz: EmbeddingVisualizer = st.session_state.visualizer
        coords = st.session_state.nlp_coords
        emb_state: TokenEmbedder = st.session_state.embedder
        nlp_mode_cur = st.session_state.get("nlp_mode", "2D")

        view_tab1, view_tab2, view_tab3 = st.tabs(
            ["📍 Scatter statico", "🎬 Animazione evoluzione", "🔢 Token & coordinate"]
        )

        # Scatter statico
        with view_tab1:
            if coords and len(coords) >= 2:
                highlight_tok = None
                sim_scores = None

                # Se l'utente ha appena cercato simili
                if "sim_query" in st.session_state and emb_state:
                    ht = st.session_state.sim_query
                    if ht in coords:
                        similar = emb_state.get_similar(ht, 10)
                        highlight_tok = [ht]
                        sim_scores = {t: s for t, s in similar if t in coords}

                fig_static = viz.plot_static(
                    coords,
                    mode=nlp_mode_cur,
                    title="Embedding Token — Spazio Ridotto",
                    highlight=highlight_tok,
                    similarity_scores=sim_scores,
                )
                st.plotly_chart(fig_static, use_container_width=True)
                st.caption(
                    f"**{len(coords)} token** proiettati con "
                    f"**{st.session_state.get('red_method', 'PCA')}** in "
                    f"**{nlp_mode_cur}**"
                )
            else:
                st.info("👈 Addestra gli embedding per vedere lo scatter.")

        # Animazione
        with view_tab2:
            if emb_state and len(emb_state.history) >= 2:
                tokens_in_coords = list(coords.keys()) if coords else []
                red_anim = DimensionReducer(
                    method="PCA",
                    n_components=3 if nlp_mode_cur == "3D" else 2,
                )
                snapshots = red_anim.reduce_snapshots(
                    emb_state.history, tokens_in_coords
                )
                if snapshots:
                    fig_anim = viz.plot_animated(snapshots, mode=nlp_mode_cur)
                    st.plotly_chart(fig_anim, use_container_width=True)
                    st.caption(
                        f"**{len(emb_state.history)} snapshot** — "
                        "ogni frame mostra come i vettori si spostano nel contesto."
                    )
                else:
                    st.info("Snapshot non sufficienti per l'animazione.")
            elif emb_state and len(emb_state.history) == 1:
                st.info("Aggiungi almeno una frase di contesto per vedere l'evoluzione.")
            else:
                st.info("👈 Addestra gli embedding prima.")

        # Tabella token
        with view_tab3:
            if coords and emb_state:
                import pandas as pd
                rows = []
                for tok, pt in coords.items():
                    row = {"token": tok}
                    for i, ax in enumerate(["x", "y", "z"]):
                        if i < len(pt):
                            row[ax] = round(float(pt[i]), 4)
                    # Norma vettore originale
                    if tok in emb_state.embeddings:
                        row["‖v‖ originale"] = round(
                            float(np.linalg.norm(emb_state.embeddings[tok])), 4
                        )
                    rows.append(row)

                df = pd.DataFrame(rows).sort_values("token")
                st.dataframe(
                    df, use_container_width=True, hide_index=True,
                    column_config={
                        "token": st.column_config.TextColumn("Token"),
                        "x": st.column_config.NumberColumn("x", format="%.4f"),
                        "y": st.column_config.NumberColumn("y", format="%.4f"),
                        "z": st.column_config.NumberColumn("z", format="%.4f"),
                        "‖v‖ originale": st.column_config.NumberColumn("‖v‖", format="%.4f"),
                    }
                )

                # Mostra token colorati
                st.markdown("**Token nel corpus:**")
                html_pills = ""
                for tok in sorted(coords.keys()):
                    col = viz._color_map.get(tok, "#aaa")
                    html_pills += (
                        f'<span class="token-pill" '
                        f'style="background:{col}22;color:{col};border:1px solid {col}55">'
                        f'{tok}</span>'
                    )
                st.markdown(html_pills, unsafe_allow_html=True)
            else:
                st.info("👈 Addestra gli embedding prima.")
