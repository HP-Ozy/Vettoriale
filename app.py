import streamlit as st
import numpy as np
import plotly.graph_objects as go
from vector_space import VectorSpace

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
</style>
""", unsafe_allow_html=True)

# ── Stato sessione ──────────────────────────────────────────────────────────
if "space" not in st.session_state:
    st.session_state.space = VectorSpace()
if "mode" not in st.session_state:
    st.session_state.mode = "2D"

space: VectorSpace = st.session_state.space

# ── Header ──────────────────────────────────────────────────────────────────
st.title("📐 Piano degli Assi — Visualizzatore Vettoriale")

col_ctrl, col_plot = st.columns([1, 2.6])

# ── Pannello controlli ───────────────────────────────────────────────────────
with col_ctrl:

    # Modalità 2D / 3D
    st.subheader("⚙️ Modalità")
    mode = st.radio("Dimensioni", ["2D", "3D"], horizontal=True, key="mode_radio",
                    index=0 if st.session_state.mode == "2D" else 1)
    if mode != st.session_state.mode:
        st.session_state.mode = mode
        st.rerun()

    st.divider()

    # Aggiungi vettore
    st.subheader("➕ Nuovo Vettore")
    v_name = st.text_input("Nome vettore", value=f"v{len(space.vectors)+1}")
    v_color = st.color_picker("Colore", value="#00d4ff")

    if st.session_state.mode == "2D":
        c1, c2 = st.columns(2)
        vx = c1.number_input("x", value=1.0, step=0.5, format="%.2f")
        vy = c2.number_input("y", value=1.0, step=0.5, format="%.2f")
        coords = [vx, vy]
        # Punto di origine
        ox = c1.number_input("ox", value=0.0, step=0.5, format="%.2f", label_visibility="collapsed") if False else 0.0
        oy = 0.0
        origin = [0.0, 0.0]
    else:
        c1, c2, c3 = st.columns(3)
        vx = c1.number_input("x", value=1.0, step=0.5, format="%.2f")
        vy = c2.number_input("y", value=0.0, step=0.5, format="%.2f")
        vz = c3.number_input("z", value=1.0, step=0.5, format="%.2f")
        coords = [vx, vy, vz]
        origin = [0.0, 0.0, 0.0]

    # Origine personalizzata
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

    # Lista vettori presenti
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

    st.divider()

    # Operazioni tra vettori
    if len(space.vectors) >= 2:
        st.subheader("🔢 Operazioni")
        names = [v["name"] for v in space.vectors]
        va_name = st.selectbox("Vettore A", names, key="op_a")
        vb_name = st.selectbox("Vettore B", names, key="op_b", index=min(1, len(names)-1))
        op = st.selectbox("Operazione", ["Somma (A+B)", "Differenza (A−B)", "Prodotto scalare", "Angolo tra A e B"])

        va = np.array(space.get_by_name(va_name)["coords"])
        vb = np.array(space.get_by_name(vb_name)["coords"])

        # Allinea dimensioni
        n = max(len(va), len(vb))
        va = np.pad(va, (0, n - len(va)))
        vb = np.pad(vb, (0, n - len(vb)))

        if op == "Somma (A+B)":
            res = va + vb
            st.info(f"**A+B** = ({', '.join(f'{x:.3f}' for x in res)})")
            if st.button("📥 Aggiungi risultato come vettore"):
                space.add_vector(f"{va_name}+{vb_name}", res[:3 if st.session_state.mode=="3D" else 2].tolist(), [0]*len(res[:3 if st.session_state.mode=="3D" else 2]), "#ff9f43")
                st.rerun()
        elif op == "Differenza (A−B)":
            res = va - vb
            st.info(f"**A−B** = ({', '.join(f'{x:.3f}' for x in res)})")
            if st.button("📥 Aggiungi risultato come vettore"):
                space.add_vector(f"{va_name}-{vb_name}", res[:3 if st.session_state.mode=="3D" else 2].tolist(), [0]*len(res[:3 if st.session_state.mode=="3D" else 2]), "#ff6b81")
                st.rerun()
        elif op == "Prodotto scalare":
            dot = float(np.dot(va, vb))
            st.metric("A · B", f"{dot:.4f}")
        elif op == "Angolo tra A e B":
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            if na > 0 and nb > 0:
                cos_a = np.clip(np.dot(va, vb) / (na * nb), -1, 1)
                angle_deg = float(np.degrees(np.arccos(cos_a)))
                st.metric("Angolo", f"{angle_deg:.2f}°")
            else:
                st.warning("Uno dei vettori è nullo.")

# ── Grafico ──────────────────────────────────────────────────────────────────
with col_plot:
    st.subheader("🗺️ Piano degli Assi")
    fig = space.build_figure(mode=st.session_state.mode)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

    # Info aggiuntive
    if space.vectors:
        st.caption("💡 Trascina per ruotare (3D) · Scroll per zoom · Doppio click per reset vista")
