# 📐 Vettoriale

Un'app web locale in Python per creare e visualizzare vettori in uno spazio 2D e 3D, con supporto alle principali operazioni dell'algebra lineare e alla visualizzazione di embedding NLP.

---

## Cosa fa

### 🗺️ Spazio Vettoriale
Permette di costruire interattivamente un piano degli assi cartesiani, popolare lo spazio con vettori personalizzati e analizzarli visivamente e matematicamente.

- Aggiunta di vettori con nome, coordinate, colore e origine personalizzabili
- Switch immediato tra modalità 2D e 3D
- Operazioni tra vettori: somma, differenza, prodotto scalare, angolo
- Aggiunta del vettore risultante direttamente al piano

![Registrazione 2026-03-10 150349](https://github.com/user-attachments/assets/e3966cc8-6fa9-42ac-a5e8-f9a1b4b76075)
![Registrazione 2026-03-10 153012](https://github.com/user-attachments/assets/9a36154d-63b9-412e-b19a-66658e5809c9)

### 🧠 NLP — Embedding Token
Un laboratorio per *capire* gli embedding partendo da un corpus di testo: il
sistema tokenizza, costruisce vettori TF-IDF per ogni token, li aggiorna in base
al contesto (finestra scorrevole, peso `alpha`) e li proietta in 2D/3D con
PCA o t-SNE.

- **Token simili** — i vicini più prossimi per similarità coseno
- **Animazione dell'evoluzione** — come i vettori si spostano frase dopo frase
- **➗ Aritmetica semantica** — risolve analogie del tipo `A − B + C ≈ ?`
  (es. `re − uomo + donna ≈ regina`) e ne disegna il parallelogramma di frecce
- **🔥 Matrice di similarità** — heatmap coseno calcolata sui vettori interi,
  per vedere il clustering reale che la proiezione 2D/3D nasconde
- **📊 Varianza spiegata** — quanto della struttura sopravvive alla riduzione PCA

> Gli embedding sono TF-IDF didattici: le analogie sono approssimative, l'obiettivo
> è *mostrare il meccanismo* degli embedding, non eguagliare modelli neurali.

## Avvio

```bash
pip install -r requirements.txt
streamlit run app.py
```
## Licenza

Distribuito sotto licenza [GPL v3](LICENSE).  
Puoi fare fork e modificare liberamente, ma le versioni derivate devono
mantenere la stessa licenza e citare l'autore originale.
