# Vettoriale

Un'app web locale in Python per creare e visualizzare vettori in uno spazio 2D e 3D, con supporto alle principali operazioni dell'algebra lineare.

---

## Cosa fa

Permette di costruire interattivamente un piano degli assi cartesiani, popolare lo spazio con vettori personalizzati e analizzarli visivamente e matematicamente.

Le funzionalità principali sono:

- **Aggiunta di vettori** — ogni vettore ha nome, coordinate, colore e origine personalizzabili
- **Visualizzazione 2D e 3D** — switch immediato tra le due modalità, con frecce, griglia, assi e piano XY
- **Hover interattivo** — passando sul vettore si vedono coordinate, norma e punto di applicazione
- **Operazioni tra vettori** — somma, differenza, prodotto scalare e angolo tra due vettori
- **Aggiunta del risultato** — il vettore risultante da un'operazione può essere aggiunto direttamente al piano
- **Gestione vettori** — rimozione singola o totale


## Avvio

```bash
pip install -r requirements.txt
streamlit run app.py
```

Apre automaticamente → **http://localhost:8501**

---


