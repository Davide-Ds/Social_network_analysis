# Roadmap Incrementale per Analisi della Diffusione dei Retweet in Chiave Frattale

Questo documento descrive una serie di obiettivi incrementali per analizzare e modellare la diffusione dei retweet, integrando tecniche di Machine Learning e metriche ispirate al concetto del triangolo di Sierpinski.

---

## 1. Preparazione e Preprocessing dei Dati
**Obiettivo:**  
Assicurarsi che i dati (tweet, etichette, file tree) siano puliti e strutturati correttamente per l'analisi.

**Task:**
- Verificare la corretta creazione dei nodi `Tweet` e delle relazioni `RETWEETS` in Neo4j.
- Estrarre ed esportare la struttura dell'albero di diffusione (ad esempio, in formato JSON o CSV).

**Metriche:**
- Numero di nodi e relazioni.
- Percentuale di retweet processati.

---

## 2. Analisi Esplorativa e Visualizzazione della Rete
**Obiettivo:**  
Comprendere le caratteristiche della rete di diffusione attraverso visualizzazioni e metriche grafiche.

**Task:**
- Creare query Cypher per visualizzare l'albero di diffusione di un tweet rappresentativo.
- Utilizzare strumenti (ad es. Neo4j Bloom o librerie Python come NetworkX e matplotlib) per visualizzare il grafo.

**Metriche:**
- Distribuzione dei delay.
- Numero di livelli dell'albero.
- Grado medio dei nodi.

---

## 3. Calcolo della Dimensione Frattale della Rete
**Obiettivo:**  
Misurare la dimensione frattale (ad es. con il metodo del box-counting) per valutare l’autosimilarità dell’albero dei retweet.

**Task:**
- Esportare la struttura del grafo e implementare un algoritmo di box-counting per calcolare la dimensione di Hausdorff o una stima della dimensione frattale.
- Confrontare il valore ottenuto con quello del triangolo di Sierpinski (~1.585) e analizzare le differenze.

**Metriche:**
- Valore della dimensione frattale.
- Variazioni in funzione di differenti tweet o cluster.

---

## 4. Estrazione di Feature per il Machine Learning
**Obiettivo:**  
Estrarre caratteristiche strutturali e temporali dalla rete di diffusione per alimentare modelli ML.

**Task:**
- Calcolare metriche di rete (es. grado, centralità, clustering coefficient) per ogni tweet o per i percorsi di diffusione.
- Estrarre serie temporali del delay e pattern ricorsivi (es. conteggio dei retweet per livello dell'albero).

**Metriche:**
- Feature statistiche (media, varianza, distribuzione).
- Feature topologiche.

---

## 5. Modellazione ML per la Predizione/Clustering della Diffusione
**Obiettivo:**  
Sviluppare e confrontare modelli ML che utilizzino le feature estratte per classificare o clusterizzare la diffusione dei rumor.

**Task:**
- **Classificazione:** Utilizzare algoritmi (SVM, Random Forest, reti neurali) per predire se un tweet sarà rumor o no basandosi sulle feature estratte.
- **Clustering:** Utilizzare tecniche di clustering (K-Means, DBSCAN, clustering gerarchico) per identificare pattern ricorrenti e gruppi omogenei nella diffusione.
- **Modelli Frattali:** Valutare se modelli basati su regole ricorsive (ispirati al Sierpinski) possano essere implementati per simulare la propagazione e confrontare con i dati reali.

**Metriche:**
- Accuratezza, F1-score, precision, recall (per classificazione).
- Silhouette score, Davies-Bouldin index (per clustering).
- Confronto tra la simulazione frattale e la diffusione reale (es. differenza tra dimensione frattale simulata e quella osservata).

---

## 6. Valutazione e Analisi dei Risultati
**Obiettivo:**  
Interpretare i risultati dei modelli e delle metriche frattali per verificare l’ipotesi di autosimilarità nella diffusione dei retweet.

**Task:**
- Confrontare le performance dei modelli ML con e senza feature frattali.
- Discutere se la struttura della rete supporta l’ipotesi di diffusione frattale, evidenziando eventuali pattern ricorsivi.
- Proporre miglioramenti basati su eventuali anomalie o discrepanze riscontrate.

**Metriche:**
- Report comparativo.
- Analisi dei residui e analisi statistica delle feature.

---

## 7. Documentazione e Presentazione dei Risultati
**Obiettivo:**  
Preparare una relazione dettagliata che documenti i modelli, le tecniche e le metriche utilizzate, evidenziando le scoperte sul comportamento frattale della diffusione.

**Task:**
- Creare grafici, diagrammi e tabelle che mostrino la struttura della rete e i risultati dei modelli ML.
- Redigere un report finale che includa metodologia, risultati, analisi critica e possibili sviluppi futuri.

**Metriche:**
- Qualità della documentazione.
- Chiarezza nell’esposizione dei risultati.
- Comparazioni con modelli di riferimento (come il triangolo di Sierpinski).

---

Questa roadmap ti guiderà attraverso fasi incrementali che ti permetteranno di:
- Esplorare e visualizzare i dati,
- Calcolare metriche frattali,
- Estrarre feature significative,
- Sviluppare e valutare modelli ML,
- E infine documentare e presentare i risultati.

---

*Fine del documento.*
