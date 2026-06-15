# Documentazione Servizio API: Sentiment Analysis

In questa cartella è presente il codice del servizio API che gestisce le richieste di predizione e il training del modello.
**Modello di riferimento:** [twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)

L'applicazione si articola principalmente in due file sorgente:
1. **main**
2. **training**

---

## 1. Main

In questo file è presente il cuore del funzionamento del servizio API con i seguenti endpoint:

* **GET /**: Restituisce lo stato del servizio con l'indicazione del modello in uso.
* **POST /predict**: Tramite chiamata POST è possibile inviare una stringa che viene valutata dal modello caricato.
    * Restituisce un dizionario con la categoria (label) che può essere di 3 tipi: **negative**, **neutral** e **positive**, insieme al grado di accuratezza della predizione (score).
    * Registra il risultato della predizione nel database locale **SQLite**, disponibile per l'interfaccia di monitoraggio **Grafana**.
* **GET /retrain**: Inserendo la chiave per Kaggle nella cartella keys/kaggle.json, è possibile eseguire il training del modello base.
    * Il database di default è: [Sentiment Analysis Dataset](https://www.kaggle.com/datasets/mdismielhossenabir/sentiment-analysis).
    * Per semplicità e velocità dell'esercizio, il processo esegue solo un'epoca.
    * Salva i risultati del retraining nel database locale SQLite, consultabili tramite Grafana.
    * I pesi del modello aggiornato vengono versionati giornalmente in locale, permettendo l'implementazione del deploy in HuggingFace (non implementato).
    * **Nota:** Questa funzione blocca le chiamate a /retrain e /predict per evitare errori e rallentamenti del servizio.
* **GET /metrics**: Restituisce il numero di predizioni eseguite.
* **GET /models_list**: Restituisce l'elenco dei modelli creati in locale dal retraining.
* **POST /change_model**: Tramite chiamata POST è possibile passare il nome del modello da caricare per le predizioni o per il retraining.
    * È possibile passare la stringa "base" per caricare il modello di default, o "latest" per caricare l'ultimo modello allenato.

---

## 2. Training

Il modulo training gestisce la logica di addestramento del modello, interfacciandosi con il dataset di Kaggle e salvando i pesi risultanti nelle cartelle di versionamento locale.

---

## 3. Database

Nel database sono disponibili le due tabelle **sentiment_logs** e **training_logs** consultabili tramite l'applicazione Grafana.