# Grafana

Per il servizio di monitoring dell'applicazione è stato scelto Grafana.
* Di default il container avvia un'istanza di Grafana con l'utente 'user' e password 'password'
    * La connessione verso il databse locale viene istanziata di default in accordo con la documentazione di Grafana [Configure the Prometheus data source](https://grafana.com/docs/grafana/latest/datasources/prometheus/configure/).
* Di default viene precaricata una Dashboard per monitorare le statistiche delle predizioni fatte dal modello nell'applicazione principale. 

È possibile modificare il tutto nelle sottocartelle qui presenti.
Sarebbe consigliabile utilizzare un databse più strutturato come Postgress, ma per questo esercizio SQLite rappresenta un buona soluzione.

> **Nota MLOps Importante**: Il database SQLite locale (`sentiment_logs.db`) serve **esclusivamente** per alimentare le dashboard di Grafana e fornire statistiche visive ai manager (es. numero di recensioni positive al giorno). 
> **Non viene utilizzato per ri-addestrare il modello.** Retrainare un modello sulle sue stesse predizioni (senza che un umano le abbia corrette) causerebbe un degrado delle performance. Il retraining avviene prelevando un dataset esterno certificato (Ground Truth) da Kaggle.