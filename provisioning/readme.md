# Grafana

Per il servizio di monitoring dell'applicazione è stato scelto Grafana.
* Di default il container avvia un'istanza di Grafana con l'utente 'worker' e password 'password'
    * La connessione verso il databse locale viene istanziata di default in accordo con la documentazione di Grafana [Configure the Prometheus data source](https://grafana.com/docs/grafana/latest/datasources/prometheus/configure/).
* Di default viene precaricata una Dashboard per monitorare le statistiche delle predizioni fatte dal modello nell'applicazione principale. 

È possibile modificare il tutto nelle sottocartelle qui presenti.
Sarebbe consigliabile utilizzare un databse più strutturato come Postgress, ma per questo esercizio SQLite rappresenta un buona soluzione.