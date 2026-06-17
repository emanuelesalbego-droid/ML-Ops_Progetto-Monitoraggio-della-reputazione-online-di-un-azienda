import sys
import os

# Aggiungi il percorso root del progetto al PYTHONPATH per importare app.training
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.training import retraining
from transformers import AutoTokenizer

if __name__ == "__main__":
    # Questo script funge da "ponte" (Entrypoint CLI) per lanciare il training senza dover avviare 
    # l'intera applicazione FastAPI. È fondamentale per la CI/CD (GitHub Actions), perché permette
    # ai server cloud di eseguire solo la logica matematica dell'addestramento e poi spegnersi.
    
    MODEL_NAME = os.getenv("MODEL_NAME", "cardiffnlp/twitter-roberta-base-sentiment-latest")
    print(f"Avvio retraining per il modello: {MODEL_NAME}")
    
    # Caricamento del tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"Errore nel caricamento del tokenizer: {e}")
        sys.exit(1)
        
    # Esecuzione della logica di training originale
    try:
        res = retraining(MODEL_NAME, tokenizer)
        print("\nRetraining completato con successo. Risultati:")
        print(res)
    except Exception as e:
        print(f"\nErrore durante il retraining: {e}")
        sys.exit(1)
