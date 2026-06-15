from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import os
from app.training import retraining
import glob

# --- CONFIGURAZIONE DATABASE (Per Grafana) ---
DATABASE_URL=os.getenv("DATABASE_URL", "sqlite:///./opt/data/sentiment_logs.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

#tabelle per le metriche
class SentimentLog(Base):
    __tablename__ = "sentiment_logs"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String)
    label = Column(String)
    score = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class TrainingLog(Base):
    #{'eval_loss': 1.0763753652572632, 'eval_model_preparation_time': 0.0024, 'eval_recall': 0.3464849354375896, 'eval_runtime': 5.3073, 'eval_samples_per_second': 18.842, 'eval_steps_per_second': 4.71}
    __tablename__ = "training_logs"
    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, default="latest")
    eval_loss = Column(Float)
    #eval_model_preparation_time = Column(Float)
    eval_recall = Column(Float)
    eval_runtime = Column(Float)
    eval_samples_per_second = Column(Float)
    eval_steps_per_second = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    #{'eval_loss': 0.003190046874806285, 'eval_recall': 1.0, 'eval_runtime': 5.1447, 'eval_samples_per_second': 19.438, 'eval_steps_per_second': 4.859, 'epoch': 10.0}

Base.metadata.create_all(bind=engine)

# --- SCHEMI PER API ---
class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    label: str
    score: float

# Stato globale dell'applicazione
class AppState:
    is_training = False

state = AppState()

# --- APP FASTAPI ---
app = FastAPI(title="MachineInnovators Sentiment API", version="1.0.0")

MODEL_PATH = "cardiffnlp/twitter-roberta-base-sentiment-latest"

@app.get("/")
def read_root():
    if state.is_training:
        return {"status": "training in progress"}
    return {"status": "online", "model": MODEL_PATH}

# --- CARICAMENTO MODELLO ---
def load_model(model_version: str = "latest"):
    global model, tokenizer, sentiment_task, MODEL_PATH
    
    BASE_DIR = "/opt/versioned_models"
    
    # 1. Cerchiamo l'ultima CARTELLA di versione creata
    if os.path.exists(BASE_DIR) and os.listdir(BASE_DIR):
        # Cerchiamo tutte le sottocartelle che iniziano con 'model_'
        list_of_versions = [os.path.join(BASE_DIR, d) for d in os.listdir(BASE_DIR) 
                            if os.path.isdir(os.path.join(BASE_DIR, d)) and d.startswith("model_")]
        
        if list_of_versions and model_version != "latest" and model_version != "base":
            # Prendiamo la più recente per data di creazione
            try:
                MODEL_PATH = os.path.join(BASE_DIR, f"{model_version}")
                if not os.path.exists(MODEL_PATH):
                    raise ValueError(f"Versione specificata '{model_version}' non trovata localmente.")
            except Exception as e:
                return f"Errore nel caricamento della versione specificata: {e}"
            print(f"Caricamento modello LOCALE dalla CARTELLA: {MODEL_PATH}")
        elif list_of_versions and model_version == "latest":
            MODEL_PATH = max(list_of_versions, key=os.path.getmtime)
            print(f"Caricamento modello LOCALE dalla CARTELLA più recente: {MODEL_PATH}")
        else:
            MODEL_PATH = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    elif model_version == "base":
        MODEL_PATH = "cardiffnlp/twitter-roberta-base-sep2022"
    else:
        MODEL_PATH = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        print(f"Nessuna versione locale trovata. Uso fallback: {MODEL_PATH}")

    try:
        # Carichiamo dalla cartella (Hugging Face leggerà config.json e i pesi automaticamente)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        sentiment_task = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        print("Sistema aggiornato con successo.")
            
    except Exception as e:
        print(f"Errore critico nel caricamento: {e}")
        # Fallback di sicurezza se la cartella è corrotta
        MODEL_PATH = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        sentiment_task = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

load_model()

@app.post("/predict", response_model=SentimentResponse)
async def predict(request: SentimentRequest, background_tasks: BackgroundTasks):
    if state.is_training:
        return {"status": "training in progress, please try later."}
    try:
        # 1. Inferenza
        result = sentiment_task(request.text)[0]
        
        # 2. Salvataggio nel DB per il monitoraggio di Grafana
        db = SessionLocal()
        new_log = SentimentLog(
            text=request.text,
            label=result['label'],
            score=result['score']
        )
        db.add(new_log)
        db.commit()
        db.close()

        return SentimentResponse(label=result['label'], score=result['score'])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- LOGICA DI RETRAINING (Placeholder per pipeline) ---
def run_retraining():
    """Esegue il training e gestisce il flag di stato"""
    try:
        state.is_training = True
            
        print("Stato impostato su BUSY. Inizio training...")
        metrics = retraining(MODEL_PATH, tokenizer) # Deve salvare in 'app/saved_model'
        
        # 2. Salvataggio nel DB tradizionale (per Grafana)
        db = SessionLocal()
        new_log = TrainingLog(
            version=f"model_{datetime.datetime.now().strftime('%Y%m%d')}", # Usiamo il Run ID come versione
            eval_loss=metrics.get('eval_loss'),
            eval_recall=metrics.get('eval_recall'),
            eval_runtime=metrics.get('eval_runtime'),
            eval_samples_per_second=metrics.get('eval_samples_per_second'),
            eval_steps_per_second=metrics.get('eval_steps_per_second'),
        )
        db.add(new_log)
        db.commit()
        db.close()
        
        # 4. RICARICAMENTO DINAMICO: Aggiorniamo la pipeline in memoria
        print("Ricaricamento modello post-training...")
        load_model()
    except Exception as e:
        print(f"Errore durante il training: {e}")
    finally:
        state.is_training = False
        print("Stato impostato su IDLE. Inferenza ripristinata.")

@app.post("/retrain")
async def retrain(background_tasks: BackgroundTasks):
    if state.is_training:
        return {"status": "training in progress, please try later."}
    background_tasks.add_task(run_retraining)
    return {"message": "Processo di retraining avviato in background."}

@app.get("/metrics")
def get_metrics():
    # Endpoint utile per capire quante predizioni abbiamo loggato
    db = SessionLocal()
    count = db.query(SentimentLog).count()
    db.close()
    return {"total_predictions_logged": count}

@app.get("/models_list")
def get_models():
    # Endpoint utile per capire quante predizioni abbiamo loggato
    BASE_DIR = "/opt/versioned_models"
    models = []
    if os.path.exists(BASE_DIR) and os.listdir(BASE_DIR):
        models = [d for d in os.listdir(BASE_DIR) 
                            if os.path.isdir(os.path.join(BASE_DIR, d)) and d.startswith("model_")]
    return {"models": models}

class version(BaseModel):
    version: str

@app.post("/change_model")
def change_model(version: version):
    """Cambia il modello attivo a una versione specifica"""
    try:
        load_model(model_version=version.version)
        return {"message": f"Modello cambiato con successo alla versione {version.version}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
