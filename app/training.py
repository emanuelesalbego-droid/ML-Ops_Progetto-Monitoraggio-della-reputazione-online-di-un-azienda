import numpy as np
import evaluate
import pandas as pd
import datetime
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import os
os.environ['KAGGLE_CONFIG_DIR'] = "app/keys"

import kaggle

def retraining(MODEL_NAME,tokenizer):
    dataset_id = "mdismielhossenabir/sentiment-analysis"

    # Scarica il dataset utilizzando l'API di Kaggle
    kaggle.api.dataset_download_files(dataset_id, path="/opt/data", unzip=True)
    dataset = pd.read_csv('opt/data/sentiment_analysis.csv')
    dataset = dataset.rename(columns={'sentiment': 'label', 'text': 'text'})
    label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
    dataset['label'] = dataset['label'].map(label_mapping)
    dataset = Dataset.from_pandas(dataset[['text', 'label']])
    dataset = DatasetDict({
        'train': dataset,
        'validation': dataset.train_test_split(test_size=0.2)['test'],
    })

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels, average='macro')

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    #MODEL_NAME = 'cardiffnlp/twitter-roberta-base-sep2022'  # change to desired model from the hub
    #tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # augment train set with test set, for downstream apps only - DO NOT EVALUATE ON TEST
    # tokenized_datasets['train+test'] = concatenate_datasets([tokenized_datasets['train'],
    #                                                          tokenized_datasets['test']])

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    training_args = TrainingArguments(
        do_eval=False,
        #evaluation_strategy='epoch',
        output_dir='test_trainer',
        #logging_dir='test_trainer',
        #logging_strategy='epoch',
        #save_strategy='epoch',
        num_train_epochs=1,
        learning_rate=1e-05,
        per_device_train_batch_size=4,
        per_gpu_eval_batch_size=4,
        #load_best_model_at_end=True,
        metric_for_best_model='recall',
    )

    metric = evaluate.load('recall')  # default metric for sentiment dataset is recall (macro)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        #eval_dataset=tokenized_datasets['validation'],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Crea un nome cartella basato sulla data
    version_folder = f"/opt/versioned_models/model_{datetime.datetime.now().strftime('%Y%m%d')}"

    # SALVATAGGIO CORRETTO (Crea una cartella con i file necessari)
    trainer.save_model(version_folder)
    tokenizer.save_pretrained(version_folder)

    res = trainer.evaluate(tokenized_datasets['validation'])
    print(res)
    return res
