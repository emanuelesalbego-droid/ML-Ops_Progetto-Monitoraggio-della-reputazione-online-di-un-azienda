# Fase 1: Base image
FROM python:3.12-slim

USER root

# installa dipendenze
ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
