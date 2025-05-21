# ElevenLabs Gradio Toolkit

Applicazione con interfaccia grafica basata su **Gradio** che consente di interagire con le API di ElevenLabs.

## Funzionalità principali

1. **TTS Avanzato**
   - Box di testo con supporto a SSML.
   - Selezione del modello vocale e della voce presente nell'account.
   - Slider per `similarity_boost`, `stability`, `style_exaggeration` e `speed`.
   - Generazione e riproduzione dell'audio risultante.

2. **Voice Changer Batch**
   - Caricamento di più file audio da convertire in una voce scelta.
   - Salvataggio automatico dei risultati in una cartella locale.

3. **Voice Cloning**
   - Caricamento di uno o più file audio per creare un clone istantaneo di voce tramite l'API `/v1/voices/add`.

## Requisiti

- Python 3.9+
- Le librerie elencate in `requirements.txt`.
- Una chiave API valida di ElevenLabs.

## Avvio

Installare le dipendenze e lanciare lo script:

```bash
pip install -r requirements.txt
python gradio_app.py
```

Si aprirà l'interfaccia web di Gradio nel browser. Inserire la propria API key e utilizzare le varie schede per generare audio, convertire file o clonare una voce.
