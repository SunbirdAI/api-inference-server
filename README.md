# Sunbird API Inference Service
This repository contains code for a flask server that's containerized and deployed to [Vertex AI on GCP.](https://cloud.google.com/vertex-ai)

The flask server provides access to the following Sunbird AI models:
- [ASR (speech to text) for Luganda](https://huggingface.co/Sunbird/sunbird-asr).
- Translation ([local languages to English](https://huggingface.co/Sunbird/sunbird-mul-en-mbart-merged) and [English to local languages](https://huggingface.co/Sunbird/sunbird-en-mul).
- [TTS](https://huggingface.co/Sunbird/sunbird-lug-tts) (coming soon to the API)

The process of deployment is as follows:
- The models are pulled from HuggingFace. See [asr_inference](app/asr_inference.py) and [translate_inference](app/translate_inference.py).
- The flask app exposes 2 endpoints: `isalive` and `predict` as required by Vertex AI. The `predict` endpoint receives a list of inference requests, passes them to the model and returns the results.
- A docker container is built from this flask app and is pushed to the Google container repository (GCR).
- On Vertex AI, a "model" is created from this container and then deployed to a Vertex endpoint.

**NOTE**: Check out [this article for a detailed tutorial](https://medium.com/nlplanet/deploy-a-pytorch-model-with-flask-on-gcp-vertex-ai-8e81f25e605f) on this process.

The resulting endpoint is then used in the main [Sunbird AI API](https://github.com/SunbirdAI/sunbird-ai-api).
