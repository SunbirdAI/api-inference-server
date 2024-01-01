# Sunbird API Inference Service
This repository contains code for a flask server that's containerized and deployed to [Vertex AI on GCP.](https://cloud.google.com/vertex-ai) and Azure Machine Learning Studio (https://studio.azureml.net/)

The flask server provides access to the following Sunbird AI models:
- [ASR (speech to text) for Luganda](https://huggingface.co/Sunbird/sunbird-asr).
- Translation ([local languages to English](https://huggingface.co/Sunbird/sunbird-mul-en-mbart-merged) and [English to local languages](https://huggingface.co/Sunbird/sunbird-en-mul).
- [TTS](https://huggingface.co/Sunbird/sunbird-lug-tts) (Available for azure only)

The process of deployment is as follows:
- The models are pulled from HuggingFace. See [asr_inference](app/asr_inference.py) and [translate_inference](app/translate_inference.py).
- The flask app exposes 2 endpoints: `isalive` and `predict` as required by Vertex AI. The `predict` endpoint receives a list of inference requests, passes them to the model and returns the results.
- A docker container is built from this flask app and is pushed to the Google container repository (GCR).
- On Vertex AI, a "model" is created from this container and then deployed to a Vertex endpoint, this is the same for azure

**NOTE**: Check out [this article for a detailed tutorial](https://medium.com/nlplanet/deploy-a-pytorch-model-with-flask-on-gcp-vertex-ai-8e81f25e605f) on this process for GCP.

Check out [https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints?view=azureml-api-2&tabs=azure-cli] on how to deploy models on online endpoints to azure

The resulting endpoint is then used in the main [Sunbird AI API](https://github.com/SunbirdAI/sunbird-ai-api).

### TODOs
- Add TTS (This is available for azure)
- Handle long audio files (This is available for azure).
- Use a smaller base container, current container (`huggingface/transformers-pytorch-gpu`) is pretty heavy and maybe unncessary. This would enable us to end up with a smaller artificat which takes up less memory.
- Automate the deployment process for both the API and this inference service (using Github Actions or Terraform...or both?)
- Come up with an end-to-end workflow from data ingestion to deployment (what tools are required for this?).
