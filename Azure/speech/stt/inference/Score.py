from transformers import Wav2Vec2ProcessorWithLM, AutoModelForCTC
import librosa
import json
import pyctcdecode
import logging
import base64
import torch
import base64


# def load_config():
#     with open("config.yaml", "r") as yamlfile:
#         return yaml.load(yamlfile, Loader=yaml.FullLoader)
def transcribe_audio_file(encoded_audio):
    audio_bytes = encoded_audio.encode('utf-8')
    with open("temp.wav", "wb") as wav_file:
        decoded_audio = base64.decodebytes(audio_bytes)
        wav_file.write(decoded_audio)
        wav_file.close()
    y, _ = librosa.load("temp.wav", sr=16000)
    inputs = processor(
        y, sampling_rate=16000, return_tensors="pt", padding="longest"
    )
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    transcription = processor.batch_decode(logits.numpy()).text[0]
    return transcription

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global processor, model
    # Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
    processor = Wav2Vec2ProcessorWithLM.from_pretrained("Sunbird/sunbird-asr")
    model = AutoModelForCTC.from_pretrained("Sunbird/sunbird-asr")
    logging.info("Init complete")


def run(request):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    data = json.loads(request)
    with open(data['path'], 'rb') as audio:
        encoded_audio = base64.b64encode(audio.read())
        encoded_audio_string = encoded_audio.decode('utf-8')
    audio.close()
    #print(transcribe_audio_file(encoded_audio_string))
    result = transcribe_audio_file(encoded_audio_string)
    return {"result": result}

    
    