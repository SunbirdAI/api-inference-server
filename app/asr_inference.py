from transformers import Wav2Vec2ProcessorWithLM, AutoModelForCTC

import librosa
import torch
import base64

PATH = "models/sunbird-asr"
processor = Wav2Vec2ProcessorWithLM.from_pretrained(PATH, local_files_only=True)
model = AutoModelForCTC.from_pretrained(PATH, local_files_only=True)


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
