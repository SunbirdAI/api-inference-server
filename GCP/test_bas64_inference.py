# from transformers import Wav2Vec2ProcessorWithLM, AutoModelForCTC

from app.asr_inference import transcribe_audio_file
import base64

# PATH = "app/models/sunbird-asr"
# processor = Wav2Vec2ProcessorWithLM.from_pretrained(PATH, local_files_only=True)
# model = AutoModelForCTC.from_pretrained(PATH, local_files_only=True)


# def transcribe_audio_file(encoded_audio):
#     audio_bytes = encoded_audio.encode('utf-8')
#     with open("temp.wav", "wb") as wav_file:
#         decoded_audio = base64.decodebytes(audio_bytes)
#         wav_file.write(decoded_audio)
#         wav_file.close()
#     y, _ = librosa.load("temp.wav", sr=16000)
#     inputs = processor(
#         y, sampling_rate=16000, return_tensors="pt", padding="longest"
#     )
#     with torch.no_grad():
#         logits = model(inputs.input_values).logits
#
#     transcription = processor.batch_decode(logits.numpy()).text[0]
#     return transcription


with open('SEMA1-2022-11-04T120932-3.wav', 'rb') as audio:
    encoded_audio = base64.b64encode(audio.read())
    encoded_audio_string = encoded_audio.decode('utf-8')

audio.close()
print(transcribe_audio_file(encoded_audio_string))
