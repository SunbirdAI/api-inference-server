import torchaudio
from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN
import io
import base64


TACOTRON2_SAMPLE_RATE = 22050

tacotron2 = Tacotron2.from_hparams(source="Sunbird/sunbird-lug-tts")
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech")

torchaudio.set_audio_backend('soundfile')


def tts(text):
    mel_output, mel_length, alignment = tacotron2.encode_text(text)
    waveforms = hifi_gan.decode_batch(mel_output)
    with io.BytesIO() as buffer:
        torchaudio.save(buffer, waveforms.squeeze(1), TACOTRON2_SAMPLE_RATE, format='wav')
        buffer.seek(0)
        encoded_audio = base64.b64encode(buffer.read())
        base64_string = encoded_audio.decode('utf-8')

    return base64_string
