import torchaudio
import torch
import datetime
import os
import yaml
from azure.storage.blob import ContainerClient
import logging
from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN

def load_config():
    with open("config.yaml", "r") as yamlfile:
        return yaml.load(yamlfile, Loader=yaml.FullLoader)


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global tacotron2, hifi_gan
    # Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
    tacotron2 = Tacotron2.from_hparams(source="Sunbird/sunbird-lug-tts", savedir="tmpdir_tts", run_opts={"device":"cpu"})
    hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")
    logging.info("Init complete")


def run(items):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    audio_clips = []

    items = [
       "n'amatu go ganaawuliranga ekigambo",
       "ekikuvaako ennyuma nga kyogera nti",
       "Lino lye kkubo, mulitambuliremu;",
       "munaakyamiranga ku mukono ogwa",
       "ddyo, era bwe munaakyamiranga ku gwa kkono.",
       "nga tutunuulira Yesu yekka omukulu",
       "w'okukkiriza kwaffe era omutuukiriza waakwo",
       "olw'essanyu eryateekebwa mu maaso",
       "ge eyagumiikiriza omusalaba, ng'anyooma ensonyi,",
       "n'atuula ku mukono ogwa ddyo ogw'entebe ya Katonda"
     ]

    for text in items:

        mel_output, mel_length, alignment = tacotron2.encode_text(text)

        waveforms = hifi_gan.decode_batch(mel_output)

        audio_path = f"{text}.wav"
        torchaudio.save(audio_path, waveforms.squeeze(1), 22050)

        audio_clips.append(torchaudio.load(audio_path)[0])

        os.remove(audio_path)

    final_audio = torch.cat(audio_clips, dim=1)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"audio-{timestamp}.wav"

    torchaudio.save(filename, final_audio, 22050)

    config = load_config()

    container_client = ContainerClient.from_connection_string(config["azure_storage_connection_string"], config["audio_cotainer"])

    blob_client = container_client.get_blob_client(filename)

    with open(filename, "rb") as data:
        blob_client.upload_blob(data)
        print("Data Uploaded  To Blob")
        os.remove(filename)

    blob_url = blob_client.url

    return {"blob_url": blob_url}

    
    