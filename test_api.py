import requests
import base64
import json

url = "http://localhost:8080"
r = requests.get(f"{url}/isalive")
print(r)

with open('SEMA1-2022-11-04T120932-3.wav', 'rb') as audio:
    encoded_audio = base64.b64encode(audio.read())
    encoded_audio_string = encoded_audio.decode('utf-8')

payload = {
    "instances": [
        {
            "audio": encoded_audio_string,
            "task": "asr"
        },
        {
            "sentence": "Mbagaliza Christmass Enungi Nomwaka Omugya Gubaberere Gwamirembe",
            "task": "translate_to_english",
            "source_language": "Luganda"
        },
        {
            "sentence": "Oburwaire bwa Korona ku bwatandikire, abantu baatandika kukora masiki z'emyenda.",
            "task": "translate_to_english",
            "source_language": "Runyankole"
        },
        {
            "sentence": "Sa'wa azo corona niri si, O'bi eyi e'do afa su omvua azini tia 'diyi ede bongo eyi si.",
            "task": "translate_to_english",
            "source_language": "Lugbara"
        },
        {
            "sentence": "Kapak na abwangunor ekorona opotu itunga ogeutu aswam aitabalan nurapis aituk keda ekume keda igoen.",
            "task": "translate_to_english",
            "source_language": "Ateso"
        },
        {
            "sentence": "I kare ma two Corona opoto, dano ocako yubu bongi me boyo um",
            "task": "translate_to_english",
            "source_language": "Acholi"
        },
        {
            "sentence": "This translation service is really cool",
            "task": "translate_from_english",
            "target_language": "Luganda"
        },
        {
            "sentence": "I would like to translate this text to Acholi",
            "task": "translate_from_english",
            "target_language": "Acholi"
        }
    ]
}

headers = {
    "Content-Type": "application/json"
}

r = requests.post(f"{url}/predict", headers=headers, data=json.dumps(payload))
print(json.dumps(r.json(), indent=3))
