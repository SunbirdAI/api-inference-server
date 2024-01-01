from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from asr_inference import transcribe_audio_file
from translate_inference import multiple_to_english, english_to_multiple
from tts_inference import tts
import json

app = Flask(__name__)
CORS(app)

@app.route("/isalive")
def is_alive():
    print("/isalive request")
    status_code = Response(status=200)
    return status_code


@app.route("/predict", methods=["POST"])
def predict():
    print("/predict request")
    # print(request)
    # audio = request.files['audio']
    # print(f"{audio.filename}")
    req_json = request.get_json()
    # print(type(req_json))
    if type(req_json) == str:
        req_json = json.loads(req_json)
    # print(req_json.keys())
    # print(req_json)
    json_instances = req_json["instances"]
    from_english_sentences = []
    to_english_sentences = []
    tts_text = []
    # audio_files = [j['audio'] for j in json_instances]
    audio_files = []
    for j in json_instances:
        if 'task' in j.keys():
            if j['task'] == 'translate_from_english':
                from_english_sentences.append((j['sentence'], j['target_language']))
            elif j['task'] == 'translate_to_english':
                to_english_sentences.append((j['sentence'], j['source_language']))
            elif j['task'] == 'asr':
                audio_files.append(j['audio'])
            elif j['task'] == 'tts':
                tts_text.append(j['sentence'])

    transcripts = [transcribe_audio_file(encoded_audio) for encoded_audio in audio_files]
    to_english_translations = [multiple_to_english(text, source_language) for text, source_language in to_english_sentences]
    from_english_translations = [english_to_multiple(text, target_language) for
                                 text, target_language in from_english_sentences]
    tts_audio = [tts(text) for text in tts_text]
    # transcript = transcripts[0]
    return jsonify({
        "transcripts": transcripts,
        "to_english_translations": to_english_translations,
        "from_english_translations": from_english_translations,
        "base64_audio": tts_audio
    })
    # return jsonify({"message": "success"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
