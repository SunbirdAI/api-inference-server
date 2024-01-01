from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MBart50Tokenizer, MBartForConditionalGeneration
import datetime
import sentencepiece
import torch
import json
import os
import yaml
from azure.storage.blob import ContainerClient
import logging

language_to_code = {
    "Acholi": ">>ach<<",
    "Ateso": ">>teo<<",
    "Luganda": ">>lug<<",
    "Lugbara": ">>lgg<<",
    "Runyankole": ">>nyn<<"
    }


language_iso_codes = {
        "Acholi": "ach",
        "Ateso": "teo",
        "Luganda": "lug",
        "Lugbara": "lgg",
        "Runyankole": "nyn",
    }



def multiple_to_english(text, language):
    mul_en_tokenizer.src_lang = language_iso_codes[language]
    inputs = mul_en_tokenizer(text, return_tensors="pt")
    tokens = mul_en_model.generate(**inputs)
    result = mul_en_tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)
    return result


def english_to_multiple(text, target_language):
    lang_code = language_to_code[target_language]
    inputs = en_mul_tokenizer(f"{lang_code}{text}", return_tensors="pt")
    tokens = en_mul_model.generate(**inputs)
    result = en_mul_tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)
    return result

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global mul_en_tokenizer, mul_en_model, en_mul_tokenizer, en_mul_model
    mul_en_tokenizer = MBart50Tokenizer.from_pretrained('Sunbird/mbart-mul-en')
    mul_en_model = MBartForConditionalGeneration.from_pretrained('Sunbird/mbart-mul-en')

    en_mul_tokenizer = AutoTokenizer.from_pretrained('Sunbird/sunbird-en-mul')
    en_mul_model = AutoModelForSeq2SeqLM.from_pretrained('Sunbird/sunbird-en-mul')

    logging.info("Init complete")


def run(request):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    
    """
    logging.info("model 1: request received")
    data = json.loads(request)

    SALT_LANGUAGE_CODES = ["ach", "lgg", "lug", "eng", "nyn", "teo", "luo"]
    offset = mul_en_tokenizer.sp_model_size + mul_en_tokenizer.fairseq_offset

    for i, code in enumerate(SALT_LANGUAGE_CODES):
        mul_en_tokenizer.lang_code_to_id[code] = i + offset
        mul_en_tokenizer.fairseq_ids_to_tokens[i + offset] = code
    if data['mul_to_eng'] == "true":
        translated = multiple_to_english(data['text'], data['language'])
        return translated
    else:
        translated = english_to_multiple(data['text'], data['language'])
        return translated   
    return translated

    
    