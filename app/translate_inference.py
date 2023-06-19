from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MBart50Tokenizer, MBartForConditionalGeneration

mul_en_tokenizer = MBart50Tokenizer.from_pretrained('Sunbird/mbart-mul-en')
mul_en_model = MBartForConditionalGeneration.from_pretrained('Sunbird/mbart-mul-en')

en_mul_tokenizer = AutoTokenizer.from_pretrained('Sunbird/sunbird-en-mul')
en_mul_model = AutoModelForSeq2SeqLM.from_pretrained('Sunbird/sunbird-en-mul')

language_to_code = {
    "Acholi": ">>ach<<",
    "Ateso": ">>teo<<",
    "Luganda": ">>lug<<",
    "Lugbara": ">>lgg<<",
    "Runyankole": ">>nyn<<"
}

SALT_LANGUAGE_CODES = ["ach", "lgg", "lug", "eng", "nyn", "teo", "luo"]
offset = mul_en_tokenizer.sp_model_size + mul_en_tokenizer.fairseq_offset

language_iso_codes = {
    "Acholi": "ach",
    "Ateso": "teo",
    "Luganda": "lug",
    "Lugbara": "lgg",
    "Runyankole": "nyn",
}

for i, code in enumerate(SALT_LANGUAGE_CODES):
    mul_en_tokenizer.lang_code_to_id[code] = i + offset
    mul_en_tokenizer.fairseq_ids_to_tokens[i + offset] = code


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
