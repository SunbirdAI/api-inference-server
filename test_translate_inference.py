from app.translate_inference import multiple_to_english, english_to_multiple

text = "Mbagaliza Christmass Enungi Nomwaka Omugya Gubaberere Gwamirembe"

print(multiple_to_english(text, 'Luganda'))

english_text = "Welcome to the meeting."

for target_language in ["Luganda", "Runyankole", "Lugbara", "Acholi", "Ateso"]:
    print(english_to_multiple(english_text, target_language))
