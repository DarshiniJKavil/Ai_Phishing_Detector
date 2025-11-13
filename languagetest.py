from googletrans import Translator
translator = Translator()

text = "mera naam rahul hai"

# Step 1: Try detecting language
detected = translator.detect(text)
print("Detected Language:", detected.lang)

# Step 2: Translate to English
translated = translator.translate(text, dest='en')
print("English Translation:", translated.text)
