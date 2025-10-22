import google.generativeai as genai

# Pega tu clave aquí
GOOGLE_API_KEY = "AIzaSyDfye1yZyq2nfJu50L8Qt_548yzUuqCaps"
genai.configure(api_key=GOOGLE_API_KEY)

for m in genai.list_models():
    print(m.name)