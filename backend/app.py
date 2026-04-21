import os, json, requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv


load_dotenv()   # charge les variables du fichier .env


app   = Flask(__name__)
CORS(app)       # autorise les requêtes cross-origin depuis le frontend


HF_TOKEN = os.getenv('HF_TOKEN')
HF_URL   = 'https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2'




def build_prompt(text: str) -> str:
    return f'''<s>[INST]
Tu es un assistant pédagogique. Analyse le texte suivant et réponds UNIQUEMENT
avec un objet JSON valide.
Format : {{"summary":["...","...","..."],"quiz":[{{"question":"...","options":["A...","B...","C...","D..."],"correct":0}},...]}}
3 idées clés, 3 questions QCM, répondre en FRANÇAIS.
TEXTE: {text[:3000]}[/INST]'''




@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')


    if not text.strip():
        return jsonify({'error': 'Texte vide'}), 400


    payload = {
        'inputs': build_prompt(text),
        'parameters': {'max_new_tokens': 800, 'temperature': 0.5,
                        'return_full_text': False}
    }
    headers = {'Authorization': f'Bearer {HF_TOKEN}'}


    try:
        r = requests.post(HF_URL, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        raw = r.json()[0]['generated_text']


        # Extraire le JSON même si le modèle ajoute du texte autour
        start = raw.index('{')
        end   = raw.rindex('}') + 1
        result = json.loads(raw[start:end])
        return jsonify(result)


    except Exception as e:
        return jsonify({'error': str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True, port=5000)
