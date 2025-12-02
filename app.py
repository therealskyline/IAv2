"""
🚀 HessGPT Web Interface - Flask Server
✅ Compatible avec ton modèle Epoch 6 (meilleur)
✅ API REST + Interface HTML
✅ Prêt pour production
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import sys
import os

# Importer ton modèle
sys.path.append('./Core/Model')
from HessGpt import HessGPT

app = Flask(__name__)
CORS(app)

# ============================================
# CONFIGURATION
# ============================================

CONFIG = {
    'vocab_size': 50257,
    'embed_dim': 512,
    'num_heads': 8,
    'num_layers': 8,
    'max_seq_len': 1024,
    'dropout': 0.05,
}

# Chemins des checkpoints (ordre de préférence)
CHECKPOINT_PATHS = [
    "./checkpoints/quality/hessgpt_sft_quality_BEST.pt",
    "./checkpoints/quality/hessgpt_sft_RESUME.pt"
]

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================
# INITIALISATION MODÈLE
# ============================================

print("="*60)
print("🚀 DÉMARRAGE SERVEUR HessGPT")
print("="*60)
print(f"✅ Device: {DEVICE}")

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Chargement modèle
checkpoint = None
for path in CHECKPOINT_PATHS:
    if os.path.exists(path):
        try:
            checkpoint = torch.load(path, map_location=DEVICE)
            print(f"✓ Checkpoint chargé: {path.split('/')[-1]}")
            print(f"✓ Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"✓ Val Loss: {checkpoint.get('val_loss', checkpoint.get('best_val_loss', 'N/A')):.4f}")
            break
        except Exception as e:
            print(f"✗ Erreur chargement {path}: {e}")
            continue

if checkpoint is None:
    print("❌ ERREUR: Aucun checkpoint trouvé!")
    print("📁 Cherché dans:")
    for path in CHECKPOINT_PATHS:
        print(f"  - {path}")
    sys.exit(1)

# Initialisation modèle
model = HessGPT(**CONFIG).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("✅ Modèle chargé et prêt!")
print("="*60)

# ============================================
# FONCTION DE GÉNÉRATION
# ============================================

def generate_response(prompt, max_tokens=100, temperature=0.7, top_k=50, top_p=0.9):
    """
    Génère une réponse avec le modèle HessGPT
    
    Args:
        prompt: Texte d'entrée
        max_tokens: Nombre maximum de tokens à générer
        temperature: Contrôle la créativité (0.1-1.0)
        top_k: Nombre de meilleurs tokens à considérer
        top_p: Nucleus sampling threshold
    
    Returns:
        str: Réponse générée
    """
    model.eval()
    
    # Formater le prompt (style Alpaca)
    formatted_prompt = f"Instruction: {prompt}\nResponse:"
    
    # Tokenization
    tokens = tokenizer.encode(formatted_prompt, return_tensors='pt').to(DEVICE)
    generated = tokens[0].tolist()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Forward pass
            input_ids = torch.tensor([generated], dtype=torch.long).to(DEVICE)
            
            # Limite la longueur du contexte si nécessaire
            if input_ids.size(1) > CONFIG['max_seq_len']:
                input_ids = input_ids[:, -CONFIG['max_seq_len']:]
            
            logits, _ = model(input_ids)
            next_token_logits = logits[0, -1, :]
            
            # Température
            next_token_logits = next_token_logits / temperature
            
            # Anti-répétition (pénalise les tokens récents)
            for token in set(generated[-50:]):
                next_token_logits[token] /= 1.2
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sampling
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Stop si EOS
            if next_token == tokenizer.eos_token_id:
                break
            
            generated.append(next_token)
    
    # Décoder et extraire la réponse
    full_text = tokenizer.decode(generated, skip_special_tokens=True)
    
    # Extraire uniquement la partie "Response:"
    if "Response:" in full_text:
        response = full_text.split("Response:")[-1].strip()
    else:
        response = full_text[len(formatted_prompt):].strip()
    
    return response

# ============================================
# ROUTES FLASK
# ============================================

@app.route('/')
def home():
    """Page d'accueil avec l'interface"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """
    API de génération
    
    Body JSON:
    {
        "prompt": "Votre question",
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    Returns:
    {
        "response": "Réponse générée",
        "success": true
    }
    """
    try:
        data = request.get_json()
        
        # Validation
        if not data or 'prompt' not in data:
            return jsonify({
                'error': 'Prompt manquant',
                'success': False
            }), 400
        
        prompt = data['prompt'].strip()
        if not prompt:
            return jsonify({
                'error': 'Prompt vide',
                'success': False
            }), 400
        
        # Paramètres avec valeurs par défaut
        max_tokens = min(int(data.get('max_tokens', 100)), 500)  # Max 500
        temperature = max(0.1, min(float(data.get('temperature', 0.7)), 1.0))  # Entre 0.1 et 1.0
        
        # Génération
        response = generate_response(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return jsonify({
            'response': response,
            'success': True,
            'params': {
                'max_tokens': max_tokens,
                'temperature': temperature
            }
        })
    
    except Exception as e:
        print(f"❌ Erreur génération: {e}")
        return jsonify({
            'error': f'Erreur serveur: {str(e)}',
            'success': False
        }), 500

@app.route('/clear', methods=['POST'])
def clear():
    """Efface l'historique (pour compatibilité frontend)"""
    return jsonify({'success': True, 'message': 'Conversation effacée'})

@app.route('/health', methods=['GET'])
def health():
    """Health check pour monitoring"""
    return jsonify({
        'status': 'healthy',
        'device': DEVICE,
        'model': 'HessGPT',
        'epoch': checkpoint.get('epoch', 'N/A'),
        'val_loss': checkpoint.get('val_loss', checkpoint.get('best_val_loss', 'N/A'))
    })

@app.route('/info', methods=['GET'])
def info():
    """Informations sur le modèle"""
    return jsonify({
        'model': 'HessGPT',
        'version': '1.0',
        'epoch': checkpoint.get('epoch', 'N/A'),
        'val_loss': checkpoint.get('val_loss', checkpoint.get('best_val_loss', 'N/A')),
        'config': {
            'vocab_size': CONFIG['vocab_size'],
            'embed_dim': CONFIG['embed_dim'],
            'num_layers': CONFIG['num_layers'],
            'num_heads': CONFIG['num_heads'],
            'max_seq_len': CONFIG['max_seq_len']
        },
        'device': DEVICE,
        'samples_seen': checkpoint.get('total_samples_seen', 'N/A')
    })

# ============================================
# DÉMARRAGE SERVEUR
# ============================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 Serveur démarré!")
    print("="*60)
    print("📍 Interface: http://localhost:5000")
    print("📍 API: http://localhost:5000/generate")
    print("📍 Health: http://localhost:5000/health")
    print("📍 Info: http://localhost:5000/info")
    print("="*60)
    print("\n⚠️  Utiliser CTRL+C pour arrêter\n")
    
    # Démarrage (mode debug pour développement)
    app.run(
        host='0.0.0.0',  # Accessible depuis réseau local
        port=5000,
        debug=False,     # Mettre True en dev, False en prod
        threaded=True
    )