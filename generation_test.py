"""
🧪 BENCHMARK HessGPT - TEST COMPLET SFT
✅ Questions alignées avec le training
✅ Catégories : Alpaca, Dolly, WizardLM, Conversations
✅ Scoring automatique
"""

import torch
import torch.nn.functional as F
import sys, time
from transformers import GPT2Tokenizer
from tqdm import tqdm

sys.path.append('./Core/Model')
from HessGpt import HessGPT

print("="*70)
print("🧪 BENCHMARK HessGPT - TEST COMPLET")
print("="*70)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n✅ Device: {device}")

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# CONFIG
CONFIG = {
    'vocab_size': 50257,
    'embed_dim': 512,
    'num_heads': 8,
    'num_layers': 8,
    'max_seq_len': 1024,
    'dropout': 0.05,
}

# Chargement modèle SFT
print("\n🤖 Chargement modèle SFT...")

# Essayer d'abord le BEST, sinon RESUME
checkpoint_paths = [
    "./checkpoints/quality/hessgpt_sft_quality_BEST.pt",
    "./checkpoints/quality/hessgpt_sft_RESUME.pt"
]

checkpoint = None
for path in checkpoint_paths:
    try:
        checkpoint = torch.load(path, map_location=device)
        print(f"✓ Checkpoint chargé: {path.split('/')[-1]}")
        break
    except FileNotFoundError:
        continue

if checkpoint is None:
    print("❌ Aucun checkpoint trouvé!")
    print("📁 Cherché dans:")
    for path in checkpoint_paths:
        print(f"  - {path}")
    sys.exit(1)

model = HessGPT(**CONFIG).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Afficher les infos
val_loss = checkpoint.get('val_loss', checkpoint.get('best_val_loss', 'N/A'))
print(f"✓ Val Loss: {val_loss}")
print(f"✓ Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"✓ Samples vus: {checkpoint.get('total_samples_seen', 'N/A'):,}" if checkpoint.get('total_samples_seen') else "")

# FONCTION DE GÉNÉRATION
def generate_text(model, prompt, max_length=150, temperature=0.7, top_k=50, top_p=0.9):
    model.eval()
    tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = tokens[0].tolist()
    
    with torch.no_grad():
        for _ in range(max_length):
            input_ids = torch.tensor([generated], dtype=torch.long).to(device)
            logits, _ = model(input_ids)
            next_token_logits = logits[0, -1, :]
            
            next_token_logits = next_token_logits / temperature
            
            # Anti-répétition
            for token in set(generated[-50:]):
                next_token_logits[token] /= 1.2
            
            # Top-k
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token == tokenizer.eos_token_id:
                break
            
            generated.append(next_token)
    
    full_text = tokenizer.decode(generated, skip_special_tokens=True)
    if "Response:" in full_text:
        return full_text.split("Response:")[-1].strip()
    return full_text[len(tokenizer.decode(tokens[0], skip_special_tokens=True)):].strip()

# BENCHMARK TESTS
benchmark_tests = {
    "💬 CONVERSATIONS (Synthétique 10K)": [
        {"q": "Hi!", "expected_type": "greeting"},
        {"q": "Hello!", "expected_type": "greeting"},
        {"q": "Hey there!", "expected_type": "greeting"},
        {"q": "Good morning!", "expected_type": "greeting"},
        {"q": "How are you?", "expected_type": "how_are_you"},
        {"q": "How's it going?", "expected_type": "how_are_you"},
        {"q": "Thank you!", "expected_type": "thanks_response"},
        {"q": "Thanks a lot!", "expected_type": "thanks_response"},
        {"q": "I appreciate it.", "expected_type": "thanks_response"},
        {"q": "Goodbye!", "expected_type": "goodbye"},
        {"q": "Bye!", "expected_type": "goodbye"},
        {"q": "See you later!", "expected_type": "goodbye"},
    ],
    
    "🎯 QUESTIONS FACTUELLES (Alpaca/Stanford)": [
        {"q": "What is the capital of France?", "keywords": ["paris"]},
        {"q": "What is the capital of Japan?", "keywords": ["tokyo"]},
        {"q": "What is the largest planet in our solar system?", "keywords": ["jupiter"]},
        {"q": "Who wrote Romeo and Juliet?", "keywords": ["shakespeare", "william"]},
        {"q": "What is the chemical symbol for water?", "keywords": ["h2o", "h₂o"]},
        {"q": "What is the speed of light?", "keywords": ["300", "million", "meters", "per second"]},
        {"q": "What year did World War II end?", "keywords": ["1945"]},
        {"q": "What is the smallest prime number?", "keywords": ["2", "two"]},
    ],
    
    "📚 INSTRUCTIONS SIMPLES (Alpaca)": [
        {"q": "Give me 3 tips for better sleep.", "check": "list_or_tips"},
        {"q": "List 5 benefits of exercise.", "check": "list_or_tips"},
        {"q": "Name 3 colors.", "check": "list_simple"},
        {"q": "What are the days of the week?", "check": "list_simple"},
        {"q": "Give me a fun fact.", "check": "provides_info"},
    ],
    
    "🧠 EXPLICATIONS (Dolly/WizardLM)": [
        {"q": "Explain what photosynthesis is.", "min_words": 15},
        {"q": "How does gravity work?", "min_words": 15},
        {"q": "What is democracy?", "min_words": 10},
        {"q": "Explain the concept of evolution.", "min_words": 15},
        {"q": "What is artificial intelligence?", "min_words": 15},
    ],
    
    "✍️ TÂCHES CRÉATIVES (WizardLM)": [
        {"q": "Write a haiku about nature.", "check": "creative"},
        {"q": "Describe a sunset in one sentence.", "check": "creative"},
        {"q": "Give me a metaphor for time.", "check": "creative"},
    ],
    
    "📝 INSTRUCTIONS AVEC INPUT (Alpaca)": [
        {
            "q": "Summarize the following text.\nInput: Python is a high-level programming language known for its simplicity and readability.\nResponse:",
            "keywords": ["python", "programming", "simple"]
        },
        {
            "q": "Translate to French.\nInput: Hello, how are you?\nResponse:",
            "keywords": ["bonjour", "comment", "allez"]
        },
    ],
}

# FONCTION DE SCORING
def score_response(response, test_case):
    """Score la qualité de la réponse (0-100)"""
    score = 0
    response_lower = response.lower()
    words = response.split()
    
    # Longueur appropriée
    if 5 < len(words) < 200:
        score += 20
    
    # Pas de répétition
    if len(words) > 3:
        unique_ratio = len(set(words)) / len(words)
        score += int(unique_ratio * 20)
    
    # Pas de tokens bizarres
    if not any(c in response for c in ['�', '\\x', '<|', 'Instruction:']):
        score += 20
    
    # Tests spécifiques
    if 'keywords' in test_case:
        keywords_found = sum(1 for kw in test_case['keywords'] if kw.lower() in response_lower)
        score += int((keywords_found / len(test_case['keywords'])) * 40)
    
    if 'expected_type' in test_case:
        expected = test_case['expected_type']
        if expected == 'greeting':
            greetings = ['hello', 'hi', 'hey', 'good', 'how can', 'how may', 'assist', 'help']
            if any(g in response_lower for g in greetings):
                score += 40
        elif expected == 'how_are_you':
            responses = ['good', 'great', 'well', 'fine', 'doing', 'how about you']
            if any(r in response_lower for r in responses):
                score += 40
        elif expected == 'thanks_response':
            responses = ['welcome', 'pleasure', 'anytime', 'problem', 'glad']
            if any(r in response_lower for r in responses):
                score += 40
        elif expected == 'goodbye':
            responses = ['bye', 'goodbye', 'see you', 'take care', 'later']
            if any(r in response_lower for r in responses):
                score += 40
    
    if 'min_words' in test_case:
        if len(words) >= test_case['min_words']:
            score += 40
    
    if 'check' in test_case:
        check = test_case['check']
        if check == 'list_or_tips':
            # Cherche des numéros ou tirets
            has_list = any(c in response for c in ['1.', '2.', '3.', '-', '•'])
            if has_list or len(words) > 20:
                score += 40
        elif check == 'list_simple':
            if len(words) > 5:
                score += 40
        elif check == 'provides_info':
            if len(words) > 10:
                score += 40
        elif check == 'creative':
            if len(words) > 5 and len(words) < 100:
                score += 40
    
    return min(score, 100)

# EXÉCUTION DU BENCHMARK
print("\n" + "="*70)
print("🧪 DÉBUT DU BENCHMARK")
print("="*70)

all_results = {}
total_score = 0
total_tests = 0

for category, tests in benchmark_tests.items():
    print(f"\n{'='*70}")
    print(category)
    print('='*70)
    
    category_scores = []
    
    for i, test in enumerate(tests, 1):
        question = test['q']
        
        # Formater la question
        if "Input:" in question:
            prompt = question  # Déjà formaté
        else:
            prompt = f"Instruction: {question}\nResponse:"
        
        # Générer
        print(f"\n[{i}/{len(tests)}] 📝 Question:")
        print(f"    {question[:80]}{'...' if len(question) > 80 else ''}")
        
        start_time = time.time()
        response = generate_text(model, prompt, max_length=100)
        gen_time = time.time() - start_time
        
        # Scorer
        score = score_response(response, test)
        category_scores.append(score)
        total_score += score
        total_tests += 1
        
        # Afficher
        print(f"\n    🤖 Réponse ({gen_time:.2f}s):")
        print(f"    {response[:150]}{'...' if len(response) > 150 else ''}")
        print(f"\n    📊 Score: {score}/100 {'✅' if score >= 70 else '⚠️' if score >= 50 else '❌'}")
    
    # Stats catégorie
    avg_score = sum(category_scores) / len(category_scores)
    all_results[category] = {
        'scores': category_scores,
        'avg': avg_score,
        'count': len(tests)
    }
    
    print(f"\n{'─'*70}")
    print(f"📈 Moyenne catégorie: {avg_score:.1f}/100")

# RÉSUMÉ FINAL
print("\n" + "="*70)
print("📊 RÉSUMÉ FINAL DU BENCHMARK")
print("="*70)

print(f"\n🎯 Résultats par catégorie:\n")
for category, results in all_results.items():
    avg = results['avg']
    emoji = "🔥" if avg >= 80 else "✅" if avg >= 70 else "⚠️" if avg >= 60 else "❌"
    print(f"  {emoji} {category}")
    print(f"     Score moyen: {avg:.1f}/100 ({results['count']} tests)")

# Score global
global_avg = total_score / total_tests
print(f"\n{'='*70}")
print(f"🏆 SCORE GLOBAL: {global_avg:.1f}/100")
print(f"{'='*70}")

# Interprétation
if global_avg >= 80:
    print("\n🔥 EXCELLENT! Le modèle performe très bien sur tous les types de tâches.")
elif global_avg >= 70:
    print("\n✅ TRÈS BON! Le modèle suit bien les instructions avec quelques imperfections.")
elif global_avg >= 60:
    print("\n⚠️  BON! Le modèle comprend les instructions mais manque de précision.")
else:
    print("\n❌ MOYEN. Le modèle a besoin de plus d'entraînement.")

print(f"\n📊 Détails:")
print(f"  • Tests totaux: {total_tests}")
print(f"  • Val Loss: {checkpoint.get('val_loss', checkpoint.get('best_val_loss', 'N/A'))}")
print(f"  • Epoch: {checkpoint.get('epoch', 'N/A')}/4")
print(f"  • Samples vus: {checkpoint.get('total_samples_seen', 'N/A')}")

print("\n" + "="*70)
print("✅ BENCHMARK TERMINÉ")
print("="*70)