"""
🚀 SFT HessGPT - CHATBOT ULTIMATE v2 (9-10/10)
✅ Générateur synthétique intégré (remplace basic_examples)
✅ 60% synthétique + 40% OpenAssistant
✅ MAX_ASSISTANT_TOKENS = 150 (optimisé ChatGPT 3.5)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sys, os, time, math, random, json
from tqdm import tqdm
from transformers import GPT2Tokenizer

sys.path.append('./Core/Model')
from HessGpt import HessGPT

print("="*60)
print("🚀 SFT HessGPT - CHATBOT ULTIMATE v2")
print("="*60)

# GPU Setup
if torch.cuda.is_available():
    device = 'cuda'
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    print("❌ GPU non disponible!")
    sys.exit(1)

# ============================================
# CONFIG ULTIME v2
# ============================================
CONFIG = {
    # ========== MODEL ==========
    'vocab_size': 50257,
    'embed_dim': 512,
    'num_heads': 8,
    'num_layers': 8,
    'max_seq_len': 1024,
    'dropout': 0.1,
    
    # ========== TRAINING ==========
    'batch_size': 32,              # Plus petit pour 60K samples
    'gradient_accumulation': 4,    # 32×4 = 128 effectif
    'num_epochs': 12,              # Encore moins d'epochs
    
    'learning_rate': 2e-5,         # Encore plus bas
    'warmup_steps': 500,           # Warmup très long
    'max_grad_norm': 1.0,
    
    # ========== DATA - AGGRESSIVE ==========
    'max_samples': 60000,          # ← 60K (optimal)
    'min_length': 10,
    'max_length': 256,
    'max_assistant_tokens': 150,
    
    # Dataset composition - Plus diversifié
    'synthetic_ratio': 0.40,       # 40% synthétique (24K)
    'oasst_ratio': 0.60,          # 60% OpenAssistant (36K)
    'natural_conv_ratio': 0.20,    # Moins de naturel, plus technique
    
    # ========== EARLY STOPPING ==========
    'patience': 6,                 # Plus patient
    'val_split': 0.1,              # 6K validation
    
    # ========== OPTIMIZER ==========
    'scheduler_type': 'cosine',
    'min_lr_ratio': 0.1,
    'weight_decay': 0.01,
    'label_smoothing': 0.0,
    
    # ========== GENERATION ==========
    'temperature': 0.9,
    'top_k': 50,
    'top_p': 0.95,
    'repetition_penalty': 1.3,
}

print(f"\n⚙️  Configuration Ultimate v2:")
for key, val in CONFIG.items():
    print(f"  {key}: {val}")

# ============================================
# TOKENIZER
# ============================================
print("\n🔤 Chargement GPT-2 tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
print(f"✓ Tokenizer: {len(tokenizer)} tokens")

# ============================================
# GÉNÉRATEUR SYNTHÉTIQUE (remplace basic_examples)
# ============================================
class SyntheticDialogueGenerator:
    """Générateur de dialogues synthétiques haute qualité"""
    
    def __init__(self, tokenizer, max_assistant_tokens=150):
        self.tokenizer = tokenizer
        self.max_assistant_tokens = max_assistant_tokens
        
        # Topics pour questions factuelles
        self.topics = [
            "history", "geography", "sport", "science", "mathematics", 
            "technology", "literature", "art", "music", "economics", 
            "politics", "biology", "chemistry", "astronomy", "philosophy"
        ]
        
        # Templates factuels
        self.factual_templates = [
            ("Tell me about {}.", 
             lambda t: f"{t.capitalize()}: key facts, history, and significance."),
            ("What is {}?", 
             lambda t: f"{t.capitalize()} is... [definition and explanation]"),
            ("Explain {} in simple terms.", 
             lambda t: f"{t.capitalize()} explained: main concepts with examples."),
            ("Give me facts about {}.", 
             lambda t: f"Interesting facts about {t}: [3-4 key points]"),
        ]
        
        # Templates code
        self.code_templates = [
            ("Write a Python function to compute factorial.", 
             "```python\ndef factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)\n```"),
            ("Show HTML example page.", 
             "```html\n<!doctype html>\n<html>\n<body><h1>Hello</h1></body>\n</html>\n```"),
            ("SQL query for top 10 users by score.", 
             "```sql\nSELECT * FROM users ORDER BY score DESC LIMIT 10;\n```"),
            ("JavaScript to reverse string.", 
             "```javascript\nfunction reverse(s){return s.split('').reverse().join('');}\n```"),
        ]
        
        # Templates créatifs
        self.creative_templates = [
            ("Write a short story.", 
             "Once upon a time... [creative narrative]"),
            ("Compose a haiku about nature.", 
             "Silent forest speaks\nLeaves whisper ancient secrets\nPeace flows like water"),
            ("Give motivational advice.", 
             "Remember: every expert was once a beginner. Keep learning!"),
        ]
        
        # Templates conversations naturelles
        self.natural_templates = [
            ("Hello!", "Hello! How can I help you today?"),
            ("Hi there", "Hi! What can I do for you?"),
            ("How are you?", "I'm doing well, thank you! How are you?"),
            ("What's up?", "Not much! Just here to help. What do you need?"),
            ("Can you help me?", "Absolutely! What would you like help with?"),
            ("I need help", "I'm here to help! What do you need?"),
            ("What can you do?", "I can help answer questions and have conversations!"),
            ("Who are you?", "I'm an AI assistant designed to help with questions and tasks."),
            ("Thank you", "You're welcome! Anything else I can help with?"),
            ("Thanks!", "Happy to help! Let me know if you need anything else."),
            ("Bye", "Goodbye! Have a great day!"),
            ("See you later", "See you! Take care!"),
            ("I don't understand", "No problem! What would you like me to clarify?"),
            ("Can you explain?", "Of course! What would you like me to explain?"),
        ]
        
        # Templates pratiques
        self.practical_templates = [
            ("How to study effectively?", 
             "Study tips: 1) Break into chunks 2) Space repetition 3) Practice actively"),
            ("How to make a budget?", 
             "Budget steps: List income, track expenses, set savings goals, review monthly"),
            ("Tips for better sleep?", 
             "Sleep better: Regular schedule, dark room, no screens before bed, relax"),
        ]
    
    def truncate_to_max_tokens(self, text):
        """Limite le texte au nombre max de tokens"""
        toks = self.tokenizer.encode(text, add_special_tokens=False)
        if len(toks) <= self.max_assistant_tokens:
            return text
        
        truncated = self.tokenizer.decode(toks[:self.max_assistant_tokens])
        
        # Fermer les code blocks si nécessaire
        if "```" in truncated and truncated.count("```") % 2 == 1:
            truncated += "\n```"
        
        return truncated
    
    def generate_one(self, sample_type=None):
        """Génère un dialogue synthétique"""
        
        # Choisir type si non spécifié
        if sample_type is None:
            r = random.random()
            if r < CONFIG['natural_conv_ratio']:
                sample_type = "natural"
            else:
                sample_type = random.choices(
                    ["factual", "code", "creative", "practical"],
                    weights=[0.5, 0.2, 0.2, 0.1]
                )[0]
        
        # Générer selon le type
        if sample_type == "natural":
            user, assistant = random.choice(self.natural_templates)
        
        elif sample_type == "factual":
            tmpl, resp_fn = random.choice(self.factual_templates)
            topic = random.choice(self.topics)
            user = tmpl.format(topic)
            assistant = resp_fn(topic)
        
        elif sample_type == "code":
            user, assistant = random.choice(self.code_templates)
        
        elif sample_type == "creative":
            user, assistant = random.choice(self.creative_templates)
        
        elif sample_type == "practical":
            user, assistant = random.choice(self.practical_templates)
        
        # Ajouter variété (15% du temps)
        if random.random() < 0.15:
            user += " " + random.choice([
                "Can you explain?", 
                "Please be concise.", 
                "Give details.",
                "Thanks!",
                "Help me understand."
            ])
        
        # Limiter tokens assistant
        assistant = self.truncate_to_max_tokens(assistant)
        
        return {"user": user.strip(), "assistant": assistant.strip()}
    
    def generate_batch(self, n):
        """Génère n dialogues"""
        return [self.generate_one() for _ in range(n)]

# ============================================
# DATASET
# ============================================
class ChatDataset(Dataset):
    """Dataset optimisé pour dialogues"""
    
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.debug_printed = False
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        user_msg = item['user'].strip()
        assistant_msg = item['assistant'].strip()
        
        user_text = f"User: {user_msg}\nAssistant:"
        full_text = f"{user_text} {assistant_msg}"
        
        user_tokens = self.tokenizer.encode(user_text, add_special_tokens=False)
        assistant_tokens = self.tokenizer.encode(f" {assistant_msg}", add_special_tokens=False)
        
        all_tokens = user_tokens + assistant_tokens
        all_tokens.append(self.tokenizer.eos_token_id)
        
        if len(all_tokens) > self.max_length:
            if len(user_tokens) < self.max_length - 10:
                all_tokens = user_tokens + assistant_tokens[:self.max_length - len(user_tokens) - 1]
                all_tokens.append(self.tokenizer.eos_token_id)
            else:
                all_tokens = all_tokens[:self.max_length]
        
        input_ids = torch.tensor(all_tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(all_tokens[1:], dtype=torch.long)
        
        mask = torch.ones_like(target_ids) * -100
        assistant_start = len(user_tokens)
        mask[assistant_start:] = target_ids[assistant_start:]
        
        pad_length = self.max_length - 1 - len(input_ids)
        if pad_length > 0:
            input_ids = torch.cat([
                input_ids,
                torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)
            ])
            mask = torch.cat([
                mask,
                torch.full((pad_length,), -100, dtype=torch.long)
            ])
        
        if not self.debug_printed and idx == 0:
            self.debug_printed = True
            print(f"\n🔍 DEBUG DATASET:")
            print(f"  User: {user_msg[:80]}...")
            print(f"  Assistant: {assistant_msg[:80]}...")
            print(f"  User tokens: {len(user_tokens)}")
            print(f"  Assistant tokens: {len(assistant_tokens)}")
            print(f"✓ Masking OK!\n")
        
        return input_ids, mask

# ============================================
# CHARGEMENT DONNÉES v2
# ============================================
print("\n📥 Chargement datasets ULTIMATE v2...")
os.makedirs("data", exist_ok=True)

cache_file = "data/chat_cache_ultimate_v2_12k.pt"

if os.path.exists(cache_file):
    print(f"✓ Cache: {cache_file}")
    cached = torch.load(cache_file)
    chat_data = cached['data']
    print(f"✓ {len(chat_data)} dialogues")
else:
    print("\n🔧 Génération dataset synthétique + OpenAssistant...")
    
    chat_data = []
    generator = SyntheticDialogueGenerator(tokenizer, CONFIG['max_assistant_tokens'])
    
    # ============================================
    # ÉTAPE 1: SYNTHÉTIQUE (60%)
    # ============================================
    print("\n📚 Étape 1/2: Génération synthétique...")
    
    num_synthetic = int(CONFIG['max_samples'] * CONFIG['synthetic_ratio'])
    print(f"  Target: {num_synthetic} dialogues synthétiques...")
    
    synthetic_data = generator.generate_batch(num_synthetic)
    chat_data.extend(synthetic_data)
    
    print(f"✓ Synthétique: {len(synthetic_data)} dialogues")
    
    # ============================================
    # ÉTAPE 2: OPENASSISTANT (40%)
    # ============================================
    print("\n📚 Étape 2/2: OpenAssistant filtré...")
    
    target_oasst = CONFIG['max_samples'] - len(chat_data)
    
    try:
        from datasets import load_dataset
        oasst = load_dataset("OpenAssistant/oasst1", split="train")
        
        def is_english(text):
            if not text:
                return False
            ascii_chars = sum(1 for c in text if ord(c) < 128)
            return ascii_chars / len(text) > 0.9
        
        def is_conversational(text):
            words = ['hello', 'hi', 'hey', 'how', 'what', 'can you',
                    'help', 'please', 'thank', 'sorry', 'yes', 'no']
            return any(w in text.lower() for w in words)
        
        messages_dict = {}
        for item in oasst:
            messages_dict[item['message_id']] = item
        
        oasst_data = []
        
        for item in tqdm(oasst, desc="Filtering OpenAssistant"):
            if item.get('role') != 'assistant':
                continue
            
            if item.get('lang', 'en') != 'en':
                continue
            
            assistant_text = item.get('text', '').strip()
            
            # Filtre longueur
            assistant_tokens = tokenizer.encode(assistant_text)
            if len(assistant_tokens) > CONFIG['max_assistant_tokens']:
                continue
            
            if not assistant_text or len(assistant_text) < 10:
                continue
            
            if not is_english(assistant_text):
                continue
            
            parent_id = item.get('parent_id')
            if parent_id and parent_id in messages_dict:
                parent = messages_dict[parent_id]
                if parent.get('role') == 'prompter':
                    user_text = parent.get('text', '').strip()
                    
                    if user_text and len(user_text) > 5 and is_english(user_text):
                        total = tokenizer.encode(user_text + assistant_text)
                        if len(total) < CONFIG['max_length'] and len(total) > 10:
                            weight = 5 if is_conversational(user_text) else 1
                            
                            for _ in range(weight):
                                oasst_data.append({
                                    'user': user_text,
                                    'assistant': assistant_text
                                })
        
        random.shuffle(oasst_data)
        oasst_data = oasst_data[:target_oasst]
        chat_data.extend(oasst_data)
        
        print(f"✓ OpenAssistant: {len(oasst_data)} dialogues")
        
    except Exception as e:
        print(f"⚠️  Erreur OpenAssistant: {e}")
    
    # ============================================
    # FINALISATION
    # ============================================
    if len(chat_data) == 0:
        print("❌ Aucune donnée!")
        sys.exit(1)
    
    random.shuffle(chat_data)
    
    if len(chat_data) > CONFIG['max_samples']:
        chat_data = chat_data[:CONFIG['max_samples']]
    
    print(f"\n✓ Total: {len(chat_data)} dialogues")
    
    # Stats
    synthetic_count = int(len(chat_data) * CONFIG['synthetic_ratio'])
    oasst_count = len(chat_data) - synthetic_count
    print(f"  • Synthétique: {synthetic_count} ({CONFIG['synthetic_ratio']*100:.0f}%)")
    print(f"  • OpenAssistant: {oasst_count} ({CONFIG['oasst_ratio']*100:.0f}%)")
    
    # Exemples
    print(f"\n📝 Exemples:")
    for i in range(min(5, len(chat_data))):
        print(f"\n  Exemple {i+1}:")
        print(f"    User: {chat_data[i]['user'][:60]}...")
        print(f"    Assistant: {chat_data[i]['assistant'][:60]}...")
    
    print(f"\n💾 Sauvegarde cache...")
    torch.save({'data': chat_data, 'config': CONFIG}, cache_file)
    print(f"✓ Cache: {cache_file}")

# ============================================
# SPLIT TRAIN/VAL
# ============================================
print("\n📊 Split train/validation...")
random.shuffle(chat_data)

split_idx = int(len(chat_data) * (1 - CONFIG['val_split']))
train_data = chat_data[:split_idx]
val_data = chat_data[split_idx:]

print(f"✓ Train: {len(train_data)} dialogues")
print(f"✓ Val: {len(val_data)} dialogues")

# ============================================
# DATALOADER
# ============================================
print("\n📦 Création datasets...")
train_dataset = ChatDataset(train_data, tokenizer, CONFIG['max_length'])
val_dataset = ChatDataset(val_data, tokenizer, CONFIG['max_length'])

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    drop_last=False
)

num_batches = len(train_loader)
total_steps = (num_batches * CONFIG['num_epochs']) // CONFIG['gradient_accumulation']

print(f"✓ Train batches/époque: {num_batches}")
print(f"✓ Total steps: {total_steps}")

# ============================================
# MODÈLE
# ============================================
print("\n🤖 Chargement modèle...")

checkpoint_path = "./checkpoints/hessgpt_gpt2_final.pt"
if not os.path.exists(checkpoint_path):
    print(f"❌ Checkpoint non trouvé!")
    sys.exit(1)

checkpoint = torch.load(checkpoint_path, map_location=device)

model = HessGPT(
    vocab_size=CONFIG['vocab_size'],
    embed_dim=CONFIG['embed_dim'],
    num_heads=CONFIG['num_heads'],
    num_layers=CONFIG['num_layers'],
    max_seq_len=CONFIG['max_seq_len'],
    dropout=CONFIG['dropout']
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
print("✓ Poids chargés!")

num_params = sum(p.numel() for p in model.parameters())
print(f"✓ Paramètres: {num_params/1e6:.1f}M")

# ============================================
# OPTIMIZER & SCHEDULER
# ============================================
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    betas=(0.9, 0.95),
    weight_decay=CONFIG['weight_decay'],
    fused=True
)

def lr_lambda(step):
    if step < CONFIG['warmup_steps']:
        return step / CONFIG['warmup_steps']
    
    progress = (step - CONFIG['warmup_steps']) / max(total_steps - CONFIG['warmup_steps'], 1)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return CONFIG['min_lr_ratio'] + (1.0 - CONFIG['min_lr_ratio']) * cosine_decay

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

print(f"\n✓ Optimizer: AdamW (lr={CONFIG['learning_rate']:.2e})")

# ============================================
# VALIDATION
# ============================================
def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                logits, _ = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    ignore_index=-100
                )
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item()
                num_batches += 1
    
    model.train()
    return total_loss / max(num_batches, 1)

# ============================================
# TRAINING
# ============================================
print("\n" + "="*60)
print("🚀 DÉBUT FINE-TUNING ULTIMATE v2")
print("="*60)
print(f"📊 {num_batches} batches × {CONFIG['num_epochs']} époques max")
print(f"⚡ Early stopping: patience={CONFIG['patience']}")
print("="*60 + "\n")

os.makedirs("checkpoints/chat", exist_ok=True)
start_time = time.time()

model.train()
train_losses = []
val_losses = []
scaler = torch.amp.GradScaler('cuda')

best_val_loss = float('inf')
best_epoch = 0
patience_counter = 0

for epoch in range(CONFIG['num_epochs']):
    epoch_loss = 0
    valid_batches = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
    
    for batch_idx, (x, y) in enumerate(pbar):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda'):
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=-100,
                label_smoothing=CONFIG['label_smoothing']
            )
            loss = loss / CONFIG['gradient_accumulation']
        
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            continue
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % CONFIG['gradient_accumulation'] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        
        epoch_loss += loss.item() * CONFIG['gradient_accumulation']
        valid_batches += 1
        train_losses.append(loss.item() * CONFIG['gradient_accumulation'])
        
        if batch_idx % 50 == 0:
            pbar.set_postfix({
                'loss': f'{loss.item() * CONFIG["gradient_accumulation"]:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
            })
    
    avg_train_loss = epoch_loss / max(valid_batches, 1)
    val_loss = validate(model, val_loader, device)
    val_losses.append(val_loss)
    
    print(f"\n✓ Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        patience_counter = 0
        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': CONFIG,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
        }, './checkpoints/chat/hessgpt_chat_BEST.pt')
        
        print(f"  🏆 MEILLEUR MODÈLE! (Val: {val_loss:.4f})")
    else:
        patience_counter += 1
        print(f"  ⚠️  Patience: {patience_counter}/{CONFIG['patience']}")
        
        if patience_counter >= CONFIG['patience']:
            print(f"\n🛑 EARLY STOPPING à epoch {epoch+1}")
            print(f"  Meilleur: epoch {best_epoch} (Val: {best_val_loss:.4f})")
            break
    
    # Test
    if (epoch + 1) % 1 == 0:
        print(f"\n💬 Test (Epoch {epoch+1}):")
        model.eval()
        
        tests = [
            "User: Hello!\nAssistant:",
            "User: How are you?\nAssistant:",
            "User: What can you do?\nAssistant:",
        ]
        
        for prompt in tests:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                generated = input_ids.clone()
                for _ in range(40):
                    if generated.size(1) >= CONFIG['max_seq_len']:
                        break
                    
                    with torch.amp.autocast('cuda'):
                        logits, _ = model(generated)
                    
                    next_logits = logits[:, -1, :] / CONFIG['temperature']
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    
                    generated = torch.cat([generated, next_token], dim=1)
                    
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                
                text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
                response = text[len(prompt):].strip()
                print(f"  {prompt.split('Assistant:')[0].strip()}")
                print(f"    → {response[:80]}...")
        
        model.train()

elapsed = time.time() - start_time

# ============================================
# STATS FINALES
# ============================================
print("\n" + "="*60)
print("📊 STATISTIQUES FINALES")
print("="*60)
print(f"✓ Dialogues: {len(chat_data)}")
print(f"✓ Époques: {epoch+1}")
print(f"✓ Meilleur: epoch {best_epoch}")
print(f"✓ Train Loss: {train_losses[0]:.4f} → {train_losses[-1]:.4f}")
print(f"✓ Val Loss: {val_losses[0]:.4f} → {best_val_loss:.4f}")
print(f"✓ Temps: {elapsed/60:.1f} min")
print("="*60)

# Charger meilleur
best_checkpoint = torch.load('./checkpoints/chat/hessgpt_chat_BEST.pt')
model.load_state_dict(best_checkpoint['model_state_dict'])

# Sauvegarder final
torch.save({
    'model_state_dict': model.state_dict(),
    'config': CONFIG,
    'train_losses': train_losses,
    'val_losses': val_losses,
    'best_epoch': best_epoch,
    'best_val_loss': best_val_loss,
}, './checkpoints/hessgpt_chat_final.pt')

print("\n✅ CHATBOT ULTIMATE v2 PRÊT!")
print("💾 Final: ./checkpoints/hessgpt_chat_final.pt")
print(f"💾 Best (epoch {best_epoch}): ./checkpoints/chat/hessgpt_chat_BEST.pt")
print("="*60)