import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import collections
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

# --- КОНФИГУРАЦИЯ ---
CONFIG = {
    'vocab_size': 30522,
    'dim': 128,           
    'n_layers': 3,         
    'n_heads': 4,
    'n_experts': 4,         
    'top_k': 1,             
    'seq_len': 64,
    'batch_size': 16,      
    'lr_pre': 1e-3,
    'lr_fine': 5e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'pretrain_steps': 200,
    'finetune_epochs': 6,
    'labels': [
        'Computer Science', 'Physics', 'Mathematics', 
        'Statistics', 'Quantitative Biology', 'Quantitative Finance'
    ]
}

# --- 1. АРХИТЕКТУРА MoE ---

class MoELayer(nn.Module):
    def __init__(self, dim, n_experts, top_k):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        # Список экспертов (каждый - небольшая нейросеть)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            ) for _ in range(n_experts)
        ])
        # Gating Network (Маршрутизатор)
        self.gate = nn.Linear(dim, n_experts)

    def forward(self, x):
        # x: [Batch, Seq, Dim]
        gate_logits = self.gate(x)
        
        # Выбираем Top-K экспертов для каждого токена
        weights, selected_experts = torch.topk(gate_logits, self.top_k)
        weights = F.softmax(weights, dim=-1) # [Batch, Seq, TopK]
        
        output = torch.zeros_like(x)
        
        # Проходим по выбранным экспертам (цикл для читаемости кода)
        for k in range(self.top_k):
            indices = selected_experts[:, :, k]
            w = weights[:, :, k].unsqueeze(-1)
            
            for e_idx in range(self.n_experts):
                # Маска: 1 там, где выбран эксперт e_idx
                mask = (indices == e_idx).unsqueeze(-1).float()
                if mask.sum() > 0:
                    expert_out = self.experts[e_idx](x)
                    output += mask * w * expert_out
                    
        return output, selected_experts

class MoEBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = nn.MultiheadAttention(cfg['dim'], cfg['n_heads'], batch_first=True)
        self.ln1 = nn.LayerNorm(cfg['dim'])
        self.moe = MoELayer(cfg['dim'], cfg['n_experts'], cfg['top_k'])
        self.ln2 = nn.LayerNorm(cfg['dim'])

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        moe_out, indices = self.moe(x)
        x = self.ln2(x + moe_out)
        return x, indices

class MoETransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding = nn.Embedding(cfg['vocab_size'], cfg['dim'])
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg['seq_len'], cfg['dim']))
        self.layers = nn.ModuleList([MoEBlock(cfg) for _ in range(cfg['n_layers'])])
        
        # Головы (Heads)
        self.mlm_head = nn.Linear(cfg['dim'], cfg['vocab_size'])
        self.cls_head = nn.Linear(cfg['dim'], len(cfg['labels']))

    def forward(self, x, task='mlm'):
        B, S = x.shape
        x = self.embedding(x) + self.pos_emb[:, :S, :]
        
        expert_logs = []
        for layer in self.layers:
            x, indices = layer(x)
            expert_logs.append(indices) # Сохраняем индексы для анализа
            
        if task == 'mlm':
            return self.mlm_head(x), expert_logs
        elif task == 'cls':
            # Mean Pooling: усредняем векторы всех токенов для классификации текста
            x = x.mean(dim=1)
            return self.cls_head(x), expert_logs

# --- 2. ПОДГОТОВКА ДАННЫХ ---
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def get_wiki_loader():
    """Стриминг Википедии для предобучения"""
    print(">>> Загрузка Википедии (streaming)...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    ds = ds.take(2000) # Берем небольшой кусок для примера
    
    texts = [x['text'][:1000] for x in ds if len(x['text']) > 100]
    tokens = tokenizer(texts, padding='max_length', truncation=True, max_length=CONFIG['seq_len'], return_tensors='pt')
    return DataLoader(TensorDataset(tokens['input_ids']), batch_size=CONFIG['batch_size'], shuffle=True)

def get_csv_loaders():
    """Чтение вашего train.csv"""
    print(">>> Обработка CSV файлов...")
    if not os.path.exists('train.csv'):
        print("ОШИБКА: Файл train.csv не найден! Пожалуйста, положите файл рядом со скриптом.")
        # Создадим dummy файл, чтобы код не падал, если вы просто тестируете скрипт
        create_dummy_data() 
    
    df = pd.read_csv("train.csv")
    
    # 1. Объединяем Title и Abstract
    # Используем fillna(''), чтобы не падать на пустых полях
    texts = (df['TITLE'].fillna('') + " " + df['ABSTRACT'].fillna('')).tolist()
    
    # 2. Извлекаем метки (One-Hot)
    # Колонки ['Computer Science', 'Physics', ...] уже содержат 0 и 1
    labels = df[CONFIG['labels']].values.astype(float)
    
    # Делим на train/val, так как в test.csv нет меток для проверки
    train_txt, val_txt, train_y, val_y = train_test_split(texts, labels, test_size=0.1, random_state=42)
    
    def create_dataset(txt, y):
        enc = tokenizer(txt, padding='max_length', truncation=True, max_length=CONFIG['seq_len'], return_tensors='pt')
        return TensorDataset(enc['input_ids'], torch.tensor(y, dtype=torch.float32))

    return (
        DataLoader(create_dataset(train_txt, train_y), batch_size=CONFIG['batch_size'], shuffle=True),
        DataLoader(create_dataset(val_txt, val_y), batch_size=CONFIG['batch_size'])
    )

def create_dummy_data():
    """Создает фейковый датасет, если оригинального нет"""
    print("!!! СОЗДАЮ FAKE DATASET ДЛЯ ТЕСТА !!!")
    data = {
        'ID': range(50),
        'TITLE': ['Physics paper'] * 25 + ['CS AI paper'] * 25,
        'ABSTRACT': ['Gravity lens stars...'] * 25 + ['Neural network code...'] * 25
    }
    # Заполняем метки
    for l in CONFIG['labels']: data[l] = 0
    df = pd.DataFrame(data)
    df.loc[:24, 'Physics'] = 1
    df.loc[25:, 'Computer Science'] = 1
    df.to_csv('train.csv', index=False)

# --- 3. ТРЕНИРОВОЧНЫЙ ЦИКЛ ---

def train():
    model = MoETransformer(CONFIG).to(CONFIG['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr_pre'])
    
    # === PHASE 1: PRE-TRAINING (MLM) ===
    # Учим модель понимать язык
    print("\n=== START MLM PRE-TRAINING ===")
    wiki_loader = get_wiki_loader()
    criterion_mlm = nn.CrossEntropyLoss()
    model.train()
    
    for step, (batch_ids,) in enumerate(tqdm(wiki_loader, total=CONFIG['pretrain_steps'])):
        if step >= CONFIG['pretrain_steps']: break
        batch_ids = batch_ids.to(CONFIG['device'])
        
        # Создаем маски для MLM (15% токенов закрываем)
        rand = torch.rand_like(batch_ids, dtype=torch.float)
        mask = (rand < 0.15) & (batch_ids != tokenizer.pad_token_id)
        
        targets = batch_ids.clone()
        targets[~mask] = -100 # -100 игнорируется лоссом
        batch_ids[mask] = tokenizer.mask_token_id
        
        optimizer.zero_grad()
        logits, _ = model(batch_ids, task='mlm')
        
        loss = criterion_mlm(logits.view(-1, CONFIG['vocab_size']), targets.view(-1))
        loss.backward()
        optimizer.step()

    # === PHASE 2: FINE-TUNING (Classification) ===
    # Учим модель классифицировать науки
    print("\n=== START FINE-TUNING ===")
    train_loader, val_loader = get_csv_loaders()
    
    # Меняем LR и Loss
    for g in optimizer.param_groups: g['lr'] = CONFIG['lr_fine']
    criterion_cls = nn.BCEWithLogitsLoss() # Идеально для Multilabel (0/1)
    
    for epoch in range(CONFIG['finetune_epochs']):
        model.train()
        total_loss = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch_x, batch_y = batch_x.to(CONFIG['device']), batch_y.to(CONFIG['device'])
            
            optimizer.zero_grad()
            logits, _ = model(batch_x, task='cls')
            loss = criterion_cls(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Loss: {total_loss / len(train_loader):.4f}")

    # === PHASE 3: EVALUATION & EXPERT ANALYSIS ===
    print("\n=== EVALUATION & EXPERT ANALYSIS ===")
    model.eval()
    all_preds, all_true = [], []
    
    # Словарь: {Label -> {Layer_Idx -> Counter(Expert_Indices)}}
    expert_stats = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(CONFIG['device'])
            logits, expert_logs = model(batch_x, task='cls')
            
            # Предсказания (Sigmoid > 0.5 = 1)
            preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            true_y = batch_y.cpu().numpy()
            
            all_preds.append(preds)
            all_true.append(true_y)
            
            # --- АНАЛИЗ ЭКСПЕРТОВ ---
            # expert_logs - список тензоров [Batch, Seq, TopK] для каждого слоя
            for i in range(batch_x.size(0)): # Для каждого примера в батче
                # Смотрим, какие метки реально стоят у примера (например, Physics=1)
                active_labels = [CONFIG['labels'][idx] for idx, val in enumerate(true_y[i]) if val == 1]
                
                if not active_labels: continue
                
                # Для этого примера смотрим, какие эксперты работали
                for layer_idx, layer_tensor in enumerate(expert_logs):
                    # Собираем индексы экспертов со всех токенов этого текста
                    # flatten() превращает матрицу [Seq, TopK] в плоский список
                    experts_used = layer_tensor[i].flatten().cpu().tolist()
                    
                    for label in active_labels:
                        expert_stats[label][layer_idx].update(experts_used)

    all_preds = np.vstack(all_preds)
    all_true = np.vstack(all_true)
    
    print(f"Micro F1-Score: {f1_score(all_true, all_preds, average='micro'):.4f}")
    print(f"Exact Match Accuracy: {accuracy_score(all_true, all_preds):.4f}")

    print("\n--- Визуализация специализации экспертов ---")
    
    last_layer_idx = CONFIG['n_layers'] - 1
    # Создаем матрицу: Метки x Эксперты
    # Число экспертов берем из конфига
    matrix = np.zeros((len(CONFIG['labels']), CONFIG['n_experts']))

    for i, label in enumerate(CONFIG['labels']):
        stats = expert_stats[label][last_layer_idx]
        total_uses = sum(stats.values())
        if total_uses > 0:
            for exp_id, count in stats.items():
                # Записываем относительную частоту (процент включения эксперта для этой метки)
                matrix[i, exp_id] = count / total_uses

    # Отрисовка
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        matrix, 
        annot=True,          # Показывать числа в ячейках
        fmt=".2f",           # Формат чисел
        cmap="YlGnBu",       # Градация цвета (желтый-синий)
        xticklabels=[f"Exp {i}" for i in range(CONFIG['n_experts'])],
        yticklabels=CONFIG['labels']
    )
    
    plt.title(f"Специализация экспертов на слое {last_layer_idx} (MoE)")
    plt.xlabel("Индексы экспертов")
    plt.ylabel("Научные области (Labels)")
    plt.tight_layout()
    
    # Сохраняем и показываем
    plt.savefig("expert_specialization.png")
    print("График сохранен в файл expert_specialization.png")
    plt.show()

if __name__ == "__main__":
    train()