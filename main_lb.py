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

# --- 1. АРХИТЕКТУРА С LOAD BALANCING ---

class MoELayer(nn.Module):
    def __init__(self, dim, n_experts, top_k):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            ) for _ in range(n_experts)
        ])
        self.gate = nn.Linear(dim, n_experts)

    def forward(self, x):
        # x: [B, S, C]
        gate_logits = self.gate(x)
        
        # --- Расчет Load Balancing (Auxiliary) Loss ---
        # probs: вероятность выбора каждого эксперта (мягкое распределение)
        probs = F.softmax(gate_logits, dim=-1) # [B, S, n_experts]
        
        # Выбор Top-K
        weights, selected_experts = torch.topk(gate_logits, self.top_k)
        topk_probs = F.softmax(weights, dim=-1)
        
        # ce_i: доля токенов, реально отправленных каждому эксперту (жесткое распределение)
        count_mask = torch.zeros_like(probs).scatter_(-1, selected_experts, 1.0)
        
        # Усредняем по батчу и по всей длине последовательности
        me_i = probs.mean(dim=(0, 1))       # Mean Expectation
        ce_i = count_mask.mean(dim=(0, 1))  # Count Expectation
        
        # Формула балансировки: n * sum(me_i * ce_i)
        aux_loss = self.n_experts * torch.sum(me_i * ce_i)
        # ----------------------------------------------

        output = torch.zeros_like(x)
        for k in range(self.top_k):
            indices = selected_experts[:, :, k]
            w = topk_probs[:, :, k].unsqueeze(-1)
            for e_idx in range(self.n_experts):
                mask = (indices == e_idx).unsqueeze(-1).float()
                if mask.sum() > 0:
                    output += mask * w * self.experts[e_idx](x)
                    
        return output, selected_experts, aux_loss

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
        
        # Теперь блок возвращает еще и aux_loss из слоя MoE
        moe_out, indices, aux_loss = self.moe(x)
        x = self.ln2(x + moe_out)
        return x, indices, aux_loss

class MoETransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding = nn.Embedding(cfg['vocab_size'], cfg['dim'])
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg['seq_len'], cfg['dim']))
        self.layers = nn.ModuleList([MoEBlock(cfg) for _ in range(cfg['n_layers'])])
        self.mlm_head = nn.Linear(cfg['dim'], cfg['vocab_size'])
        self.cls_head = nn.Linear(cfg['dim'], len(cfg['labels']))

    def forward(self, x, task='mlm'):
        B, S = x.shape
        x = self.embedding(x) + self.pos_emb[:, :S, :]
        
        total_aux_loss = 0
        expert_logs = []
        for layer in self.layers:
            x, indices, l_aux = layer(x)
            expert_logs.append(indices)
            total_aux_loss += l_aux
            
        if task == 'mlm':
            return self.mlm_head(x), expert_logs, total_aux_loss
        elif task == 'cls':
            x = x.mean(dim=1)
            return self.cls_head(x), expert_logs, total_aux_loss

# --- 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (ДАННЫЕ) ---

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def get_wiki_loader():
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    ds = ds.take(1000)
    texts = [x['text'][:1000] for x in ds if len(x['text']) > 100]
    tokens = tokenizer(texts, padding='max_length', truncation=True, max_length=CONFIG['seq_len'], return_tensors='pt')
    return DataLoader(TensorDataset(tokens['input_ids']), batch_size=CONFIG['batch_size'], shuffle=True)

def get_csv_loaders():
    if not os.path.exists('train.csv'):
        # Dummy data generation if file not found
        df = pd.DataFrame({'TITLE': ['T']*50, 'ABSTRACT': ['A']*50})
        for l in CONFIG['labels']: df[l] = np.random.randint(0,2,50)
        df.to_csv('train.csv', index=False)
    
    df = pd.read_csv("train.csv")
    texts = (df['TITLE'].fillna('') + " " + df['ABSTRACT'].fillna('')).tolist()
    labels = df[CONFIG['labels']].values.astype(float)
    t_txt, v_txt, t_y, v_y = train_test_split(texts, labels, test_size=0.1, random_state=42)
    
    def mk_ds(txt, y):
        enc = tokenizer(txt, padding='max_length', truncation=True, max_length=CONFIG['seq_len'], return_tensors='pt')
        return TensorDataset(enc['input_ids'], torch.tensor(y, dtype=torch.float32))

    return DataLoader(mk_ds(t_txt, t_y), batch_size=CONFIG['batch_size'], shuffle=True), \
           DataLoader(mk_ds(v_txt, v_y), batch_size=CONFIG['batch_size'])

# --- 3. ЦИКЛЫ ОБУЧЕНИЯ ---

def train():
    model = MoETransformer(CONFIG).to(CONFIG['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr_pre'])
    alpha = 0.01 # Коэффициент влияния Load Balancing Loss
    
    # PHASE 1: PRE-TRAINING
    print("\n>>> Phase 1: Pre-training (MLM + Load Balancing)")
    wiki_loader = get_wiki_loader()
    crit_mlm = nn.CrossEntropyLoss()
    model.train()
    
    for step, (batch_ids,) in enumerate(tqdm(wiki_loader, total=CONFIG['pretrain_steps'])):
        if step >= CONFIG['pretrain_steps']: break
        batch_ids = batch_ids.to(CONFIG['device'])
        
        rand = torch.rand_like(batch_ids, dtype=torch.float)
        mask = (rand < 0.15) & (batch_ids != tokenizer.pad_token_id)
        targets = batch_ids.clone(); targets[~mask] = -100
        batch_ids[mask] = tokenizer.mask_token_id
        
        optimizer.zero_grad()
        logits, _, aux_loss = model(batch_ids, task='mlm')
        
        loss = crit_mlm(logits.view(-1, CONFIG['vocab_size']), targets.view(-1))
        total_loss = loss + alpha * aux_loss
        total_loss.backward()
        optimizer.step()

    # PHASE 2: FINE-TUNING
    print("\n>>> Phase 2: Fine-tuning (CLS + Load Balancing)")
    train_loader, val_loader = get_csv_loaders()
    for g in optimizer.param_groups: g['lr'] = CONFIG['lr_fine']
    crit_cls = nn.BCEWithLogitsLoss()
    
    for epoch in range(CONFIG['finetune_epochs']):
        model.train()
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch_x, batch_y = batch_x.to(CONFIG['device']), batch_y.to(CONFIG['device'])
            optimizer.zero_grad()
            logits, _, aux_loss = model(batch_x, task='cls')
            loss = crit_cls(logits, batch_y)
            total_loss = loss + alpha * aux_loss
            total_loss.backward()
            optimizer.step()

    # PHASE 3: EVALUATION & HEATMAP (НЕ МЕНЯЕМ ПОСТРОЕНИЕ)
    print("\n>>> Phase 3: Evaluation & Visualization")
    model.eval()
    all_preds, all_true = [], []
    expert_stats = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(CONFIG['device'])
            logits, expert_logs, _ = model(batch_x, task='cls')
            preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            true_y = batch_y.cpu().numpy()
            all_preds.append(preds); all_true.append(true_y)
            
            for i in range(batch_x.size(0)):
                active_labels = [CONFIG['labels'][idx] for idx, val in enumerate(true_y[i]) if val == 1]
                if not active_labels: continue
                for layer_idx, layer_tensor in enumerate(expert_logs):
                    experts_used = layer_tensor[i].flatten().cpu().tolist()
                    for label in active_labels:
                        expert_stats[label][layer_idx].update(experts_used)

    # --- HEATMAP VISUALIZATION (ORIGINAL) ---
    all_preds = np.vstack(all_preds); all_true = np.vstack(all_true)
    print(f"F1: {f1_score(all_true, all_preds, average='micro'):.4f}")
    
    n_layers = CONFIG['n_layers']
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 6), sharey=True)
    if n_layers == 1: axes = [axes]

    for layer_idx in range(n_layers):
        matrix = np.zeros((len(CONFIG['labels']), CONFIG['n_experts']))
        for i, label in enumerate(CONFIG['labels']):
            stats = expert_stats[label][layer_idx]
            total_uses = sum(stats.values())
            if total_uses > 0:
                for exp_id, count in stats.items():
                    matrix[i, exp_id] = count / total_uses
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[layer_idx],
                    cbar=False, xticklabels=[f"E{i}" for i in range(CONFIG['n_experts'])],
                    yticklabels=CONFIG['labels'] if layer_idx == 0 else False)
        axes[layer_idx].set_title(f"Layer {layer_idx}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    train()