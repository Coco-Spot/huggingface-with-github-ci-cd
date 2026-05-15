import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
import os
import json
import numpy as np
import re
import string
import joblib

# 1. Load Data
data_path = "imdb_balanced_10k.csv"
glove_path = "tiny_glove.json"

if not os.path.exists(data_path) or not os.path.exists(glove_path):
    print(f"Dataset or glove file not found. Ensure {data_path} and {glove_path} exist.")
    exit(1)

print("Loading dataset...")
df = pd.read_csv(data_path)
X_text = df['text'].values
y_labels = df['label'].values

# 2. Preprocessing (Tokenization & GloVe Embedding)
print("Loading GloVe...")
with open(glove_path, 'r', encoding='utf-8') as f:
    glove_dict = json.load(f)

vocab = {"<PAD>": 0, "<UNK>": 1}
embed_dim = len(next(iter(glove_dict.values())))

# Build embedding matrix
embedding_matrix = [np.zeros(embed_dim), np.random.normal(scale=0.1, size=(embed_dim,))]

for word, vec in glove_dict.items():
    if word not in vocab:
        vocab[word] = len(vocab)
        embedding_matrix.append(np.array(vec))

embedding_matrix = torch.tensor(np.array(embedding_matrix), dtype=torch.float32)

print("Tokenizing texts...")
# 优化1：增加截断长度。原有的 150 会切掉长评论中结尾的情感反转部分
MAX_LEN = 300
X_indices = []

def tokenize(text):
    text = str(text).lower()
    text = re.sub(r'[' + string.punctuation + ']', ' ', text)
    return text.split()

for text in X_text:
    tokens = tokenize(text)
    indices = [vocab.get(t, 1) for t in tokens[:MAX_LEN]]
    # Pad if necessary
    if len(indices) < MAX_LEN:
        indices += [0] * (MAX_LEN - len(indices))
    X_indices.append(indices)

X = np.array(X_indices)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# DataLoader for batched training (helps scale to larger datasets like 10k)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 3. Define Neural Network Model with Embedding Layer
class SentimentNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, pretrained_weights):
        super(SentimentNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 解冻词向量
        self.embedding.weight = nn.Parameter(pretrained_weights, requires_grad=True)
        
        # 优化1：采用 Bi-LSTM 模型
        hidden_dim = 128
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, 
                            bidirectional=True, batch_first=True, dropout=0.5)
        
        # 优化2：将使用全局平均池化(Avg)和全局最大池化(Max)组合，因此输出特征尺寸为 hidden_dim * 4
        self.fc1 = nn.Linear(hidden_dim * 4, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch_size, max_len]
        embeds = self.embedding(x) # shape: [batch_size, max_len, embed_dim]
        
        # lstm_out shape: [batch_size, max_len, hidden_dim * 2]
        lstm_out, _ = self.lstm(embeds)
        
        # 优化3：同时捕捉“语气最强烈的关键词”(Max)与“整体句子的连贯大意”(Avg)
        avg_pool = torch.mean(lstm_out, dim=1)
        max_pool, _ = torch.max(lstm_out, dim=1)
        hidden_cat = torch.cat((avg_pool, max_pool), dim=1)
        
        out = self.fc1(hidden_cat)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

model = SentimentNN(len(vocab), embed_dim, embedding_matrix)

# 4. Training
criterion = nn.BCELoss()
# 优化4：加入小额正则化（Weight Decay）减缓它在训练集快速过拟合
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 增加轮数让它充分收敛
epochs = 15
print("Starting training...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    if (epoch+1) % 2 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')

# Evaluate
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_labels = (predictions > 0.5).float()
    accuracy = (predicted_labels == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f"Test Accuracy: {accuracy:.4f}")

# 5. Save the Model and Artifacts
print("Saving model artifacts...")
torch.save(model.state_dict(), "sentiment_nn.pth")
joblib.dump(vocab, "vocab.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("Saved sentiment_nn.pth, vocab.pkl, and label_encoder.pkl")
