---
language: 
  - en
  - zh
tags:
- PyTorch
- Text Classification
- Sentiment Analysis
- GloVe
datasets:
- imdb
widget:
- text: "This movie was absolutely wonderful. The acting was great and the plot was engaging."
- text: "I hated this film. It was a complete waste of time."
---

[English](#english) | [简体中文](#chinese)

<a id="english"></a>
# Simple Sentiment Analysis NN with GloVe Embeddings

This is a PyTorch-based neural network model for binary sentiment classification (positive/negative) on the IMDB dataset.

## Model Description

The model was built as a lightweight Feed-Forward Neural Network that utilizes pre-trained GloVe embeddings for token representations. It performs average pooling over the embedded tokens of a sequence to create a sentence-level representation, which then passes through two linear layers to output a sentiment probability.

### Architecture

- **Tokenization:** Custom whitespace and punctuation tokenizer. Sequence length is padded/truncated to 150 tokens.
- **Embedding Layer:** Pre-trained weights loaded from a custom `tiny_glove.json` dictionary. The embedding layer is frozen during training.
- **Pooling:** Average pooling across the sequence dimension `[batch_size, max_len, embed_dim] -> [batch_size, embed_dim]`.
- **Fully Connected Network:** 
  - `Linear(embed_dim, 64)` + `ReLU()`
  - `Linear(64, 1)` + `Sigmoid()`

## Training Data

The model was trained on a balanced subset of the IMDB movie reviews dataset containing 10,000 samples (`imdb_balanced_10k.csv`).
- **Train Split:** 8,000 samples (80%)
- **Test Split:** 2,000 samples (20%)

## Training Parameters

- **Loss Function:** Binary Cross Entropy Loss (`BCELoss`)
- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Batch Size:** 64
- **Epochs:** 10

## Artifacts included

- `sentiment_nn.pth`: The PyTorch `state_dict` of the trained model.
- `vocab.pkl`: A serialized dictionary mapping string tokens to integer indices (includes `<PAD>` and `<UNK>` tokens).
- `label_encoder.pkl`: Scikit-learn LabelEncoder used to encode string labels to binary classes.

## How to use

```python
import torch
import torch.nn as nn
import joblib
import re
import string
import numpy as np

# Load Vocab and Label Encoder
vocab = joblib.load("vocab.pkl")
label_encoder = joblib.load("label_encoder.pkl")
embed_dim = 300 # Depends on the Glove embeddings used

# Recreate the PyTorch model
class SentimentNN(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SentimentNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embeds = self.embedding(x)
        out = embeds.mean(dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

model = SentimentNN(len(vocab), embed_dim)
model.load_state_dict(torch.load("sentiment_nn.pth"), strict=False)
model.eval()

# Inference
MAX_LEN = 150
text = "This movie is amazing!"
text = str(text).lower()
text = re.sub(r'[' + string.punctuation + ']', ' ', text)
tokens = text.split()

indices = [vocab.get(t, 1) for t in tokens[:MAX_LEN]]
if len(indices) < MAX_LEN:
    indices += [0] * (MAX_LEN - len(indices))

input_tensor = torch.tensor([indices], dtype=torch.long)

with torch.no_grad():
    prediction = model(input_tensor).item()
    
# Get String Label
predicted_class = 1 if prediction > 0.5 else 0
print("Sentiment:", label_encoder.inverse_transform([predicted_class])[0])
print(f"Confidence: {prediction:.4f}")
```

---

<a id="chinese"></a>
# 基于 GloVe 词向量的简单情感分析神经网络

这是一个基于 PyTorch 构建的二分类情感分析（正向/负向）神经网络模型，在 IMDB 数据集上进行训练。

## 模型描述

该模型被设计为一个轻量级的前馈神经网络（Feed-Forward Neural Network），利用预训练的 GloVe 词向量作为 token 的嵌入表示。它通过对句子中所有 token 的嵌入向量进行平均池化（Average Pooling）来生成句子的全局特征表示，随后通过两个紧接的全连接线性层输出该句的情感概率预测。

### 模型架构

- **分词（Tokenization）:** 自定义基于空格和标点符号截断的分词器。输入序列长度固定统一为 150 个 tokens（不足在末尾补长，超出则截断）。
- **嵌入层（Embedding Layer）:** 初始化权重加载自项目中提供的 `tiny_glove.json` 预训练词向量文件字典。该层参数在训练期间冻结。
- **池化层（Pooling）:** 在序列维度即单个句子的长度方向上进行全局平均池化：`[batch_size, max_len, embed_dim] -> [batch_size, embed_dim]`。
- **全连接层（Fully Connected Network）:** 
  - 第一层 `Linear(embed_dim, 64)` + `ReLU()`
  - 第二层 `Linear(64, 1)` + `Sigmoid()` 激活函数输出最终概率。

## 训练数据

模型在一个内容分布平衡的 IMDB 电影评论高质量子集（包含 10,000 条样本数据，即项目中的 `imdb_balanced_10k.csv`）上完成训练。
- **训练集 (Train Split):** 8,000 条样本 (占比 80%)
- **测试集 (Test Split):** 2,000 条样本 (占比 20%)

## 训练参数

- **损失函数（Loss Function）:** 二元交叉熵损失函数 (`BCELoss`)
- **优化器（Optimizer）:** Adam
- **学习率（Learning Rate）:** 0.001
- **批大小（Batch Size）:** 64
- **总轮数（Epochs）:** 10

## 包含的文件

包含以下工作流自动生成的参数模型和必要的前置推理组件：
- `sentiment_nn.pth`: 训练完毕后的 PyTorch 网络模型 `state_dict` 权重字典。
- `vocab.pkl`: 一个映射字典，将文本中的 tokens 映射转化为具体的整型 ID（保留了包括 `<PAD>` 和 `<UNK>` 字段）。
- `label_encoder.pkl`: Scikit-learn 的 LabelEncoder 对象，用于预测通过后把二分类数值复原回原本的字符串文字标签。

## 如何使用

可以参考以下的 Python 推理模板逻辑将文本转为正确结果：

```python
import torch
import torch.nn as nn
import joblib
import re
import string
import numpy as np

# 加载词表和标签编码器
vocab = joblib.load("vocab.pkl")
label_encoder = joblib.load("label_encoder.pkl")
embed_dim = 300 # 需要与预训练词向量维度一致

# 重新声明 PyTorch 模型结构
class SentimentNN(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SentimentNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embeds = self.embedding(x)
        out = embeds.mean(dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

model = SentimentNN(len(vocab), embed_dim)
model.load_state_dict(torch.load("sentiment_nn.pth"), strict=False)
model.eval()

# 推理预测
MAX_LEN = 150
text = "This movie is amazing!"
text = str(text).lower()
text = re.sub(r'[' + string.punctuation + ']', ' ', text)
tokens = text.split()

indices = [vocab.get(t, 1) for t in tokens[:MAX_LEN]]
if len(indices) < MAX_LEN:
    indices += [0] * (MAX_LEN - len(indices))

input_tensor = torch.tensor([indices], dtype=torch.long)

with torch.no_grad():
    prediction = model(input_tensor).item()
    
# 获取结果
predicted_class = 1 if prediction > 0.5 else 0
print("情感判定 (Sentiment):", label_encoder.inverse_transform([predicted_class])[0])
print(f"置信度 (Confidence): {prediction:.4f}")
```
