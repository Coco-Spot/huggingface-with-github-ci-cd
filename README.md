---
language: en
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
