import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# 1. Load Data (Modify path as needed in your CI environment)
# Assuming the file is in the repository or downloaded via script. 
# Here we'll read from session-400/datasets/imdb_top_500.csv if run locally
data_path = "imdb_top_500.csv"
if not os.path.exists(data_path):
    print(f"Dataset not found at {data_path}. Please ensure it exists.")
    exit(1)

df = pd.read_csv(data_path)
# 假设数据集有 "review" 和 "sentiment" 列
X_text = df['review'].values
y_labels = df['sentiment'].values

# 2. Preprocessing (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X_text).toarray()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 3. Define Neural Network Model
class SentimentNN(nn.Module):
    def __init__(self, input_dim):
        super(SentimentNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

model = SentimentNN(X_train.shape[1])

# 4. Training
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
print("Starting training...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 2 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 5. Save the Model properly
print("Saving model artifacts...")
torch.save(model.state_dict(), "sentiment_nn.pth")
import joblib
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("Saved sentiment_nn.pth, tfidf_vectorizer.pkl, and label_encoder.pkl")