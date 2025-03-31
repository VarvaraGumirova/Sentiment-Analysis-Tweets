import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import re  # For text preprocessing

# Text preprocessing function
def preprocess(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove everything except letters and spaces
    return text

# Load data
df = pd.read_csv(
    r"C:\Users\vgoum\vscode\python_pytorch_withGPT\training.1600000.processed.noemoticon.csv",
    encoding="latin-1", header=None)

# Keep only label and text columns
df = df[[0, 5]]
df.columns = ['label', 'text']

# Convert labels: only 0 and 4
df = df[df['label'].isin([0, 4])]
df['label'] = df['label'].apply(lambda x: 1 if x == 4 else 0)

# Apply preprocessing to texts
df['text'] = df['text'].apply(preprocess)

# UNDERSAMPLING: Balance the number of positive and negative tweets
pos_df = df[df['label'] == 1]
neg_df = df[df['label'] == 0].sample(n=len(pos_df), random_state=42)

df = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# Shuffle and take the first 100,000 samples
df = df[:100000]

# Tokenization
tokenized_text = []
for text in df['text']:
    tokens = text.split()
    tokenized_text.append(tokens)

# Create a vocabulary
word2index = {}
index = 1
for text in tokenized_text:
    for word in text:
        if word not in word2index:
            word2index[word] = index
            index += 1

# Convert tokenized texts to indexed sequences
indexed_texts = []
for text in tokenized_text:
    indexed_sentence = [word2index.get(word, 0) for word in text]
    indexed_texts.append(indexed_sentence)

# Padding
max_len = max(len(text) for text in indexed_texts)
padded_texts = []
for text in indexed_texts:
    if len(text) < max_len:
        text += [0] * (max_len - len(text))
    else:
        text = text[:max_len]
    padded_texts.append(text)

# Convert to tensors
X = torch.tensor(padded_texts)  # Not float!
Y = torch.tensor(df['label'].values).float().unsqueeze(1)

# Vocabulary size and embedding dimensions
vocab_size = len(word2index) + 1
embedding_dim = 100
hidden_dim = 100  # Hidden layer size

# Model with Embedding
class MyModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)  # First linear layer
        self.relu = nn.ReLU()  # Non-linearity
        self.dropout = nn.Dropout(0.3)  # Dropout for regularization
        self.linear2 = nn.Linear(hidden_dim, 1)  # Second linear layer

    def forward(self, x):
        embedded = self.embedding(x)  # (batch, max_len, emb_dim)
        averaged = embedded.mean(dim=1)  # Average over words
        hidden = self.relu(self.linear1(averaged))  # Apply first layer and ReLU
        dropped = self.dropout(hidden)  # Apply dropout
        out = self.linear2(dropped)  # Apply second layer
        return out  # No sigmoid, as we use BCEWithLogitsLoss

# Create the model
model = MyModel(vocab_size, embedding_dim, hidden_dim)

# Loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()  # Includes sigmoid
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(200):
    model.train()  # Training mode
    optimizer.zero_grad()  # Clear gradients
    Y_pred = model(X)  # Forward pass
    loss = loss_fn(Y_pred, Y)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# === TEST ON A NEW TWEET ===

def predict_new_tweet(new_text, model, word2index, max_len):
    # Apply the same preprocessing
    new_text = preprocess(new_text)

    # Tokenization
    tokens = new_text.split()

    # Indexing
    indexed = [word2index.get(word, 0) for word in tokens]

    # Padding
    if len(indexed) < max_len:
        indexed += [0] * (max_len - len(indexed))
    else:
        indexed = indexed[:max_len]

    # Convert to tensor
    x_test = torch.tensor([indexed])  # (1, max_len)

    # Prediction
    model.eval()  # Evaluation mode
    with torch.no_grad():
        prediction = model(x_test)
        prob = torch.sigmoid(prediction).item()  # Apply sigmoid to get probability

    return prob

# Example test
new_text = "I love this beautiful day"
prob = predict_new_tweet(new_text, model, word2index, max_len)
print(f"\nPrediction: {prob:.4f}")
if prob >= 0.5:
    print("Positive tweet")
else:
    print("Negative tweet")