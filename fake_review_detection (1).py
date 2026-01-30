import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import re

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Load Data
dataset_path = 'fake_reviews_dataset.csv'
try:
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    print(f"Error: File {dataset_path} not found. Please make sure the file is in the same directory.")
    exit()

# Handle missing values
df.dropna(subset=['text_', 'label', 'category'], inplace=True)

# --- Analysis: Find Category with Most Fake Reviews ---
print("Analyzing fake reviews per category...")
fake_reviews_df = df[df['label'] == 'CG']
category_counts = fake_reviews_df['category'].value_counts()
if not category_counts.empty:
    most_fake_category = category_counts.idxmax()
    count = category_counts.max()
    print(f"\n[INFO] Category with the most fake reviews: {most_fake_category} (Count: {count})")
    print("Top 5 Categories with Fake Reviews:") 
    print(category_counts.head(5))
else:
    print("[INFO] No fake reviews found for analysis.")
print("-" * 30)
# ----------------------------------------------------

# 2. Preprocessing & Vectorization (Spacy + TF-IDF with N-grams)
print("Loading Spacy model...")
try:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    print("Error: Spacy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    exit()
except ImportError:
    print("Error: Spacy not installed. Please run: pip install spacy")
    exit()

def spacy_tokenizer(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    return tokens

print("Vectorizing text with Spacy tokenizer (Unigrams + Bigrams)...")
# Increased max_features and added ngram_range for better accuracy
max_features = 10000 
vectorizer = TfidfVectorizer(max_features=max_features, 
                             tokenizer=spacy_tokenizer, 
                             token_pattern=None,
                             ngram_range=(1, 2)) # Use unigrams and bigrams

X = vectorizer.fit_transform(df['text_']).toarray()

# 4. Label Encoding
print("Encoding labels...")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])
print(f"Label Mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. PyTorch Dataset
class ReviewDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels) 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = ReviewDataset(X_train, y_train)
test_dataset = ReviewDataset(X_test, y_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 7. Neural Network Architecture (Improved)
class BinaryNN(nn.Module):
    def __init__(self, input_dim):
        super(BinaryNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256) # Increased neurons
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4) # Increased dropout
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1) 

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out

input_dim = max_features
model = BinaryNN(input_dim).to(device)

# 8. Training Loop
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005) # Lower learning rate for better convergence

num_epochs = 8 # Increased epochs
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)
        labels = labels.unsqueeze(1) 

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# 9. Evaluation
print("Evaluating model...")
model.eval()
y_pred_list = []
y_true_list = []

with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        predicted = torch.round(torch.sigmoid(outputs)) 
        
        # Robust scalar conversion
        y_pred_list.extend(predicted.squeeze().detach().cpu().tolist())
        y_true_list.extend(labels.squeeze().detach().cpu().tolist())

y_pred_list = [int(i) for i in y_pred_list] 
y_true_list = [int(i) for i in y_true_list]

print("\nClassification Report:")
print(classification_report(y_true_list, y_pred_list, target_names=label_encoder.classes_))
print(f"Accuracy: {accuracy_score(y_true_list, y_pred_list):.4f}")
