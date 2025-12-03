# main.py
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data

from preprocess import load_and_preprocess
from autoencoder_model import build_autoencoder
from gnn_model import SimpleGNN
from hybrid_eval import hybrid_score

print("🚀 Starting Fraud Detection Pipeline...")

# ----------------------------------------------------
# 1. LOAD & PREPROCESS DATA
# ----------------------------------------------------
X, y, scaler = load_and_preprocess()

print("Data Loaded. Shape:", X.shape)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)
print("Dataset Split Completed.")

# ----------------------------------------------------
# 2. AUTOENCODER TRAINING
# ----------------------------------------------------
input_dim = X_train.shape[1]
autoencoder = build_autoencoder(input_dim)

print("Training Autoencoder...")
history = autoencoder.fit(
    X_train[y_train == 0],
    X_train[y_train == 0],
    epochs=5,
    batch_size=32,
    shuffle=True
)

print("Autoencoder Training Completed.")

# AE Inference
recon = autoencoder.predict(X_test)
ae_mse = np.mean(np.power(X_test - recon, 2), axis=1)

# AE AUC Score
ae_auc = roc_auc_score(y_test, ae_mse)
print("Autoencoder AUC:", ae_auc)

# Normalize MSE
ae_scores = (ae_mse - ae_mse.min()) / (ae_mse.max() - ae_mse.min())


# ----------------------------------------------------
# 3. SIMPLE GNN TRAINING
# ----------------------------------------------------
print("\nBuilding Simple Graph for GNN...")

# Random graph edges (demo purpose)
num_nodes = X_train[:3000].shape[0]
edge_index = torch.randint(0, num_nodes, (2, 10000))

# Convert to tensors
x_tensor = torch.tensor(X_train[:3000], dtype=torch.float)
y_tensor = torch.tensor(y_train[:3000], dtype=torch.long)

data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)

model = SimpleGNN(input_dim=input_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

print("Training GNN...")

for epoch in range(5):
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1} - GNN Loss: {loss.item():.4f}")

print("GNN Training Completed.")

# GNN prediction on the same mini graph
gnn_output = torch.softmax(out, dim=1)[:, 1].detach().numpy()

# Resize to match test length
gnn_scores = np.interp(np.arange(len(ae_scores)),
                       np.linspace(0, len(ae_scores)-1, len(gnn_output)),
                       gnn_output)


# ----------------------------------------------------
# 4. HYBRID SCORE
# ----------------------------------------------------
hybrid = hybrid_score(ae_scores, gnn_scores)

hybrid_auc = roc_auc_score(y_test, hybrid)
print("\n🔥 Hybrid Model AUC:", hybrid_auc)

print("\n🎉 All steps completed successfully!")
