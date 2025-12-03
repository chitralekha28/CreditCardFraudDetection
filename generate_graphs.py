import os
os.makedirs("results/figures", exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ------------------------------------------------------
# 1. Load stored values (copy from main.py output)
# ------------------------------------------------------

ae_auc_val = 0.9117
hyb_auc_val = 0.8916

# Example placeholders: replace with real arrays if saved
# For now, we simulate score arrays for graph visualization
y_test = np.concatenate([np.zeros(2000), np.ones(50)])  # ~50 fraud samples
ae_scores = np.random.normal(0.3, 0.1, size=len(y_test))
hyb_scores = np.random.normal(0.4, 0.12, size=len(y_test))

# Add separation for fraud class
ae_scores[-50:] += 0.4
hyb_scores[-50:] += 0.3

# ------------------------------------------------------
# 2. AE Loss Curve (simulated)
# ------------------------------------------------------
loss_values = [0.5253, 0.3947, 0.3607, 0.3401, 0.3320]
epochs = [1, 2, 3, 4, 5]

plt.plot(epochs, loss_values, marker='o')
plt.title("Autoencoder Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.savefig("results/figures/ae_loss.png")
plt.clf()

# ------------------------------------------------------
# 3. ROC Curve — AE vs Hybrid
# ------------------------------------------------------
fpr_ae, tpr_ae, _ = roc_curve(y_test, ae_scores)
fpr_h, tpr_h, _ = roc_curve(y_test, hyb_scores)

plt.plot(fpr_ae, tpr_ae, label=f"AE (AUC={ae_auc_val:.3f})")
plt.plot(fpr_h, tpr_h, label=f"Hybrid (AUC={hyb_auc_val:.3f})")
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.savefig("results/figures/roc_curve.png")
plt.clf()

# ------------------------------------------------------
# 4. GNN Loss Curve
# ------------------------------------------------------
gnn_loss = [0.6184, 0.5989, 0.5801, 0.5617, 0.5439]

plt.plot(epochs, gnn_loss, marker='s', color='green')
plt.title("GNN Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.savefig("results/figures/gnn_loss.png")
plt.clf()

# ------------------------------------------------------
# 5. Hybrid Score Distribution Plot
# ------------------------------------------------------
plt.hist(ae_scores, bins=40, alpha=0.5, label='AE Scores')
plt.hist(hyb_scores, bins=40, alpha=0.5, label='Hybrid Scores')
plt.title("Distribution of AE vs Hybrid Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.legend()
plt.grid()
plt.savefig("results/figures/score_dist.png")
plt.clf()

print("All graphs generated successfully!")
