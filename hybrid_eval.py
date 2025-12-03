import numpy as np

def hybrid_score(ae_scores, gnn_probs, alpha=0.6):
    return alpha * ae_scores + (1 - alpha) * gnn_probs
