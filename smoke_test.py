# smoke_test.py
import torch
from torch_geometric.data import Data
from gnn_model import SimpleGNN

# tiny random data
x = torch.randn(10, 30)            # 10 nodes, 30 features
edge_index = torch.randint(0, 10, (2, 20))
y = torch.randint(0, 2, (10,))

data = Data(x=x, edge_index=edge_index, y=y)
model = SimpleGNN(input_dim=30)

out = model(data)                  # forward
print("Smoke test output shape:", out.shape)
