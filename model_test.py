import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv
import torchvision.models as models
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
import numpy as np
import time
import json
from tqdm import tqdm
from collections import defaultdict

# ========================
# 1. COCO Data Preparation
# ========================
class COCOGraphDataset:
    def __init__(self, annotation_path, image_dir, max_objects=20):
        self.image_dir = image_dir
        with open(annotation_path) as f:
            self.data = json.load(f)
        
        # Extract object annotations
        self.annotations = defaultdict(list)
        for ann in self.data['annotations']:
            if ann['iscrowd'] == 0:  # Skip crowd annotations
                self.annotations[ann['image_id']].append({
                    'bbox': [ann['bbox'][0], ann['bbox'][1], 
                            ann['bbox'][0]+ann['bbox'][2], 
                            ann['bbox'][1]+ann['bbox'][3]],  # Convert to xyxy
                    'category_id': ann['category_id']
                })
        
        # Pre-trained ResNet for feature extraction
        self.cnn = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1] ) # Remove final FC
        self.cnn.eval()
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_id = list(self.annotations.keys())[idx]
        objects = self.annotations[img_id][:max_objects]
        
        # Extract features (simplified - real impl would use image patches)
        obj_features = torch.randn(len(objects), 512)  # Mock CNN features
        
        # Build fully connected graph
        edge_index = []
        for i in range(len(objects)):
            for j in range(len(objects)):
                if i != j:
                    edge_index.append([i, j])
        
        # Ground-truth relationships (simulated for demo)
        gt_edges = []
        if len(objects) >= 2:
            gt_edges.append([0, 1])  # Assume first two objects are related
        
        return Data(
            x=obj_features,
            edge_index=torch.tensor(edge_index).t().contiguous(),
            bboxes=torch.tensor([obj['bbox'] for obj in objects]),
            gt_edges=torch.tensor(gt_edges)
        )

# ========================
# 2. Advanced Sparsification
# ========================
class DynamicGraphPruner:
    def __init__(self, spatial_thresh=0.1, semantic_thresh=0.5):
        self.spatial_thresh = spatial_thresh
        self.semantic_thresh = semantic_thresh
        # Mock semantic similarity (real impl would use GloVe)
        self.semantic_sim = torch.rand(100, 100)  # 100 COCO categories
        
    def prune(self, graph_data):
        edge_indices = []
        for i, j in graph_data.edge_index.t():
            # Spatial score
            bbox_i = graph_data.bboxes[i]
            bbox_j = graph_data.bboxes[j]
            iou = box_iou(bbox_i.unsqueeze(0), bbox_j.unsqueeze(0)).item()
            
            # Semantic score (mock)
            cat_i = torch.randint(0, 100, (1,)).item()  # Simulated categories
            cat_j = torch.randint(0, 100, (1,)).item()
            semantic_score = self.semantic_sim[cat_i, cat_j]
            
            # Combined score
            if iou > self.spatial_thresh or semantic_score > self.semantic_thresh:
                edge_indices.append([i, j])
        
        graph_data.edge_index = torch.tensor(edge_indices).t().contiguous()
        return graph_data

# ========================
# 3. Scalable GNN Model
# ========================
class GNNCaptioner(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.gnn1 = GATConv(input_dim, hidden_dim)
        self.gnn2 = GATConv(hidden_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1000)  # Mock vocab size
        
    def forward(self, data):
        x = F.relu(self.gnn1(data.x, data.edge_index))
        x = self.gnn2(x, data.edge_index)
        
        # Mock caption generation (single step)
        hiddens, _ = self.lstm(x.unsqueeze(0))
        logits = self.fc(hiddens.squeeze(0))
        return logits

# ========================
# 4. Benchmarking Suite
# ========================
class EfficiencyBenchmark:
    @staticmethod
    def run(model, dataloader, pruner=None):
        metrics = {
            'inference_time': [],
            'edge_counts': [],
            'retention_rates': []
        }
        
        with torch.no_grad():
            for data in tqdm(dataloader):
                # Prune if specified
                if pruner:
                    data = pruner.prune(data)
                    metrics['edge_counts'].append(data.edge_index.shape[1])
                    
                    # Calculate retention
                    kept = set([tuple(e) for e in data.edge_index.t().tolist()])
                    gt = set([tuple(e) for e in data.gt_edges.tolist()])
                    metrics['retention_rates'].append(len(kept & gt) / max(1, len(gt)))
                
                # Time forward pass
                start = time.time()
                _ = model(data)
                metrics['inference_time'].append(time.time() - start)
        
        return metrics

# ========================
# 5. Main Experiment
# ========================
if __name__ == "__main__":
    # Config
    BATCH_SIZE = 32
    NUM_EPOCHS = 3
    
    # Mock dataset (replace with real COCO loader)
    dataset = [Data(x=torch.randn(10, 512), 
                    edge_index=torch.randint(0, 10, (2, 90)), 
                    bboxes=torch.rand(10, 4), 
                    gt_edges=torch.tensor([[0,1], [2,3]])) 
               for _ in range(100)]
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    
    # Models
    dense_model = GNNCaptioner()
    pruner = DynamicGraphPruner()
    
    # Benchmark
    print("=== Dense Graph Benchmark ===")
    dense_metrics = EfficiencyBenchmark.run(dense_model, loader)
    
    print("\n=== Pruned Graph Benchmark ===")
    sparse_metrics = EfficiencyBenchmark.run(dense_model, loader, pruner)
    
    # Results
    print("\n=== Results ===")
    print(f"Dense Graph:")
    print(f"- Avg inference time: {np.mean(dense_metrics['inference_time'])*1000:.2f}ms")
    
    print(f"\nPruned Graph:")
    print(f"- Avg edges/graph: {np.mean(sparse_metrics['edge_counts']):.1f} (orig: 90)")
    print(f"- GT retention: {np.mean(sparse_metrics['retention_rates'])*100:.1f}%")
    print(f"- Avg inference time: {np.mean(sparse_metrics['inference_time'])*1000:.2f}ms")
    print(f"- Speedup: {np.mean(dense_metrics['inference_time'])/np.mean(sparse_metrics['inference_time']):.1f}x")
    
    # Training Loop Example
    optimizer = optim.Adam(dense_model.parameters(), lr=0.001)
    for epoch in range(NUM_EPOCHS):
        for data in loader:
            data = pruner.prune(data)  # Dynamic pruning
            optimizer.zero_grad()
            outputs = dense_model(data)
            loss = F.cross_entropy(outputs, torch.randint(0, 1000, (len(data.x),)))  # Mock loss
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
