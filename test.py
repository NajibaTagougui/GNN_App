import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import time
import matplotlib.pyplot as plt
import numpy as np

# ========================
# 1. Synthetic Data Setup
# ========================
def generate_synthetic_image_data(num_images=5, num_objects_per_image=5):
    """Creates mock image data with objects and relationships"""
    data_list = []
    for _ in range(num_images):
        # Random object features (pretrained CNN-like embeddings)
        x = torch.randn(num_objects_per_image, 64)
        
        # Random bounding boxes [x1, y1, x2, y2] normalized to [0,1]
        bboxes = torch.rand(num_objects_per_image, 4)
        bboxes[:, 2:] += bboxes[:, :2]  # Ensure x2 > x1, y2 > y1
        bboxes = torch.clamp(bboxes, 0, 1)
        
        # Create 2-3 ground-truth relationships per image
        gt_edges = set()
        for _ in range(2):
            i, j = np.random.choice(num_objects_per_image, 2, replace=False)
            gt_edges.add((i, j))
        
        # Fully connected graph (all possible edges)
        edge_index = []
        for i in range(num_objects_per_image):
            for j in range(num_objects_per_image):
                if i != j:
                    edge_index.append([i, j])
        
        data = Data(
            x=x,
            edge_index=torch.tensor(edge_index).t().contiguous(),
            bboxes=bboxes,
            gt_edges=list(gt_edges)
        )
        data_list.append(data)
    return data_list

# ========================
# 2. Dynamic Sparsification
# ========================
def spatial_score(bbox1, bbox2):
    """IoU-based edge relevance scoring"""
    # Calculate intersection
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    return inter_area / (area1 + area2 - inter_area + 1e-6)

def prune_edges(graph_data, threshold=0.3):
    """Dynamic edge pruning based on spatial scores"""
    edge_indices = []
    for i, j in graph_data.edge_index.t():
        score = spatial_score(graph_data.bboxes[i], graph_data.bboxes[j])
        if score > threshold:
            edge_indices.append([i, j])
    
    pruned_graph = Data(
        x=graph_data.x,
        edge_index=torch.tensor(edge_indices).t().contiguous(),
        bboxes=graph_data.bboxes,
        gt_edges=graph_data.gt_edges
    )
    return pruned_graph

# ========================
# 3. Tiny GNN Model
# ========================
class TinyGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(64, 32, heads=2)
    
    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        return x

# ========================
# 4. Evaluation Metrics
# ========================
def edge_reduction(original, pruned):
    return 1 - (pruned.edge_index.shape[1] / original.edge_index.shape[1])

def retention_rate(pruned_graph):
    kept_edges = set([tuple(e) for e in pruned_graph.edge_index.t().tolist()])
    gt_edges = set([tuple(e) for e in pruned_graph.gt_edges])
    return len(kept_edges & gt_edges) / len(gt_edges)

# ========================
# 5. Visualization
# ========================
def plot_graph(bboxes, edges, title):
    plt.figure(figsize=(5,5))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.gca().invert_yaxis()
    
    # Draw boxes
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        plt.gca().add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1, fill=False))
        plt.text(x1, y1, f"Obj{i}", fontsize=8)
    
    # Draw edges
    for (i,j) in edges:
        plt.plot(
            [bboxes[i][0]+(bboxes[i][2]-bboxes[i][0])/2, bboxes[j][0]+(bboxes[j][2]-bboxes[j][0])/2],
            [bboxes[i][1]+(bboxes[i][3]-bboxes[i][1])/2, bboxes[j][1]+(bboxes[j][3]-bboxes[j][1])/2],
            'r-', alpha=0.5
        )
    plt.title(title)
    plt.show()

# ========================
# 6. Main Experiment
# ========================
if __name__ == "__main__":
    # Generate synthetic data
    data = generate_synthetic_image_data(num_images=1)[0]
    
    # Before pruning
    print(f"Original edges: {data.edge_index.shape[1]}")
    plot_graph(data.bboxes, data.edge_index.t().tolist(), "Original Graph")
    
    # Prune edges
    pruned_data = prune_edges(data, threshold=0.2)
    print(f"Pruned edges: {pruned_data.edge_index.shape[1]}")
    print(f"Edge reduction: {edge_reduction(data, pruned_data):.1%}")
    print(f"GT retention: {retention_rate(pruned_data):.1%}")
    plot_graph(pruned_data.bboxes, pruned_data.edge_index.t().tolist(), "Pruned Graph")
    
    # Efficiency test
    model = TinyGNN()
    
    # Warmup
    for _ in range(3):
        _ = model(data)
        _ = model(pruned_data)
    
    # Benchmark
    start = time.time()
    _ = model(data)
    dense_time = time.time() - start
    
    start = time.time()
    _ = model(pruned_data)
    sparse_time = time.time() - start
    
    print(f"\nInference Time:")
    print(f"- Dense graph: {dense_time*1000:.2f}ms")
    print(f"- Pruned graph: {sparse_time*1000:.2f}ms")
    print(f"- Speedup: {dense_time/sparse_time:.1f}x")
