import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data,Batch


import torch

def extract_3d_patches(feature_map, patch_size=(4, 4, 4)):
    """
    Extracts non-overlapping 3D patches from a feature map to form nodes in a graph.
    Parameters:
    - feature_map (Tensor): The feature map from the CNN of shape (B, C, D, H, W).
      D, H, and W are the depth, height, and width of the feature map after CNN feature extraction.
    - patch_size (tuple): Spatial size (depth, height, width) of each patch.

    Returns:
    - patches (Tensor): Tensor of shape (B, N, C, patch_d, patch_h, patch_w), where N is the number of patches per image.
    """
    b, c, d, h, w = feature_map.size()
    patch_d, patch_h, patch_w = patch_size

    # Unfold extracts sliding patches; here we align so that they are non-overlapping
    patches = feature_map.unfold(2, patch_d, patch_d).unfold(3, patch_h, patch_h).unfold(4, patch_w, patch_w)

    # Rearrange to have patches as separate units
    patches = patches.permute(0, 2, 3, 4, 1, 5, 6,7).contiguous()
    patches = patches.view(b, -1, c, patch_d, patch_h, patch_w)
    return patches



def compute_correlation(node_features, i, j):
    """
    Computes the correlation coefficient between two nodes.
    Parameters:
    - node_features (Tensor): Tensor of shape (N, C * patch_h * patch_w) containing all node features.
    - i (int): Index of the first node.
    - j (int): Index of the second node.

    Returns:
    - correlation (float): Correlation coefficient between the two nodes.
    """
    x = node_features[i]
    y = node_features[j]
    correlation = torch.corrcoef(torch.stack([x, y]))[0, 1]
    return correlation.item()


def construct_graph_from_3d_patch(patch_index, patch_shape, image_shape, node_features):
    """
    Constructs edges between 3D patch nodes based on spatial adjacency (k-connectivity).
    This follows the approach described in Section 3.2 of SAG-ViT, where patches
    are arranged in a 3D grid and connected to their spatial neighbors.

    Parameters:
    - patch_index (int): Index of the current patch node.
    - patch_shape (tuple): (patch_depth, patch_height, patch_width).
    - image_shape (tuple): (depth, height, width) of the feature map.

    Returns:
    - G (nx.Graph): A graph with a single node and edges to its neighbors (to be composed globally).
    """
    G = nx.Graph()

    # Compute grid dimensions (how many patches along depth, height, and width)
    grid_depth = image_shape[0] // patch_shape[0]
    grid_height = image_shape[1] // patch_shape[1]
    grid_width = image_shape[2] // patch_shape[2]

    # Current node index in a flattened grid
    current_node = patch_index
    G.add_node(current_node)

    # 26-neighborhood connectivity (all neighbors in 3D: 6 face neighbors, 12 edge neighbors, and 8 corner neighbors)
    neighbor_offsets = [
        (-1, 0, 0),  (1, 0, 0),  # Depth neighbors
        (0, -1, 0),  (0, 1, 0),  # Height neighbors
        (0, 0, -1),  (0, 0, 1),  # Width neighbors
        (-1, -1, 0), (-1, 1, 0), (1, -1, 0), (1, 1, 0),  # Diagonal neighbors in 2D planes
        (-1, 0, -1), (-1, 0, 1), (1, 0, -1), (1, 0, 1),
        (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1),
        (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
        (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)  # Corner neighbors
    ]

    # Recover depth, height, width from patch_index
    depth = current_node // (grid_height * grid_width)
    row = (current_node % (grid_height * grid_width)) // grid_width
    col = current_node % grid_width

    for dr, dh, dw in neighbor_offsets:
        neighbor_depth = depth + dr
        neighbor_row = row + dh
        neighbor_col = col + dw
        # Check if the neighbor is within bounds
        if 0 <= neighbor_depth < grid_depth and 0 <= neighbor_row < grid_height and 0 <= neighbor_col < grid_width:
            neighbor_node = neighbor_depth * (grid_height * grid_width) + neighbor_row * grid_width + neighbor_col
            correlation = compute_correlation(node_features, current_node, neighbor_node)
            G.add_edge(current_node, neighbor_node, weight=correlation)

    return G


def build_graph_from_patches(feature_map, patch_size=(4, 4, 4), is_3d=False):
    """
    Builds a global graph for each image in the batch, where each node corresponds
    to a patch, and edges represent spatial adjacency. This graph captures local
    spatial relationships of the patches.

    Parameters:
    - feature_map (Tensor): CNN output (B, C, D, H, W) or (B, C, H', W') for 3D or 2D.
    - patch_size (tuple): Size of each patch (patch_d, patch_h, patch_w) for 3D or (patch_h, patch_w) for 2D.
    - is_3d (bool): Flag indicating whether the data is 3D or 2D.

    Returns:
    - G_global_batch (list): A list of NetworkX graphs, one per image in the batch.
    - patches (Tensor): The extracted patches.
    """

    patches = extract_3d_patches(feature_map, patch_size)
    batch_size = patches.size(0)

    # grid_depth = feature_map.size(2) // patch_size[0]
    # grid_height = feature_map.size(3) // patch_size[1]
    # grid_width = feature_map.size(4) // patch_size[2]
    # num_patches = grid_height * grid_width * grid_depth

    G_global_batch = []
    for batch_idx in range(batch_size):
        G_global = nx.Graph()
        node_features = patches[batch_idx].view(patches.size(1), -1)  # Flattening each patch

        for patch_idx in range(patches.size(1)):  # Loop through all patches
            G_patch = construct_graph_from_3d_patch(
                patch_index=patch_idx,
                patch_shape=patch_size,
                image_shape=(feature_map.size(2), feature_map.size(3), feature_map.size(4)),
                node_features=node_features
            )

            G_global = nx.compose(G_global, G_patch)

        G_global_batch.append(G_global)

    return G_global_batch, patches

def build_graph_data_from_patches(G_global_batch, patches):
    """
    Converts NetworkX graphs and associated patches into PyTorch Geometric Data objects.
    Each node corresponds to a patch vectorized into a feature node embedding.

    Parameters:
    - G_global_batch (list): List of global graphs (one per image) in NetworkX form.
    - patches (Tensor): (B, N, C, patch_h, patch_w) patch tensor.

    Returns:
    - data_list (list): List of PyTorch Geometric Data objects, where data.x are node features,
      and data.edge_index is the adjacency from the constructed graph.
    """
    data_list = []
    batch_size, num_patches, channels,patch_d, patch_h, patch_w, = patches.size()

    for batch_idx, G_global in enumerate(G_global_batch):
        # 提取节点特征
        node_features = patches[batch_idx].view(num_patches, -1)
        batch = torch.ones(node_features.size(0), dtype=torch.long) * batch_idx  # 所有节点来自一张图
        # 提取边索引和边属性
        edge_index = torch.tensor(list(G_global.edges)).t().contiguous()
        edge_attr = torch.tensor(
            [G_global[u][v]['weight'] for u, v in G_global.edges],
            dtype=torch.float
        ).reshape(-1, 1)

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        data_list.append(data)

    return data_list


def build_graph_from_img_3d(img):
    G_global_batch, patches = build_graph_from_patches(img)
    data_list = build_graph_data_from_patches(G_global_batch, patches)
    data = Batch.from_data_list(data_list)

    return data

if __name__ == '__main__':
    x = torch.rand(2,1,40,40,24)
    # G_global_batch, patches = build_graph_from_patches(x)
    data = build_graph_from_img_3d(x)
    print(data)