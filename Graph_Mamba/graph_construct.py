import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data,Batch
####################################################################
# These functions reflect the methods described in Section 3.1 and 3.2
# of the SAG-ViT paper, where high-fidelity feature patches are extracted
# from the CNN feature maps and organized into a graph structure.
####################################################################


def extract_patches(feature_map, patch_size=(4, 4)):
    """
    Extracts non-overlapping patches from a feature map to form nodes in a graph.
    Parameters:
    - feature_map (Tensor): The feature map from the CNN of shape (B, C, H', W').
      H' and W' are reduced spatial dimensions after CNN feature extraction.
    - patch_size (tuple): Spatial size (height, width) of each patch.

    Returns:
    - patches (Tensor): Tensor of shape (B, N, C, patch_h, patch_w), where N is the number of patches per image.
    """
    b, c, h, w = feature_map.size()
    patch_h, patch_w = patch_size

    # Unfold extracts sliding patches; here we align so that they are non-overlapping
    patches = feature_map.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)

    # Rearrange to have patches as separate units
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(b, -1, c, patch_h, patch_w)
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


def construct_graph_from_patch(patch_index, patch_shape, image_shape, node_features):
    """
    Constructs edges between patch nodes based on spatial adjacency (k-connectivity).
    This follows the approach described in Section 3.2 of SAG-ViT, where patches
    are arranged in a grid and connected to their spatial neighbors.

    Parameters:
    - patch_index (int): Index of the current patch node.
    - patch_shape (tuple): (patch_height, patch_width).
    - image_shape (tuple): (height, width) of the feature map.

    Returns:
    - G (nx.Graph): A graph with a single node and edges to its neighbors (to be composed globally).
    """
    G = nx.Graph()

    # Compute grid dimensions (how many patches along height and width)
    grid_height = image_shape[0] // patch_shape[0]
    grid_width = image_shape[1] // patch_shape[1]

    # Current node index in a flattened grid
    current_node = patch_index

    G.add_node(current_node)

    # 8-neighborhood connectivity (up, down, left, right, diagonals)
    neighbor_offsets = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    # Recover row, col from patch_index
    row = current_node // grid_width
    col = current_node % grid_width

    for dr, dc in neighbor_offsets:
        neighbor_row = row + dr
        neighbor_col = col + dc
        if 0 <= neighbor_row < grid_height and 0 <= neighbor_col < grid_width:
            neighbor_node = neighbor_row * grid_width + neighbor_col
            correlation = compute_correlation(node_features, current_node, neighbor_node)
            G.add_edge(current_node, neighbor_node, weight=correlation)

    return G


def build_graph_from_patches(feature_map, patch_size=(4, 4)):
    """
    Builds a global graph for each image in the batch, where each node corresponds
    to a patch, and edges represent spatial adjacency. This graph captures local
    spatial relationships of the patches, as outlined in Sections 3.1 and 3.2 of SAG-ViT.

    Parameters:
    - feature_map (Tensor): CNN output (B, C, H', W').
    - patch_size (tuple): Size of each patch (patch_h, patch_w).

    Returns:
    - G_global_batch (list): A list of NetworkX graphs, one per image in the batch.
    - patches (Tensor): The extracted patches (B, N, C, patch_h, patch_w).
    """
    patches = extract_patches(feature_map, patch_size)
    batch_size = patches.size(0)

    grid_height = feature_map.size(2) // patch_size[0]
    grid_width = feature_map.size(3) // patch_size[1]
    num_patches = grid_height * grid_width

    G_global_batch = []
    for batch_idx in range(batch_size):
        G_global = nx.Graph()
        # Flatten each patch into a feature vector
        node_features = patches[batch_idx].view(num_patches, -1)
        # Construct a global graph by composing individual patch-based graphs
        for patch_idx in range(num_patches):
            G_patch = construct_graph_from_patch(
                patch_index=patch_idx,
                patch_shape=patch_size,
                image_shape=(feature_map.size(2), feature_map.size(3)),
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
    from_networkx_ = from_networkx  # local alias to avoid confusion

    data_list = []
    batch_size, num_patches, channels, patch_h, patch_w = patches.size()

    for batch_idx, G_global in enumerate(G_global_batch):
        # 提取节点特征
        node_features = patches[batch_idx].view(num_patches, -1)

        batch = torch.ones(node_features.size(0), dtype=torch.long) * batch_idx  # 所有节点来自一张图
        # 提取边索引和边属性
        edge_index = torch.tensor(list(G_global.edges)).t().contiguous()
        edge_attr = torch.tensor(
            [G_global[u][v]['weight'] for u, v in G_global.edges],
            dtype=torch.float
        ).reshape(-1,1) #实验（1,2,3）未修改

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr,batch=batch)
        data_list.append(data)

    return data_list

def build_graph_from_img(img):
    G_global_batch, patches = build_graph_from_patches(img)
    data_list = build_graph_data_from_patches(G_global_batch, patches)
    data = Batch.from_data_list(data_list)

    return data

if __name__ == '__main__':

    x = torch.rand(2, 108, 400, 12)
    # (h/4 * w/4), c* (4*4)
    # patches = extract_patches(x)
    # print(patches.shape)
    # G_global_batch, patches = build_graph_from_patches(x)
    # data = build_graph_data_from_patches(G_global_batch, patches)
    # print(data)
    data = build_graph_from_img(x)
    print(data)