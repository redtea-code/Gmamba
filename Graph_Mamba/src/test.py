from Graph_Mamba.src._shared_imports import *
from Graph_Mamba.src.dataset import compute_nmf_embeddings
from Graph_Mamba.src.main import parse_arguments
from Graph_Mamba.src.graph import compute_WL_graph_embeddings, create_gcondnet_graph_dataset, compute_graph_embeddings, \
    create_edges_graphs_sparse_relative_distance, load_random_graph, create_edges_graphs_knn
from Graph_Mamba.src.graph import GCNEncoder
import torch_geometric as tg
from torch_geometric.nn.models import GAE

from Graph_Mamba.src.models import compute_kaiming_normal_std, FirstLinearLayer, DNN


def get_embedding_matrix(X_train_raw, embedding_size):
    """
    Return matrix D x M

    Use a the shared hyper-parameter self.args.embedding_preprocessing.
    """

    # Preprocess the data for the embeddings
    X_for_embeddings = MinMaxScaler().fit_transform(X_train_raw)

    emd = compute_nmf_embeddings(X_for_embeddings, rank=embedding_size)
    emd_data = emd.data
    emd_data.requires_grad = False
    return emd_data.type(torch.float32)


def build_graph_from_table(X):
    list_of_edges = create_edges_graphs_sparse_relative_distance(X)
    graphs_dataset_all = create_gcondnet_graph_dataset(
        X,
        list_of_edges=list_of_edges
    )
    return graphs_dataset_all


class GCondNet(torch.nn.Module):
    def __init__(self, X, graphs_dataset_all,dim=None,layer=None):
        super().__init__()

        args = parse_arguments()
        args.num_features = X.shape[1]

        decoder = None
        std_kaiming_first_layer = compute_kaiming_normal_std(
            torch.zeros(args.feature_extractor_dims[0], args.num_features))

        in_channels = graphs_dataset_all.data[0].x.shape[1]
        out_channels = args.winit_graph_embedding_size
        gnn = GAE(GCNEncoder(in_channels, out_channels, dropout_rate=0.5)).cuda()
        # put all graphs into one batch for easy forward passing through the GNN
        gnn_dataloader = tg.loader.DataLoader(graphs_dataset_all, batch_size=len(graphs_dataset_all),
                                              shuffle=False)
        gnn_batch_train = next(iter(gnn_dataloader))
        gnn_batch_train.edge_index = tg.utils.coalesce(gnn_batch_train.edge_index, is_sorted=False)

        first_layer = FirstLinearLayer(args,
                                       layer_type='diet_gnn',
                                       alpha_interpolation=args.winit_first_layer_interpolation,
                                       sparsity_type=args.sparsity_type,
                                       # kwargs arguments
                                       gnn_model=gnn, gnn_batch=gnn_batch_train,
                                       std_kaiming_normal=std_kaiming_first_layer)

        self.model = DNN(args, first_layer, decoder=decoder,dim=dim,layer=layer)

    def forward(self, x):
        x = self.model(x)

        return x


if __name__ == '__main__':
    X = torch.rand(4,33)
    graphs_dataset_all = build_graph_from_table(X)
    model = GCondNet(X, graphs_dataset_all).cuda()
    y = model(X.cuda())
    print(y.shape)
