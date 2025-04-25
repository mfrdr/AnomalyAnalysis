import torch
import torch.nn as nn
from graphino.readout_MLP import ONI_MLP  # Module for the final prediction step
from graphino.GCN.graph_conv_layer import GraphConvolution  # Custom GCN layer
from graphino.structure_learner import EdgeStructureLearner  # Learns graph structure if not given
from utilities.utils import get_activation_function  # Utility function for activation functions

class GCN_2(nn.Module):
    """
    Graph Convolutional Network (GCN) implementation.
    This model learns node embeddings and makes predictions based on graph data.
    """

    def __init__(self, net_params, static_feat_atm=None, static_feat_oc=None, adj_atm=None, adj_oc=None, device='cuda', outsize=1, verbose=True):
        """
        Initialize the GCN model.

        Parameters:
        - net_params: Dictionary with model hyperparameters.
        - static_feat: Optional node features provided beforehand.
        - adj: Adjacency matrix (defines graph structure). If None, it will be learned.
        - device: Device to run the model on ('cpu' or 'cuda').
        - outsize: Output size for predictions (e.g., number of target classes).
        - verbose: If True, prints the layers created for debugging.
        """
        super().__init__()
        # Number of GCN layers
        self.L = net_params['L']
        assert self.L > 1, "The model must have at least 2 layers."

        # Activation function for the model
        self.act = net_params['activation']
        
        # Dimensionality of the output embeddings from the GCN layers
        self.out_dim = self.mlp_input_dim = net_params['out_dim'] 
        
        # Whether to use batch normalization after GCN layers
        self.batch_norm = net_params['batch_norm']
        
        # How to summarize the graph (e.g., mean, sum, max)
        self.graph_pooling = net_params['readout'].lower()
        
        # Whether to use Jumping Knowledge (JK) for richer embeddings
        self.jumping_knowledge = net_params['jumping_knowledge']
        
        # Dropout rate for regularization
        dropout = net_params['dropout']
        
        # Hidden layer dimension for the GCN layers
        hid_dim = net_params['hidden_dim']
        
        # Number of nodes in the graph
        num_nodes = net_params['num_nodes']

        # Get activation function (e.g., ReLU, Tanh)
        activation = get_activation_function(self.act, functional=True, num=1, device=device)

        # Common parameters for all GCN layers
        conv_kwargs = {'activation': activation, 'batch_norm': self.batch_norm,
                       'residual': net_params['residual'], 'dropout': dropout}

        # ATMOSPHERIC GCN
        print("Atmospheric GCN")
        # Create GCN layers
        layers_atm = [GraphConvolution(net_params['in_dim_atm'], hid_dim, **conv_kwargs)]  # Input layer
        layers_atm += [GraphConvolution(hid_dim, hid_dim, **conv_kwargs) for _ in range(self.L - 2)]  # Hidden layers
        layers_atm.append(GraphConvolution(hid_dim, self.out_dim, **conv_kwargs))  # Output layer
        self.layers_atm = nn.ModuleList(layers_atm)  # Store the layers as a ModuleList

        # OCEANIC GCN
        print("Oceanic GCN")
        # Create GCN layers
        layers_oc = [GraphConvolution(net_params['in_dim_oc'], hid_dim, **conv_kwargs)]  # Input layer
        layers_oc += [GraphConvolution(hid_dim, hid_dim, **conv_kwargs) for _ in range(self.L - 2)]  # Hidden layers
        layers_oc.append(GraphConvolution(hid_dim, self.out_dim, **conv_kwargs))  # Output layer
        self.layers_oc = nn.ModuleList(layers_oc)  # Store the layers as a ModuleList

        # Adjust input size for the MLP if Jumping Knowledge or pooling is used
        if self.jumping_knowledge:
            self.mlp_input_dim += hid_dim * (self.L - 1)  # Add dimensions from intermediate layers
        if self.graph_pooling in ['sum+mean', 'mean+sum', 'mean&sum', 'sum&mean']:
            self.mlp_input_dim *= 2  # Double input size for dual pooling

        # Final MLP layer to make predictions from the graph embeddings
        self.MLP_layer = ONI_MLP(self.mlp_input_dim * 2, outsize, act_func=self.act, 
                                 batch_norm=net_params['mlp_batch_norm'], dropout=dropout, device=device) # *2 for atm and oc

        # ATMOSPHERIC ADJ
        # Handle adjacency matrix (graph structure)
        if adj_atm is None:
            # Learn the adjacency matrix if not provided
            print("Creating a new connectivity structure!!!")
            self.adj_atm, self.learn_adj = None, True
            max_num_edges = int(net_params['avg_edges_per_node'] * num_nodes)
            print(num_nodes, max_num_edges)
            self.graph_learner_atm = EdgeStructureLearner(
                num_nodes, max_num_edges, dim=net_params['adj_dim'], device=device, static_feat=static_feat_atm,
                alpha1=net_params['tanh_alpha'], alpha2=net_params['sig_alpha'], self_loops=net_params['self_loop']
            )
            print("Created")
        else:
            # Use a pre-defined adjacency matrix
            print('Using a static connectivity structure !!!')
            self.adj_atm, self.learn_adj = torch.from_numpy(adj_atm).float().to(device), False

        # Print the layers if verbose is enabled
        if verbose:
            print([x for x in self.layers_atm])
        
        # OCEANIC ADJ
        # Handle adjacency matrix (graph structure)
        if adj_oc is None:
            # Learn the adjacency matrix if not provided
            print("Creating a new connectivity structure!!!")
            self.adj_oc, self.learn_adj = None, True
            max_num_edges = int(net_params['avg_edges_per_node'] * num_nodes)
            print(num_nodes, max_num_edges)
            self.graph_learner_oc = EdgeStructureLearner(
                num_nodes, max_num_edges, dim=net_params['adj_dim'], device=device, static_feat=static_feat_oc,
                alpha1=net_params['tanh_alpha'], alpha2=net_params['sig_alpha'], self_loops=net_params['self_loop']
            )
            print("Created")
        else:
            # Use a pre-defined adjacency matrix
            print('Using a static connectivity structure !!!')
            self.adj_oc, self.learn_adj = torch.from_numpy(adj_oc).float().to(device), False

        # Print the layers if verbose is enabled
        if verbose:
            print([x for x in self.layers_oc])


    def get_adj(self):
        """
        Return the adjacency matrix (learned or provided).
        """
        if self.learn_adj:
            return self.graph_learner_atm.forward(), self.graph_learner_oc.forward()  # Generate the adjacency matrix dynamically
        return self.adj_atm, self.adj_oc

    def forward(self, input_atm, input_oc, readout=True):
        """
        Forward pass through the GCN model.

        Parameters:
        - input: Node features (e.g., numerical data describing each node).
        - readout: Whether to perform graph-level pooling for final predictions.

        Returns:
        - Output of the model (graph embeddings or final predictions).
        """
        if self.learn_adj:
            # Learn the adjacency matrix if not provided
            self.adj_atm = self.graph_learner_atm.forward()
            self.adj_oc = self.graph_learner_oc.forward()

        # ATMOSPHERIC GCN
        # Pass input through the first GCN layer
        # print(f"Input ATM Shape before GCN: {input_atm.shape}, Adj ATM Shape: {self.adj_atm.shape}")
        node_embs = self.layers_atm[0](input_atm, self.adj_atm)  # Shape: (batch_size, num_nodes, hidden_dim)
        X_all_embeddings = node_embs.clone()  # Clone for Jumping Knowledge

        # Pass through subsequent GCN layers
        for conv in self.layers_atm[1:]:
            node_embs = conv(node_embs, self.adj_atm)  # Update node embeddings
            if self.jumping_knowledge:
                # Concatenate embeddings from all layers for richer representations
                X_all_embeddings = torch.cat((X_all_embeddings, node_embs), dim=2)
        
        # Final node embeddings (with or without Jumping Knowledge)
        final_embs_atm = X_all_embeddings if self.jumping_knowledge else node_embs

        # OCEANIC GCN
        # Pass input through the first GCN layer
        node_embs = self.layers_oc[0](input_oc, self.adj_oc)  # Shape: (batch_size, num_nodes, hidden_dim)
        X_all_embeddings = node_embs.clone()  # Clone for Jumping Knowledge

        # Pass through subsequent GCN layers
        for conv in self.layers_oc[1:]:
            node_embs = conv(node_embs, self.adj_oc)  # Update node embeddings
            if self.jumping_knowledge:
                # Concatenate embeddings from all layers for richer representations
                X_all_embeddings = torch.cat((X_all_embeddings, node_embs), dim=2)
        
        # Final node embeddings (with or without Jumping Knowledge)
        final_embs_oc = X_all_embeddings if self.jumping_knowledge else node_embs

        # Perform graph-level pooling (summarize node embeddings into a graph embedding)
        if self.graph_pooling == 'sum':
            g_emb_atm = torch.sum(final_embs_atm, dim=1)  # Sum pooling
            g_emb_oc = torch.sum(final_embs_oc, dim=1)  # Sum pooling
        elif self.graph_pooling == 'mean':
            g_emb_atm = torch.mean(final_embs_atm, dim=1)  # Mean pooling
            g_emb_oc = torch.mean(final_embs_oc, dim=1)  # Mean pooling
        elif self.graph_pooling == 'max':
            g_emb_atm, _ = torch.max(final_embs_atm, dim=1)  # Max pooling
            g_emb_oc, _ = torch.max(final_embs_oc, dim=1)  # Max pooling
        elif self.graph_pooling in ['sum+mean', 'mean+sum', 'mean&sum', 'sum&mean']:
            # Combine sum and mean pooling
            xmean = torch.mean(final_embs_atm, dim=1)
            xsum = torch.sum(final_embs_atm, dim=1)  # (batch_size, out_dim)
            g_emb_atm = torch.cat((xmean, xsum), dim=1)  # Combine results into a single embedding

            xmean = torch.mean(final_embs_oc, dim=1)
            xsum = torch.sum(final_embs_oc, dim=1)  # (batch_size, out_dim)
            g_emb_oc = torch.cat((xmean, xsum), dim=1)  # Combine results into a single embedding
        else:
            raise ValueError('Unsupported readout operation')

        # If readout is True, make predictions from the graph embedding
        out = self.graph_embedding_to_pred(g_emb_atm=g_emb_atm, g_emb_oc=g_emb_oc) if readout else (g_emb_atm, g_emb_oc)
        return out

    def graph_embedding_to_pred(self, g_emb_atm, g_emb_oc):
        """
        Convert graph embedding into a prediction using the MLP.
        """
        g_emb = torch.cat((g_emb_atm, g_emb_oc), dim=1)
        out = self.MLP_layer.forward(g_emb).squeeze(1)  # Squeeze to remove extra dimensions
        return out
