import numpy as np
import networkx as nx
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List
import random

class GraphType(Enum):
    PATH_COMMUNITY = "path_community"
    CIRCULANT_COMMUNITY = "circulant_community"
    APPROX_CIRCULANT = "approx_circulant"
    LOCAL_BANDED = "local_banded"
    HIERARCHICAL = "hierarchical"  # Bonus type combining others

@dataclass
class GraphParams:
    n: int  # Number of nodes
    graph_type: GraphType
    # Community parameters
    num_communities: Optional[int] = None
    inter_community_prob: Optional[float] = None
    intra_community_prob: Optional[float] = None
    # Circulant parameters
    k_neighbors: Optional[int] = None
    perturbation_prob: Optional[float] = None
    # Local connectivity parameters
    min_connections: Optional[int] = None
    distance_decay: Optional[float] = None
    # General parameters
    periodic: bool = True
    allow_overlapping: bool = False

class LaplacianGenerator:
    def __init__(self):
        self.supported_types = {
            GraphType.PATH_COMMUNITY: self._generate_path_community,
            GraphType.CIRCULANT_COMMUNITY: self._generate_circulant_community,
            GraphType.APPROX_CIRCULANT: self._generate_approx_circulant,
            GraphType.LOCAL_BANDED: self._generate_local_banded,
            GraphType.HIERARCHICAL: self._generate_hierarchical
        }

    def generate_random_params(self, min_nodes=100, max_nodes=1000) -> GraphParams:
        """Generate random parameters for graph generation"""
        n = random.randint(min_nodes, max_nodes)
        graph_type = random.choice(list(GraphType))
        
        params = GraphParams(
            n=n,
            graph_type=graph_type,
            # Community parameters
            num_communities=random.randint(2, int(np.sqrt(n))) if random.random() < 0.7 else None,
            inter_community_prob=random.uniform(0.01, 0.3) if random.random() < 0.7 else None,
            intra_community_prob=random.uniform(0.01, 0.2) if random.random() < 0.7 else None,
            # Circulant parameters
            k_neighbors=random.randint(1, int(np.log2(n))) if random.random() < 0.7 else None,
            perturbation_prob=random.uniform(0, 0.2) if random.random() < 0.7 else None,
            # Local connectivity parameters
            min_connections=random.randint(1, 3) if random.random() < 0.7 else None,
            distance_decay=random.uniform(0.1, 2.0) if random.random() < 0.7 else None,
            # General parameters
            periodic=random.choice([True, False]),
            allow_overlapping=random.choice([True, False])
        )
        return params

    def generate(self, params: GraphParams) -> Tuple[np.ndarray, nx.Graph]:
        """Generate Laplacian matrix and corresponding graph based on parameters"""
        generator_func = self.supported_types[params.graph_type]
        G = generator_func(params)
        
        # make graph connected
        if not nx.is_connected(G):
            self._make_connectivity(G)
            
        # Compute Laplacian
        L = nx.laplacian_matrix(G).toarray()
        return L, G

    def _generate_path_community(self, params: GraphParams) -> nx.Graph:
        """Generate path-based community graph"""
        G = nx.Graph()
        community_size = params.n // params.num_communities
        
        # Create path graphs for each community
        for c in range(params.num_communities):
            start = c * community_size
            path = nx.path_graph(range(start, start + community_size))
            G.add_edges_from(path.edges())
        
        # Add inter-community connections
        for i in range(params.num_communities):
            for j in range(i + 1, params.num_communities):
                start_i = i * community_size
                start_j = j * community_size
                for a in range(community_size):
                    if np.random.random() < params.inter_community_prob:
                        G.add_edge(start_i + a, start_j + a)
        
        return G

    def _generate_circulant_community(self, params: GraphParams) -> nx.Graph:
        """Generate circulant community graph"""
        G = nx.Graph()
        community_size = params.n // params.num_communities
        
        # Create circulant structure within communities
        for c in range(params.num_communities):
            start = c * community_size
            for i in range(start, start + community_size):
                for j in range(1, params.k_neighbors + 1):
                    node1 = i
                    node2 = start + ((i - start + j) % community_size)
                    G.add_edge(node1, node2)
                    
        # Add inter-community connections with varying patterns
        for i in range(params.num_communities):
            for j in range(i + 1, params.num_communities):
                start_i = i * community_size
                start_j = j * community_size
                for a in range(community_size):
                    if np.random.random() < params.inter_community_prob:
                        # Mix of diagonal and straight connections
                        if np.random.random() < 0.5:
                            G.add_edge(start_i + a, start_j + (community_size - a - 1))
                        else:
                            G.add_edge(start_i + a, start_j + a)
        
        return G

    def _generate_approx_circulant(self, params: GraphParams) -> nx.Graph:
        """Generate approximately circulant graph"""
        G = nx.Graph()
        G.add_nodes_from(range(params.n))
        
        # Add base circulant structure
        for i in range(params.n):
            for j in range(1, params.k_neighbors + 1):
                if params.periodic:
                    G.add_edge(i, (i + j) % params.n)
                    G.add_edge(i, (i - j) % params.n)
                else:
                    if i + j < params.n:
                        G.add_edge(i, i + j)
                    if i - j >= 0:
                        G.add_edge(i, i - j)
        
        # Add perturbations
        if params.perturbation_prob:
            for i in range(params.n):
                for j in range(i + 1, params.n):
                    if np.random.random() < params.perturbation_prob:
                        if G.has_edge(i, j):
                            G.remove_edge(i, j)
                        else:
                            G.add_edge(i, j)
        
        return G

    def _generate_local_banded(self, params: GraphParams) -> nx.Graph:
        """Generate locally connected graph with natural banding"""
        G = nx.Graph()
        G.add_nodes_from(range(params.n))
        
        # Use distance-based probability for connections
        for i in range(params.n):
            # make minimum connections
            connected = 0
            while connected < params.min_connections:
                # Sample potential neighbors based on distance
                for j in range(params.n):
                    if i != j:
                        distance = min(abs(i - j), params.n - abs(i - j)) if params.periodic else abs(i - j)
                        prob = np.exp(-distance * params.distance_decay)
                        if np.random.random() < prob:
                            G.add_edge(i, j)
                            connected += 1
                            if connected >= params.min_connections:
                                break
        
        return G

    def _generate_hierarchical(self, params: GraphParams) -> nx.Graph:
        """Generate hierarchical graph combining multiple types"""
        # Start with a base graph (e.g., path communities)
        base_params = GraphParams(
            n=params.n,
            graph_type=GraphType.PATH_COMMUNITY,
            num_communities=params.num_communities,
            inter_community_prob=params.inter_community_prob
        )
        G = self._generate_path_community(base_params)
        
        # Add circulant structure within communities
        if params.k_neighbors:
            community_size = params.n // params.num_communities
            for c in range(params.num_communities):
                start = c * community_size
                for i in range(start, start + community_size):
                    for j in range(1, params.k_neighbors + 1):
                        node2 = start + ((i - start + j) % community_size)
                        if np.random.random() < 0.5:  # Random mixing of structures
                            G.add_edge(i, node2)
        
        return G

    def _make_connectivity(self, G: nx.Graph):
        """make graph is connected by adding minimum necessary edges"""
        components = list(nx.connected_components(G))
        while len(components) > 1:
            c1 = random.choice(list(components[0]))
            c2 = random.choice(list(components[1]))
            G.add_edge(c1, c2)
            components = list(nx.connected_components(G))

def generate_training_data(num_samples: int, min_nodes: int = 100, max_nodes: int = 1000) -> List[Tuple[np.ndarray, GraphParams]]:
    """Generate diverse training data with varied parameters"""
    generator = LaplacianGenerator()
    samples = []
    
    for _ in range(num_samples):
        params = generator.generate_random_params(min_nodes, max_nodes)
        L, _ = generator.generate(params)
        samples.append((L, params))
    
    return samples


def visualize_matrices(samples, num_display=5):
    """Visualize random selection of generated Laplacian matrices"""
    if len(samples) < num_display:
        num_display = len(samples)
    
    # Randomly select samples
    display_samples = random.sample(samples, num_display)
    
    # Create subplot grid
    fig, axes = plt.subplots(1, num_display, figsize=(20, 4))
    if num_display == 1:
        axes = [axes]
    
    # Custom colormap
    cmap = mcolors.LinearSegmentedColormap.from_list('custom', 
        ['blue', 'black', 'red'], N=256)
    
    for idx, (ax, (L, params)) in enumerate(zip(axes, display_samples)):
        # Normalize matrix values
        vmax = max(abs(L.min()), abs(L.max()))
        
        # Plot matrix
        im = ax.imshow(L, cmap=cmap, aspect='equal', 
                      vmin=-vmax, vmax=vmax)
        
        # Add title with parameters
        title = f"Type: {params.graph_type.value}\n"
        title += f"n={params.n}"
        if params.num_communities:
            title += f"\ncomm={params.num_communities}"
        ax.set_title(title, size=8)
        ax.axis('off')
    
    # Add colorbar
    fig.colorbar(im, ax=axes, location='right', shrink=0.8)
    plt.tight_layout()
    plt.show()
    
    # Print detailed parameters
    print("\nDetailed parameters for displayed matrices:")
    for idx, (L, params) in enumerate(display_samples):
        print(f"\nMatrix {idx+1}:")
        print(f"  Type: {params.graph_type.value}")
        print(f"  Nodes: {params.n}")
        if params.num_communities:
            print(f"  Communities: {params.num_communities}")
        if params.k_neighbors:
            print(f"  Neighbors: {params.k_neighbors}")
        if params.inter_community_prob:
            print(f"  Inter-comm prob: {params.inter_community_prob:.3f}")
        if params.intra_community_prob:
            print(f"  Intra-comm prob: {params.intra_community_prob:.3f}")
        if params.perturbation_prob:
            print(f"  Perturbation prob: {params.perturbation_prob:.3f}")
        if params.distance_decay:
            print(f"  Distance decay: {params.distance_decay:.3f}")

def main():
    # Generate sample dataset
    print("Generating Laplacian matrices...")
    samples = generate_training_data(
        num_samples=1000,
        min_nodes=20,
        max_nodes=500
    )
    
    # Visualize random selection
    print("\nVisualizing 5 random samples:")
    visualize_matrices(samples, num_display=5)
    
    # Print summary statistics
    print("\nGenerated dataset summary:")
    type_counts = {}
    for _, params in samples:
        type_counts[params.graph_type.value] = type_counts.get(params.graph_type.value, 0) + 1
    
    print("\nGraph type distribution:")
    for type_name, count in type_counts.items():
        print(f"  {type_name}: {count}")

if __name__ == "__main__":
    main()
