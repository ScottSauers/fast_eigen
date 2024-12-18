import numpy as np
import os
import networkx as nx
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
import pickle

class GraphType(Enum):
   PATH_COMMUNITY = "path_community"
   CIRCULANT_COMMUNITY = "circulant_community"
   APPROX_CIRCULANT = "approx_circulant"
   LOCAL_BANDED = "local_banded"
   HIERARCHICAL = "hierarchical"
   MULTI_SCALE = "multi_scale"
   DIFFUSION = "diffusion"

@dataclass
class GraphParams:
   n: int
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
   # Multi-scale parameters
   scale_widths: Optional[List[float]] = None
   scale_strengths: Optional[List[float]] = None
   # Diffusion parameters
   num_sources: Optional[int] = None
   diffusion_length: Optional[float] = None
   diffusion_strength: Optional[float] = None
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
           GraphType.HIERARCHICAL: self._generate_hierarchical,
           GraphType.MULTI_SCALE: self._generate_multi_scale,
           GraphType.DIFFUSION: self._generate_diffusion
       }

   def generate_random_params(self, min_nodes=100, max_nodes=1000) -> GraphParams:
       n = random.randint(min_nodes, max_nodes)
       graph_type = random.choice(list(GraphType))
       
       params = {
           'n': n,
           'graph_type': graph_type,
           'periodic': random.choice([True, False]),
           'allow_overlapping': random.choice([True, False])
       }
       
       if graph_type == GraphType.PATH_COMMUNITY:
           params.update({
               'num_communities': random.randint(2, int(np.sqrt(n))),
               'inter_community_prob': random.uniform(0.01, 0.3)
           })
       
       elif graph_type == GraphType.CIRCULANT_COMMUNITY:
           params.update({
               'num_communities': random.randint(2, int(np.sqrt(n))),
               'k_neighbors': random.randint(1, int(np.log2(n))),
               'inter_community_prob': random.uniform(0.01, 0.3),
               'intra_community_prob': random.uniform(0.01, 0.2)
           })
       
       elif graph_type == GraphType.APPROX_CIRCULANT:
           params.update({
               'k_neighbors': random.randint(1, int(np.log2(n))),
               'perturbation_prob': random.uniform(0, 0.2)
           })
       
       elif graph_type == GraphType.LOCAL_BANDED:
           params.update({
               'min_connections': random.randint(1, 3),
               'distance_decay': random.uniform(0.1, 2.0)
           })
       
       elif graph_type == GraphType.HIERARCHICAL:
           params.update({
               'num_communities': random.randint(2, int(np.sqrt(n))),
               'k_neighbors': random.randint(1, int(np.log2(n))),
               'inter_community_prob': random.uniform(0.01, 0.3)
           })
           
       elif graph_type == GraphType.MULTI_SCALE:
           num_scales = random.randint(2, 4)
           params.update({
               'scale_widths': [random.uniform(1, n/4) for _ in range(num_scales)],
               'scale_strengths': [random.uniform(0.1, 1.0) for _ in range(num_scales)]
           })
           
       elif graph_type == GraphType.DIFFUSION:
           params.update({
               'num_sources': random.randint(1, int(np.sqrt(n))),
               'diffusion_length': random.uniform(n/20, n/5),
               'diffusion_strength': random.uniform(0.1, 1.0)
           })
       
       return GraphParams(**params)

   def generate(self, params: GraphParams) -> Tuple[np.ndarray, nx.Graph]:
       # Set default parameters based on graph type
       if params.graph_type == GraphType.APPROX_CIRCULANT:
           params.k_neighbors = params.k_neighbors if params.k_neighbors is not None else 2
       elif params.graph_type == GraphType.CIRCULANT_COMMUNITY:
           params.k_neighbors = params.k_neighbors if params.k_neighbors is not None else 2
           params.num_communities = params.num_communities if params.num_communities is not None else 2
           params.inter_community_prob = params.inter_community_prob if params.inter_community_prob is not None else 0.1
       elif params.graph_type == GraphType.PATH_COMMUNITY:
           params.num_communities = params.num_communities if params.num_communities is not None else 2
           params.inter_community_prob = params.inter_community_prob if params.inter_community_prob is not None else 0.1
       elif params.graph_type == GraphType.LOCAL_BANDED:
           params.min_connections = params.min_connections if params.min_connections is not None else 1
           params.distance_decay = params.distance_decay if params.distance_decay is not None else 0.5
   
       # Try generating a valid graph with retries
       max_retries = 3
       for attempt in range(max_retries):
           generator_func = self.supported_types[params.graph_type]
           G = generator_func(params)
           
           # Check if graph is empty
           if len(G.nodes()) == 0:
               if attempt == max_retries - 1:
                   raise RuntimeError(f"Failed to generate non-null graph after {max_retries} attempts")
               continue
               
           # Make graph connected if needed
           if not nx.is_connected(G):
               self._make_connectivity(G)
   
           L = nx.laplacian_matrix(G).toarray()
           return L, G
    
   def _generate_path_community(self, params: GraphParams) -> nx.Graph:
       G = nx.Graph()
   
       node_positions = {i: random.uniform(0, 100) for i in range(params.n)}
       sorted_nodes = sorted(node_positions.keys(), key=lambda x: node_positions[x])
   
       # Generate variable-sized communities by partitioning the line
       remaining_nodes = params.n
       community_sizes = []
       min_size = 1
       max_size = max(min_size, params.n // params.num_communities)
   
       while remaining_nodes > 0:
           size = random.randint(min_size, min(max_size, remaining_nodes))
           community_sizes.append(size)
           remaining_nodes -= size
   
       # Create communities by grouping adjacent nodes
       communities = []
       idx = 0
       for size in community_sizes:
           community_nodes = sorted_nodes[idx:idx + size]
           communities.append(community_nodes)
           idx += size
   
       # Create local connections within communities
       for community_nodes in communities:
           size = len(community_nodes)
           # Connect nodes within the community based on proximity
           for i in range(size):
               for j in range(i + 1, size):
                   node_i = community_nodes[i]
                   node_j = community_nodes[j]
                   distance = abs(node_positions[node_i] - node_positions[node_j])
                   # Higher probability for closer nodes for bandedness
                   prob = np.exp(-distance / random.uniform(2.0, 5.0)) * random.uniform(0.7, 1.0)
                   if random.random() < prob:
                       G.add_edge(node_i, node_j)
   
       # Add inter-community connections with decreasing probability based on distance
       for idx in range(len(communities) - 1):
           community_a = communities[idx]
           community_b = communities[idx + 1]
   
           # Select a subset of nodes to attempt inter-community connections
           num_connections = random.randint(1, min(len(community_a), len(community_b)))
           nodes_a = random.sample(community_a, num_connections)
           nodes_b = random.sample(community_b, num_connections)
   
           for node_i, node_j in zip(nodes_a, nodes_b):
               distance = abs(node_positions[node_i] - node_positions[node_j])
               # Reduced probability for inter-community connections
               prob = params.inter_community_prob * np.exp(-distance / random.uniform(5.0, 15.0)) * random.uniform(0.5, 1.0)
               if random.random() < prob:
                   G.add_edge(node_i, node_j)
   
       # A few long-range connections
       num_long_range = random.randint(0, params.num_communities)
       for _ in range(num_long_range):
           node_i = random.choice(sorted_nodes)
           node_j = random.choice(sorted_nodes)
           if node_i != node_j:
               distance = abs(node_positions[node_i] - node_positions[node_j])
               # Very low probability for long-range connections
               prob = 0.01 * np.exp(-distance / random.uniform(20.0, 50.0))
               if random.random() < prob:
                   G.add_edge(node_i, node_j)
   
       return G

   def _generate_circulant_community(self, params: GraphParams) -> nx.Graph:   
       G = nx.Graph()
       n = params.n
       num_communities = params.num_communities
       community_size = n // num_communities
       node_offset = 0
       community_graphs = []
       
       for c in range(num_communities):
           # Create a community graph with organic structure
           # Using the Watts-Strogatz small-world model for intra-community connections
           # k is even and less than community size
           possible_k = [k for k in range(2, min(community_size, 20), 2)]  # Limit k to reasonable values
           if not possible_k:
               k = 2  # Default to 2 if community_size is very small
           else:
               k = np.random.choice(possible_k)
           p = np.random.uniform(0.1, 0.5)  # Rewiring probability for randomness
           
           # Generate the small-world community
           community = nx.watts_strogatz_graph(community_size, k, p)
           
           # Relabel nodes to have unique labels across communities
           mapping = {node: node + node_offset for node in community.nodes()}
           community = nx.relabel_nodes(community, mapping)
           
           G = nx.compose(G, community)
           
           community_graphs.append((community, mapping))
           node_offset += community_size
   
       # Add inter-community edges with organic patterns
       # Randomly select nodes to connect between communities
       inter_community_edges = max(1, int(params.inter_community_prob * n))
   
       for _ in range(inter_community_edges):
           # Select two different communities
           i, j = np.random.choice(num_communities, 2, replace=False)
           community_i_nodes = list(community_graphs[i][0].nodes())
           community_j_nodes = list(community_graphs[j][0].nodes())
           
           # Preferentially select nodes with higher degrees
           degrees_i = np.array([G.degree(node) for node in community_i_nodes])
           degrees_j = np.array([G.degree(node) for node in community_j_nodes])
           
           # Avoid division by zero by handling the case when sum of degrees is zero
           if degrees_i.sum() == 0:
               probs_i = np.ones_like(degrees_i) / len(degrees_i)
           else:
               probs_i = degrees_i / degrees_i.sum()
           
           if degrees_j.sum() == 0:
               probs_j = np.ones_like(degrees_j) / len(degrees_j)
           else:
               probs_j = degrees_j / degrees_j.sum()
           
           # Randomly select nodes based on degree probabilities
           u = np.random.choice(community_i_nodes, p=probs_i)
           v = np.random.choice(community_j_nodes, p=probs_j)
           G.add_edge(u, v)
       
       return G

   def _generate_approx_circulant(self, params: GraphParams) -> nx.Graph:
       G = nx.Graph()
       G.add_nodes_from(range(params.n))
       
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
       G = nx.Graph()
       G.add_nodes_from(range(params.n))
       
       for i in range(params.n):
           connected = 0
           while connected < params.min_connections:
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
       G = nx.Graph()
       G.add_nodes_from(range(params.n))
       
       # Create multiple levels of hierarchy
       num_levels = random.randint(2, 4)
       nodes_per_level = []
       remaining_nodes = params.n
       
       for level in range(num_levels - 1):
           size = remaining_nodes // (num_levels - level)
           nodes_per_level.append(size)
           remaining_nodes -= size
       nodes_per_level.append(remaining_nodes)
       
       # Generate connections at each level
       current_node = 0
       clusters = []
       
       for level, size in enumerate(nodes_per_level):
           level_clusters = []
           cluster_size = max(3, size // random.randint(3, 6))
           
           while current_node < sum(nodes_per_level[:level+1]):
               cluster_nodes = list(range(current_node, 
                                        min(current_node + cluster_size, 
                                            sum(nodes_per_level[:level+1]))))
               
               # Create intra-cluster connections
               density = 0.7 / (level + 1)  # Decrease density at lower levels
               for i in cluster_nodes:
                   for j in cluster_nodes:
                       if i < j and random.random() < density:
                           G.add_edge(i, j)
               
               level_clusters.append(cluster_nodes)
               current_node += len(cluster_nodes)
           
           clusters.append(level_clusters)
       
       # Connect clusters across levels with varying densities
       for level in range(len(clusters) - 1):
           upper_clusters = clusters[level]
           lower_clusters = clusters[level + 1]
           
           for upper in upper_clusters:
               num_connections = random.randint(1, 3)
               target_clusters = random.sample(lower_clusters, 
                                             min(num_connections, len(lower_clusters)))
               
               for lower in target_clusters:
                   # Create multiple connections between clusters
                   num_edges = random.randint(1, 3)
                   for _ in range(num_edges):
                       source = random.choice(upper)
                       target = random.choice(lower)
                       G.add_edge(source, target)
       
       return G

   def _generate_multi_scale(self, params: GraphParams) -> nx.Graph:   
       n = params.n
       G = nx.Graph()
       G.add_nodes_from(range(n))
   
       # Randomly generate per-node parameters for organic diversity
       node_strengths = np.random.uniform(0.5, 1.5, n)
       node_scales = np.random.uniform(1, 5, n)
   
       # Maximum scale to determine neighbor consideration range
       max_scale = np.max(node_scales)
       Dmax = int(np.ceil(3 * max_scale))  # Maximum distance to consider
   
       for i in range(n):
           strength_i = node_strengths[i]
           scale_i = node_scales[i]
   
           # For each node, consider neighbors within Dmax indices without wrap-around
           start_j = max(0, i - Dmax)
           end_j = min(n, i + Dmax + 1)  # +1 because range end is exclusive
   
           for j in range(start_j, end_j):
               if i == j:
                   continue  # Skip self-loops
   
               distance = abs(i - j)
   
               strength_j = node_strengths[j]
               scale_j = node_scales[j]
   
               # Compute average strength and scale
               strength = (strength_i + strength_j) / 2.0
               scale = (scale_i + scale_j) / 2.0
   
               # Calculate connection probability using exponential decay
               prob = strength * np.exp(-distance / scale)
               prob = min(prob, 1.0)  # Cap probability at 1
   
               if np.random.random() < prob:
                   G.add_edge(i, j)
   
       return G
   
   def _generate_diffusion(self, params: GraphParams) -> nx.Graph:
       G = nx.Graph()
       G.add_nodes_from(range(params.n))
   
       # Assign positions to nodes along a line with clustering tendencies
       positions = {}
       current_pos = 0.0
       while len(positions) < params.n:
           # Random step size to create clusters and varied spacing
           step = random.expovariate(1.0) * random.choice([-1, 1]) * random.uniform(0.5, 1.5)
           current_pos += step
           positions[len(positions)] = current_pos
   
       # Normalize positions to range [0, 1] to avoid dependence on node index or total nodes
       min_pos = min(positions.values())
       max_pos = max(positions.values())
       range_pos = max_pos - min_pos if max_pos != min_pos else 1.0
       for node in positions:
           positions[node] = (positions[node] - min_pos) / range_pos
   
       # Define multiple regions with varied diffusion properties
       num_regions = random.randint(3, 6)
       regions = []
       for _ in range(num_regions):
           region_center = random.uniform(0, 1)
           region = {
               'center': region_center,
               'width': random.uniform(0.05, 0.2),
               'diffusion_length': params.diffusion_length * random.uniform(0.5, 2.0),
               'diffusion_strength': params.diffusion_strength * random.uniform(0.5, 2.0),
               'influence': random.uniform(0.5, 1.5)
           }
           regions.append(region)
   
       # Assign nodes to regions based on their positions
       node_regions = {i: [] for i in range(params.n)}
       for node in range(params.n):
           pos = positions[node]
           for idx, region in enumerate(regions):
               # Compute distance to region center with periodic boundary conditions
               dist_to_center = min(abs(pos - region['center']), 1 - abs(pos - region['center']))
               if dist_to_center <= region['width']:
                   node_regions[node].append(idx)
   
       # Define decay function for edge probability based on distance
       def connection_probability(distance, decay_length):
           return np.exp(-distance / decay_length)
   
       # Define maximum distance for potential connections to enforce bandedness and sparsity
       max_connection_distance = random.uniform(0.05, 0.15)  # Adjusted to control sparsity and bandedness
   
       # For each node, consider potential connections within the max distance
       for i in range(params.n):
           pos_i = positions[i]
           # Find nodes within the maximum connection distance
           potential_neighbors = []
           for j in range(params.n):
               if i != j:
                   pos_j = positions[j]
                   # Compute minimal distance considering periodicity
                   distance = min(abs(pos_i - pos_j), 1 - abs(pos_i - pos_j))
                   if distance <= max_connection_distance:
                       potential_neighbors.append((j, distance))
   
           # Shuffle potential neighbors to introduce randomness
           random.shuffle(potential_neighbors)
   
           # Iterate over potential neighbors to decide on edge creation
           for j, distance in potential_neighbors:
               # Accumulate influences from shared regions
               total_influence = 0.0
               shared_regions = set(node_regions[i]) & set(node_regions[j])
               if shared_regions:
                   for region_idx in shared_regions:
                       region = regions[region_idx]
                       # Incorporate randomness into influence calculation
                       prob = (
                           region['diffusion_strength'] *
                           connection_probability(distance, region['diffusion_length']) *
                           region['influence'] *
                           random.uniform(0.9, 1.1)
                       )
                       total_influence += prob
               else:
                   # Base influence for nodes not sharing a region
                   base_diffusion_length = params.diffusion_length * random.uniform(0.8, 1.2)
                   base_diffusion_strength = params.diffusion_strength * random.uniform(0.1, 0.4)
                   prob = (
                       base_diffusion_strength *
                       connection_probability(distance, base_diffusion_length) *
                       random.uniform(0.8, 1.0)
                   )
                   total_influence += prob
   
               # Introduce additional randomness to mimic natural fluctuations
               total_influence *= random.uniform(0.7, 1.3)
   
               #  Probabilities stay within [0, 1]
               total_influence = min(total_influence, 1.0)
   
               # Add edge based on computed influence
               if random.random() < total_influence:
                   G.add_edge(i, j)
   
       return G


   def _make_connectivity(self, G: nx.Graph):
       components = list(nx.connected_components(G))
       while len(components) > 1:
           c1 = random.choice(list(components[0]))
           c2 = random.choice(list(components[1]))
           G.add_edge(c1, c2)
           components = list(nx.connected_components(G))

def generate_training_data(num_samples: int, data_dir: str = "data", min_nodes: int = 100, max_nodes: int = 1000) -> List[Tuple[np.ndarray, GraphParams]]:
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Check existing samples
    existing_samples = len([f for f in os.listdir(data_dir) if f.endswith('.pkl')])
    samples_needed = max(0, num_samples - existing_samples)
    
    if samples_needed == 0:
        print(f"Already have {existing_samples} samples in {data_dir}, no new samples needed.")
        return []
        
    print(f"Found {existing_samples} existing samples. Generating {samples_needed} new samples...")
    
    generator = LaplacianGenerator()
    samples = []
    
    with tqdm(total=samples_needed, desc="Generating samples") as pbar:
        for i in range(samples_needed):
            pbar.set_description(f"Generating sample {i+1}/{samples_needed}")
            
            params = generator.generate_random_params(min_nodes, max_nodes)
            print(f"\nGenerating {params.graph_type.value} graph with {params.n} nodes...")
            L, _ = generator.generate(params)
            
            # Save each sample
            sample_path = os.path.join(data_dir, f'sample_{existing_samples + i}.pkl')
            with open(sample_path, 'wb') as f:
                pickle.dump((L, params), f)
            
            samples.append((L, params))
            
            pbar.update(1)
            percentage = ((i + 1) / samples_needed) * 100
            pbar.set_postfix({"Completed": f"{percentage:.1f}%"})
    
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
    
    # Create vibrant colormap with narrow black band for zero
    colors = [
        '#0000FF',  # Vibrant blue
        '#000080',  # Dark blue
        '#000000',  # Black for zero
        '#800000',  # Dark red
        '#FF0000'   # Vibrant red
    ]
    # Position the colors to create narrow black band
    positions = [0, 0.49, 0.5, 0.51, 1]
    cmap = mcolors.LinearSegmentedColormap.from_list('custom', 
        list(zip(positions, colors)), N=256)
    
    for idx, (ax, (L, params)) in enumerate(zip(axes, display_samples)):
        # Normalize matrix values
        vmax = max(abs(L.min()), abs(L.max()))
        
        # Create norm separately
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        
        # Plot matrix with high contrast
        im = ax.imshow(L, cmap=cmap, aspect='equal', norm=norm)
        
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
    print("Generating Laplacian matrices...")
    samples = generate_training_data(
        num_samples=10000,
        data_dir='data',
        min_nodes=10,
        max_nodes=500
    )
    if samples:  # Only visualize if new samples were generated
        print("\nVisualizing 5 random samples:")
        visualize_matrices(samples, num_display=5)
        
        print("\nGenerated dataset summary:")
        type_counts = {}
        for _, params in samples:
            type_counts[params.graph_type.value] = type_counts.get(params.graph_type.value, 0) + 1
        
        print("\nGraph type distribution:")
        for type_name, count in type_counts.items():
            print(f"  {type_name}: {count}")

if __name__ == "__main__":
    main()
