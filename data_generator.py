import numpy as np
import networkx as nx
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

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
   
       generator_func = self.supported_types[params.graph_type]
       G = generator_func(params)
       
       if not nx.is_connected(G):
           self._make_connectivity(G)
           
       L = nx.laplacian_matrix(G).toarray()
       return L, G

   
   def _generate_path_community(self, params: GraphParams) -> nx.Graph:
       G = nx.Graph()
   
       # Randomly shuffle node IDs to avoid index-based structure
       all_nodes = list(range(params.n))
       random.shuffle(all_nodes)
   
       # Generate variable-sized communities using a random distribution
       remaining_nodes = all_nodes[:]
       community_sizes = []
       alpha = random.uniform(2.0, 3.0)  # Add variation to the power law exponent
   
       for _ in range(params.num_communities - 1):
           if len(remaining_nodes) <= 5:  # Ensure we don't over-sample
               break
           size = max(5, min(int(len(remaining_nodes) * random.random() ** alpha), len(remaining_nodes)))
           community_sizes.append(size)
           remaining_nodes = remaining_nodes[size:]
       community_sizes.append(len(remaining_nodes))  # Include remaining nodes
   
       # Create communities with internal structure
       used_nodes = set()
       community_centers = []
   
       for size in community_sizes:
           size = min(size, len(all_nodes))  # Clamp size to available nodes
           community_nodes = random.sample(all_nodes, size)
           all_nodes = [node for node in all_nodes if node not in community_nodes]
           core_size = max(3, int(size * random.uniform(0.1, 0.3)))  # Variable core size
           core_size = min(core_size, len(community_nodes))  # Ensure core size is valid
           core_nodes = community_nodes[:core_size]
           periphery_nodes = community_nodes[core_size:]
   
           # Create a dense core with randomized connectivity
           for i in core_nodes:
               for j in core_nodes:
                   if i != j and random.random() < random.uniform(0.7, 0.9):
                       G.add_edge(i, j)
   
           # Connect periphery nodes to the core with random probabilities
           for node in periphery_nodes:
               if core_nodes:
                   center = random.choice(core_nodes)
                   G.add_edge(node, center)
                   for core_node in core_nodes:
                       if random.random() < random.uniform(0.2, 0.5):
                           G.add_edge(node, core_node)
   
           # Select a random center node for inter-community connections
           if core_nodes:
               community_centers.append(random.choice(core_nodes))
           used_nodes.update(community_nodes)
   
       # Add inter-community connections based on distance
       for i, center1 in enumerate(community_centers):
           for j, center2 in enumerate(community_centers[i + 1:], i + 1):
               distance = abs(i - j)
               prob = params.inter_community_prob * random.uniform(0.5, 1.5) * np.exp(-distance / random.uniform(1.5, 3.0))
               if random.random() < prob:
                   num_bridges = random.randint(1, 3)
                   for _ in range(num_bridges):
                       if used_nodes:
                           source = random.choice(list(used_nodes))
                           target = random.choice(list(used_nodes))
                           G.add_edge(source, target)
   
       return G

   def _generate_circulant_community(self, params: GraphParams) -> nx.Graph:
       G = nx.Graph()
       community_size = params.n // params.num_communities
       
       for c in range(params.num_communities):
           start = c * community_size
           for i in range(start, start + community_size):
               for j in range(1, params.k_neighbors + 1):
                   node1 = i
                   node2 = start + ((i - start + j) % community_size)
                   G.add_edge(node1, node2)
                   
       for i in range(params.num_communities):
           for j in range(i + 1, params.num_communities):
               start_i = i * community_size
               start_j = j * community_size
               for a in range(community_size):
                   if np.random.random() < params.inter_community_prob:
                       if np.random.random() < 0.5:
                           G.add_edge(start_i + a, start_j + (community_size - a - 1))
                       else:
                           G.add_edge(start_i + a, start_j + a)
       
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
       G = nx.Graph()
       G.add_nodes_from(range(params.n))
       
       for i in range(params.n):
           for j in range(i + 1, params.n):
               total_prob = 0
               for width, strength in zip(params.scale_widths, params.scale_strengths):
                   distance = min(abs(i - j), params.n - abs(i - j)) if params.periodic else abs(i - j)
                   total_prob += strength * np.exp(-distance/width)
               
               if np.random.random() < total_prob:
                   G.add_edge(i, j)
       
       return G

   def _generate_diffusion(self, params: GraphParams) -> nx.Graph:
       G = nx.Graph()
       G.add_nodes_from(range(params.n))
   
       # Assign random positions to nodes in a 2D plane
       positions = {i: (random.uniform(0, 1), random.uniform(0, 1)) for i in range(params.n)}
   
       # Define multiple regions with different diffusion characteristics
       num_regions = random.randint(3, 6)
       regions = []
       for _ in range(num_regions):
           region = {
               'center': (random.uniform(0, 1), random.uniform(0, 1)),
               'radius': random.uniform(0.1, 0.3),
               'diffusion_length': params.diffusion_length * random.uniform(0.5, 1.5),
               'diffusion_strength': params.diffusion_strength * random.uniform(0.5, 1.5)
           }
           regions.append(region)
   
       # Assign nodes to regions based on proximity
       node_regions = {i: [] for i in range(params.n)}
       for node in range(params.n):
           x_node, y_node = positions[node]
           for idx, region in enumerate(regions):
               x_center, y_center = region['center']
               distance_to_center = ((x_node - x_center) ** 2 + (y_node - y_center) ** 2) ** 0.5
               if distance_to_center <= region['radius']:
                   node_regions[node].append(idx)
   
       # Define a base decay function for edge probability based on Euclidean distance
       def connection_probability(distance, decay_length):
           return np.exp(-distance / decay_length)
   
       # Loop over all pairs of nodes to decide edge creation based on proximity and regional properties
       for i in range(params.n):
           for j in range(i + 1, params.n):
               x_i, y_i = positions[i]
               x_j, y_j = positions[j]
               distance = ((x_i - x_j) ** 2 + (y_i - y_j) ** 2) ** 0.5
   
               # Initialize edge probability
               total_influence = 0
   
               # Check for shared regions and accumulate influences
               shared_regions = set(node_regions[i]) & set(node_regions[j])
               if shared_regions:
                   for region_idx in shared_regions:
                       region = regions[region_idx]
                       prob = region['diffusion_strength'] * connection_probability(distance, region['diffusion_length'])
                       # Randomly vary the probability within a small range
                       prob *= random.uniform(0.9, 1.1)
                       total_influence = max(total_influence, prob)
               else:
                   # Nodes not in the same region have a base low probability to connect
                   base_diffusion_length = params.diffusion_length * random.uniform(0.8, 1.2)
                   base_diffusion_strength = params.diffusion_strength * random.uniform(0.2, 0.5)
                   prob = base_diffusion_strength * connection_probability(distance, base_diffusion_length)
                   # Randomly vary the probability within a small range
                   prob *= random.uniform(0.8, 1.0)
                   total_influence = max(total_influence, prob)
   
               # Probability is within [0, 1]
               total_influence = min(max(total_influence, 0), 1)
   
               # Add edge based on computed probability
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

def generate_training_data(num_samples: int, min_nodes: int = 100, max_nodes: int = 1000) -> List[Tuple[np.ndarray, GraphParams]]:
   generator = LaplacianGenerator()
   samples = []
   
   with tqdm(total=num_samples, desc="Generating samples") as pbar:
       for i in range(num_samples):
           pbar.set_description(f"Generating sample {i+1}/{num_samples}")
           
           params = generator.generate_random_params(min_nodes, max_nodes)
           print(f"\nGenerating {params.graph_type.value} graph with {params.n} nodes...")
           L, _ = generator.generate(params)
           samples.append((L, params))
           
           pbar.update(1)
           percentage = ((i + 1) / num_samples) * 100
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
    # Generate sample dataset
    print("Generating Laplacian matrices...")
    samples = generate_training_data(
        num_samples=20,
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
