import matplotlib.pyplot as plt
import numpy as np

def read_edges_and_create_adjacency_matrix(file_path):
    edge_list = []
    id_mapping = {}
    current_index = 0

    # Read edges and create a mapping of original IDs to new compact IDs
    with open(file_path, 'r') as f:
        for line in f:
            source, target = line.strip().split()
            
            # Assign new ID if not already mapped
            if source not in id_mapping:
                id_mapping[source] = current_index
                current_index += 1
            if target not in id_mapping:
                id_mapping[target] = current_index
                current_index += 1

            # Add edge using remapped IDs
            edge_list.append((id_mapping[source], id_mapping[target]))

    # Create adjacency matrix
    num_nodes = len(id_mapping)
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for source, target in edge_list:
        adjacency_matrix[source, target] = 1  # Assuming a directed graph

    return adjacency_matrix

def calculate_page_rank(adjacency_matrix, alpha=0.85):
    num_nodes = adjacency_matrix.shape[0]
    out_degree = np.sum(adjacency_matrix, axis=1)
    stochastic_matrix = np.zeros_like(adjacency_matrix, dtype=float)

    print("out_degree: ", out_degree)

    for i in range(num_nodes):
        if out_degree[i] == 0:
            stochastic_matrix[i] = 1.0 / num_nodes  # Handle dangling nodes
        else:
            stochastic_matrix[i] = adjacency_matrix[i] / out_degree[i]

    J = np.ones((num_nodes, num_nodes)) / num_nodes
    M = alpha * stochastic_matrix + (1 - alpha) * J

    # Use numpy.linalg.eig to compute eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(M.T)
    
    # Find the index of the largest eigenvalue (close to 1)
    dominant_index = np.argmax(np.abs(eigenvalues))
    
    # Extract the corresponding eigenvector
    principal_eigenvector = np.real(eigenvectors[:, dominant_index])
    
    # Normalize to sum to 1
    principal_eigenvector = principal_eigenvector / np.sum(principal_eigenvector)
    
    return principal_eigenvector

def plot_page_rank(page_rank):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(page_rank)), page_rank, color='skyblue')
    plt.xlabel('Nodes')
    plt.ylabel('PageRank Value')
    plt.title('PageRank Distribution')
    plt.xticks(range(len(page_rank)))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


if __name__ == '__main__':
    path = 'twitter/12831.edges'

    matrix = read_edges_and_create_adjacency_matrix(path)
    
    pr = calculate_page_rank(matrix, alpha=0.1)
    #print(pr)
    print("sum of page rank: ", np.sum(pr))

    plot_page_rank(pr)
