def kmeans(X, k, max_iter=100, tolerance=1e-4):
    """
    Args:
        X (torch.Tensor): Input data of shape (N, D).
        k (int): Number of clusters.
        max_iter (int): Maximum number of iterations.
        tolerance (float): Convergence threshold.
    
    Returns:
        centroids (torch.Tensor): Final cluster centroids of shape (k, D).
        labels (torch.Tensor): Cluster labels for each data point, shape (N,).
    """
    assert X.dim() == 2, "Input data X must be a 2D tensor (N, D)"

    # 1) Random initialization of centroids: pick k points from X
    indices = torch.randperm(X.size(0))[:k]
    centroids = X[indices].clone()  # shape: (k, D)

    # Labels for each data point
    labels = torch.zeros(X.size(0), dtype=torch.long)

    for i in range(max_iter):
        # 2) Assign points to the nearest centroid
        #    - We compute distance from each point to each centroid
        #    - Then pick the centroid with the minimal distance
        # shape of distances: (N, k)
        distances = torch.cdist(X, centroids, p=2)
        new_labels = distances.argmin(dim=1)  # shape: (N,)
        
        # 3) Update centroids by taking the mean of assigned points
        new_centroids = []
        for cluster_idx in range(k):
            cluster_points = X[new_labels == cluster_idx]
            if len(cluster_points) == 0:
                # If a cluster is empty, re-initialize its centroid randomly
                # Or you could pick some other approach
                new_centroids.append(X[torch.randint(0, X.size(0), (1,))][0])
            else:
                new_centroids.append(cluster_points.mean(dim=0))
        new_centroids = torch.stack(new_centroids)
        
        # Check for convergence
        shift = (centroids - new_centroids).pow(2).sum().sqrt()
        centroids = new_centroids
        labels = new_labels

        if shift < tolerance:
            print(f"K-means converged after {i+1} iterations.")
            break

    return centroids, labels
#====================================================================================================================================================






import torch

def kmeans(dataset, k, max_iters=100):
    assert len(dataset) >= k, "Dataset size must be greater than or equal to the number of clusters (k)."

    # Initialize cluster centers
    centers = dataset[:k]

    for _ in range(max_iters):
        # Reset points and new centers
        points = [[] for _ in range(k)]
        new_centers = []

        # Assign points to the nearest cluster
        for data in dataset:
            dis = torch.sqrt(torch.sum((data - centers) ** 2, dim=1))
            cluster_idx = torch.argmin(dis, dim=0).item()
            points[cluster_idx].append(data)

        # Update cluster centers
        for i, cluster in enumerate(points):
            if len(cluster) > 0:
                new_centers.append(torch.mean(torch.stack(cluster), dim=0))
            else:
                # Retain previous center if no points assigned
                new_centers.append(centers[i])

        new_centers = torch.stack(new_centers)

        # Check for convergence
        if torch.allclose(new_centers, centers):
            break

        centers = new_centers

    return centers

def euclidean_distance(a, b):
    return np.sqrt(((a - b) ** 2).sum(axis=1))
    
def k_means_clustering(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]], max_iterations: int) -> list[tuple[float, float]]:
    points = np.array(points)
    centroids = np.array(initial_centroids)
    for iteration in range(max_iterations):
        distances = np.array([euclidean_distance(points, c) for c in centroids])
        idx = np.argmin(distances, dim=0)
        new_centroids = np.array([np.mean(points[idx == i] ,axis=0) for i in range(k)])
        centroids = new_centroids
    return [tuple(centroid) for centroid in centroids]
