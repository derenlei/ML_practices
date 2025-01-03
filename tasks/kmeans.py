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
