import torch
import torch.nn.functional as F


def _make_generator(device, seed):
    if seed is None:
        return None
    generator = torch.Generator() if device.type == "cpu" else torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    return generator


def _initial_centroids(features, num_clusters, generator=None):
    perm = torch.randperm(features.shape[0], device=features.device, generator=generator)[:num_clusters]
    return features[perm].clone()


def _cluster_means(features, assignments, centroids):
    new_centroids = centroids.clone()
    for cluster_idx in range(centroids.shape[0]):
        cluster_features = features[assignments == cluster_idx]
        if cluster_features.numel() > 0:
            new_centroids[cluster_idx] = cluster_features.mean(dim=0)
    return F.normalize(new_centroids, p=2, dim=1)


@torch.no_grad()
def torch_kmeans(features, num_clusters, num_iters=20, chunk_size=4096, generator=None):
    """Spherical K-Means over L2-normalized features."""
    if features.ndim != 2:
        raise ValueError("features must be a 2D tensor")
    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive")

    features = F.normalize(features.float(), p=2, dim=1)
    num_samples = features.shape[0]
    if num_samples == 0:
        raise ValueError("cannot run k-means on an empty tensor")

    if num_samples <= num_clusters:
        repeats = (num_clusters + num_samples - 1) // num_samples
        centroids = features.repeat(repeats, 1)[:num_clusters].clone()
        return F.normalize(centroids, p=2, dim=1)

    centroids = _initial_centroids(features, num_clusters, generator=generator)
    for _ in range(int(num_iters)):
        assignments = []
        for start in range(0, num_samples, max(int(chunk_size), 1)):
            sims = features[start:start + chunk_size] @ centroids.t()
            assignments.append(sims.argmax(dim=1))
        assignments = torch.cat(assignments, dim=0)
        centroids = _cluster_means(features, assignments, centroids)
    return centroids


@torch.no_grad()
def identity_kmeans(features, pids, num_classes, prototypes_per_id, num_iters=20, seed=None):
    """Build fixed-count, identity-owned prototype slots."""
    features = F.normalize(features.float(), p=2, dim=1)
    pids = pids.long().to(features.device)
    dim = features.shape[1]
    generator = _make_generator(features.device, seed)
    banks = []

    for pid in range(int(num_classes)):
        identity_features = features[pids == pid]
        if identity_features.numel() == 0:
            raise ValueError(
                f"prototype initialization did not see any samples for train identity {pid}; "
                "check the init loader and contiguous pid remapping"
            )
        centroids = torch_kmeans(
            identity_features,
            int(prototypes_per_id),
            num_iters=num_iters,
            generator=generator,
        )
        banks.append(F.normalize(centroids.reshape(int(prototypes_per_id), dim), p=2, dim=1))

    return torch.cat(banks, dim=0)
