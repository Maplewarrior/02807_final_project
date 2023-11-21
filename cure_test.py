# This code is meant for testing the cure algorithm and visualizing it on a simple dataset
# We also make a comparison to pyclusting implementation of cure
# ===============================================



from pyclustering.cluster.cure import cure;
from pyclustering.utils import read_sample;
from pyclustering.samples.definitions import FCPS_SAMPLES;
# import cluster_visualizer
from pyclustering.cluster import cluster_visualizer;
import uuid
import random 
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import cdist
import itertools
import math



# load data from the FCPS set that is provided by the library.
sample = read_sample(FCPS_SAMPLES.SAMPLE_LSUN);

# make each sample a dict with id and point
sample_with_id = [{'id': str(i), 'point': point} for i, point in enumerate(sample)]


# ===============================================
# create instance of cure algorithm for cluster analysis
# request for allocation of two clusters.
# cure_instance = cure(sample, 3, 5, 0.5, True);
# # run cluster analysis
# cure_instance.process();
# # get results of clustering
# clusters = cure_instance.get_clusters();
# # visualize obtained clusters.
# visualizer = cluster_visualizer();
# visualizer.append_clusters(clusters, sample);
# visualizer.show();
# ===============================================

# -----------------------------------------------
# Implementing the cure algorithm
# -----------------------------------------------

# ---- 1. Pick a random subsample of the data ----
#  Pick fraction of points to be used as representatives randomly
fraction = 0.4
# shuffle randomly to get random representatives
num_to_pick = int(len(sample_with_id) * fraction)

# Randomly pick points
randomly_picked_points = random.sample(sample_with_id, num_to_pick)

# If you prefer using NumPy
np_randomly_picked_points = np.random.choice(sample_with_id, size=num_to_pick, replace=False)

# print(np_randomly_picked_points)

# ---- 2. Cluster using hierachical clustering method ----

# Extracting points from the randomly picked sample
points = np.array([item['point'] for item in randomly_picked_points])

# Perform Agglomerative Clustering
n_clusters = 3  # Example number of clusters
clustering = AgglomerativeClustering(n_clusters=n_clusters)

# Fit the model
clustering.fit(points)

# Assign cluster labels back to the dictionaries in 'randomly_picked_points'
for i, point_dict in enumerate(randomly_picked_points):
    point_dict['cluster'] = clustering.labels_[i]


# ---- 3. Select representative points to be as far away from each other as possible ----

def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

def find_furthest_points(cluster, n):
    """Find n points that are progressively farther away from each other."""
    c_points = np.array([item['point'] for item in cluster])
    point_ids = np.array([item['id'] for item in cluster])

    if n <= 0 or not c_points.any() or n > len(c_points):
        return []

    # Start by finding the two farthest points
    max_distance = 0
    farthest_pair_indices = None
    for i, j in itertools.combinations(range(len(c_points)), 2):
        distance = calculate_distance(c_points[i], c_points[j])
        if distance > max_distance:
            max_distance = distance
            farthest_pair_indices = (i, j)

    selected_indices = list(farthest_pair_indices)

    # Iteratively find the next point that is farthest from all previously selected points
    while len(selected_indices) < n:
        next_index = None
        max_distance_to_set = 0

        for i, point in enumerate(c_points):
            if i not in selected_indices:
                min_distance_to_set = min(calculate_distance(point, c_points[selected_index]) for selected_index in selected_indices)
                if min_distance_to_set > max_distance_to_set:
                    max_distance_to_set = min_distance_to_set
                    next_index = i

        if next_index is not None:
            selected_indices.append(next_index)

    return [cluster[i] for i in selected_indices]


    # distances = cdist(c_points, c_points)
    # np.fill_diagonal(distances, np.inf)
    # min_distances = np.min(distances, axis=1)
    # rep_idx = np.argsort(min_distances)[:n_representatives]
    # print(rep_idx)
    # representatives = [cluster[i] for i in rep_idx]
    return representatives

n_representatives = 4
representatives = []
for i in range(n_clusters):
    cluster = [obs for obs in randomly_picked_points if obs['cluster'] == i]
    # print(cluster)
    representatives.append(find_furthest_points(cluster, n=n_representatives))
    
print(representatives) 





# ---- 4. Move representative points a fraction f closer to centroid of cluster ----
# Move representative points a fraction f closer to centroid of cluster



# ---- 5. Rescan dataset and place points in "closes" cluster. Find closest representative to p and assign to this cluster ----



# Extract points and cluster labels
points = np.array([item['point'] for item in randomly_picked_points])
cluster_labels = np.array([item['cluster'] for item in randomly_picked_points])
ids_in_sample = np.array([item['id'] for item in randomly_picked_points])

# id_not_in_sample = np.array([item['id'] for item in sample_with_id if item['id'] not in ids_in_sample])
# point not in sample 
points_not_in_sample = np.array([item['point'] for item in sample_with_id if item['id'] not in ids_in_sample])

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(points_not_in_sample[:, 0], points_not_in_sample[:, 1], c='gray', marker='x', alpha=0.5)
plt.scatter(points[:, 0], points[:, 1], c=cluster_labels, cmap='viridis', marker='o')

for nc in range(n_representatives-1):
    n_reps= representatives[nc]
    pts = np.array([item['point'] for item in n_reps])
    plt.scatter(pts[:, 0], pts[:, 1], c='red', marker='x', s=100)

    # add label to pts
    for i, txt in enumerate([item['id'] for item in n_reps]):
        plt.annotate(nc, (pts[i, 0], pts[i, 1]), fontsize=12)


# Optionally, adding text labels (IDs) to the points
# for point in randomly_picked_points:
#     plt.text(point['point'][0], point['point'][1], point['id'], fontsize=12)

plt.title('Clustered Points with IDs')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.colorbar(label='Cluster ID')
plt.show()