import numpy as np

class KMeans(object):
    """
    K-Means clustering class.

    We also use it to make prediction by attributing labels to clusters.
    """

    def __init__(self, K, max_iters=100):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            K (int): number of clusters
            max_iters (int): maximum number of iterations
        """
        self.K = K
        self.max_iters = max_iters
        self.centers = None
        self.cluster_center_label = None

    def k_means(self, data, max_iter=100):
        """
        Main K-Means algorithm that performs clustering of the data.

        Arguments:
            data (array): shape (N,D) where N is the number of data samples, D is number of features.
            max_iter (int): the maximum number of iterations
        Returns:
            centers (array): shape (K,D), the final cluster centers.
            cluster_assignments (array): shape (N,) final cluster assignment for each data point.
        """

        centers = self.init_centers(data, self.K)

        for i in range(max_iter):
            if ((i+1) % 10 == 0):
                print(f"Iteration {i+1}/{max_iter}...")
            old_centers = centers.copy()  # keep in memory the centers of the previous iteration

            distances = self.compute_distance(data, old_centers)
            cluster_assignments = self.find_closest_cluster(distances)
            centers = self.compute_centers(data, cluster_assignments, self.K)

            if np.all(centers == old_centers):
                print(f"K-Means has converged after {i+1} iterations!")
                break

        cluster_assignments = self.find_closest_cluster(distances)
        return centers, cluster_assignments

    def fit(self, training_data, training_labels):
        """
        Train the model and return predicted labels for training data.

        You will need to first find the clusters by applying K-means to the data, then to attribute a label to each cluster based on the labels.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): labels of shape (N,)
        Returns:
            pred_labels (array): labels of shape (N,)
        """

        self.centers, cluster_assignments = self.k_means(training_data, self.max_iters)
        self.cluster_center_label = np.zeros(self.K)
        for i in range(self.K):
            label = np.argmax(np.bincount(training_labels[cluster_assignments == i]))
            self.cluster_center_label[i] = label
        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data given the cluster center and their labels.

        To do this, first assign data points to their closest cluster, then use the label
        of that cluster as prediction.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """

        distances = self.compute_distance(test_data, self.centers)
        cluster_assignments = self.find_closest_cluster(distances)

        pred_labels = self.cluster_center_label[cluster_assignments]
        return pred_labels

    def init_centers(self, data, K):
        """
        Randomly pick K data points from the data as initial cluster centers.

        Arguments:
            data: array of shape (NxD) where N is the number of data points and D is the number of features (:=pixels).
            K: int, the number of clusters.
        Returns:
            centers: array of shape (KxD) of initial cluster centers
        """
        # 1. plain initialization
        centers = data[np.random.permutation(data.shape[0])[:K]]

        # 2. farthest initialization
        # centers = []
        # centers.append(data[np.random.randint(data.shape[0])])
        # for i in range(self.K-1):
        #     dist = np.zeros((data.shape[0], len(centers)))
        #     for j in range(len(centers)):
        #         dist[:,j] = np.sum((data - centers[j])**2, axis=1)
        #     min_dist = np.min(dist, axis=1)
        #     new_center_idx = np.argmax(min_dist)
        #     centers.append(data[new_center_idx])

        return np.array(centers)

    def compute_distance(self, data, centers):
        """
        Compute the euclidean distance between each datapoint and each center.

        Arguments:
            data: array of shape (N, D) where N is the number of data points, D is the number of features (:=pixels).
            centers: array of shape (K, D), centers of the K clusters.
        Returns:
            distances: array of shape (N, K) with the distances between the N points and the K clusters.
        """
        N = data.shape[0]
        K = centers.shape[0]

        distances = np.zeros((N,K))
        for k in range(K):
            distances[:, k] = np.sqrt(np.sum((data-centers[k])**2, axis=1))
        return distances

    def compute_centers(self, data, cluster_assignments, K):
        """
        Compute the center of each cluster based on the assigned points.

        Arguments:
            data: data array of shape (N,D), where N is the number of samples, D is number of features
            cluster_assignments: the assigned cluster of each data sample as returned by find_closest_cluster(), shape is (N,)
            K: the number of clusters
        Returns:
            centers: the new centers of each cluster, shape is (K,D) where K is the number of clusters, D the number of features
        """
        centers = np.zeros((K, data.shape[-1]))
        for k in range(K):
            centers[k] = np.mean(data[cluster_assignments == k], axis = 0)
        return centers

    def find_closest_cluster(self, distances):
        """
        Assign datapoints to the closest clusters.

        Arguments:
            distances: array of shape (N, K), the distance of each data point to each cluster center.
        Returns:
            cluster_assignments: array of shape (N,), cluster assignment of each datapoint, which are an integer between 0 and K-1.
        """
        cluster_assignments = np.argmin(distances, axis=1)
        return cluster_assignments
