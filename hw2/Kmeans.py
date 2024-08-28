import numpy as np

class Kmeans:
    def __init__(self, data_train, cluster_num):
        self.X_train = data_train
        self.data_num = data_train.shape[0]
        self.K = cluster_num

    def euclidean_distance(self, data, center):
        dis = np.linalg.norm(data - center, axis = 1)
        return dis
    
    def k_means(self):
        # K center points
        center_pos = np.random.choice(self.data_num, size = self.K, replace = False)
        center_list = self.X_train[center_pos] # self.K's center points
        # iterate 500 times to get precise cluster classification
        for i in range(200):
            self_cluster = np.zeros(self.data_num)
            # classify cluster
            for j in range(self.data_num):
                self_cluster[j] = np.argmin([self.euclidean_distance(self.X_train[j], center_list)])
            # assign new center position
            for cluster_idx in range(self.K):
                belong_center_list = np.where(self_cluster == cluster_idx)[0]
                temp = np.zeros(self.X_train.shape[1])
                for idx in belong_center_list: 
                    temp += self.X_train[idx]
                center_list[cluster_idx] = temp / len(belong_center_list)
        
        sigma = np.zeros(self.X_train.shape[1])
        for cluster_idx in range(self.K):
            sigma[cluster_idx] = np.sum(self.euclidean_distance(self.X_train[self_cluster == cluster_idx], center_list[cluster_idx])) / len(self.X_train[self_cluster == cluster_idx])

        return center_list, sigma

