import numpy as np

class Training_Model:
    def __init__(self, learning_rate, iteration, m, sigma, cluster_num):
        self.learning_rate = learning_rate
        self.epoch = iteration
        self.m = m
        self.sigma = sigma
        self.w = np.random.rand(cluster_num)
        self.theta = self.w[0] # bias
        self.K = cluster_num
        self.Fx = 0

    def euclidean_distance(self, data, center):
        return np.linalg.norm(data - center, axis = 1)

    def activation(self, data, m, sigma):
        return np.exp(-1 / (2 * (sigma ** 2)) * (self.euclidean_distance(m, data) ** 2))

    def predict_output(self, data):
        active = self.activation(data, self.m, self.sigma)
        Fx = np.dot(self.w.T, active) + self.theta
        return active, Fx

    def RBFN(self, X_train, label_train, is_training_label, window):
        for i in range(1, self.epoch + 1):
            for data, label in zip(X_train, label_train):
                # 正向傳遞
                active, self.Fx = self.predict_output(data)
                error = label - self.Fx
                # 倒傳遞 : update all parameter
                w = self.w + (self.learning_rate * error * active)
                theta = self.theta + (self.learning_rate * error)
                pre_m = self.learning_rate * error * self.w * active * (1 / (self.sigma ** 2))
                m = self.m + np.array([pre_m[i] * (data - self.m)[i] for i in range(len(pre_m))])
                sigma = self.sigma + (self.learning_rate * error * self.w * active * (1 / (self.sigma ** 3)) * (self.euclidean_distance(data, self.m) ** 2))
                
                self.w, self.theta, self.m, self.sigma = w, theta, m, sigma

            if i % 5 == 0:
                print(f'current iter: {i}')
                is_training_label.configure(text = f'Training iter : {i}')
                window.update()
        
        is_training_label.configure(text = f"Training iter : {self.epoch} (Training Finished)")
        window.update()

