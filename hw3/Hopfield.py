import numpy as np

unsigned_int = np.uint64

class Hopfield:
    def __init__(self, image_num, n):
        self.image_num, self.n = image_num, n
        self.w = np.zeros([n, n])

    def adjust_weight(self, data_arr, data_arr_mean):
        adjust = np.zeros([self.n, self.n])
        for i in range(self.n):
            # Kronecker product
            adjust[i] = (data_arr - data_arr_mean)[i] * (data_arr - data_arr_mean)
        
        return adjust / (self.n ** 2) / (data_arr_mean * (1 - data_arr_mean))

    # train hopfield
    def hop_train(self, train_arr):
        for i in range(self.image_num):
            data_arr = train_arr[i]
            data_arr_mean = float(data_arr.sum()) / len(data_arr)
            # adjust hopfield's weight
            self.w = self.w + self.adjust_weight(data_arr, data_arr_mean)
            # weight's diagonal set to 0
            idx = range(0, self.n)
            self.w[idx, idx] = 0.0

    # run hopfield to output
    def hop_run(self, data_arr):
        for i in range(self.image_num):
            data_matrix = np.tile(data_arr, (self.n, 1))
            data_matrix = self.w * data_matrix
            ouput_arr = data_matrix.sum(axis = 1)
            # normalization
            ouput_arr = (ouput_arr - float(np.amin(ouput_arr))) / \
                        (float(np.amax(ouput_arr)) - float(np.amin(ouput_arr)))

            ouput_arr[ouput_arr > 0.5] = 1.0
            ouput_arr[ouput_arr <= 0.5] = 0.0
            
            return ouput_arr

