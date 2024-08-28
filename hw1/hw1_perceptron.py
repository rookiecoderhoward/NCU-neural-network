import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tkinter import *
from tkinter import ttk

def combo_box_on_select(event):
    selected_file.set(combo_box.get())
    print(f'The chosen file is {selected_file.get()}')

def reform_data(dataset):
    np.random.shuffle(dataset)
    train_num = int(2 * dataset.shape[0] / 3)

    data_train = dataset[:train_num, ...]
    data_test = dataset[train_num:, ...]
    data_train_x, data_train_y, data_train_d = np.hsplit(data_train, dataset.shape[1])
    data_train = np.hstack((data_train_x, data_train_y))
    
    data_test_x, data_test_y, data_test_d = np.hsplit(data_test, dataset.shape[1])
    data_test = np.hstack((data_test_x, data_test_y))

    return data_train, data_train_d, data_test, data_test_d

data_path = os.path.join(os.getcwd(), "NN_HW1_DataSet")
hw1_dataset_list = os.listdir(data_path)
# hw1_dataset_list = ['2Ccircle1.txt', '2Circle1.txt', '2CloseS.txt', '2CloseS2.txt', '2CloseS3.txt', '2cring.txt', '2CS.txt', '2Hcircle1.txt', '2ring.txt', 'perceptron1.txt', 'perceptron2.txt']

def plot_data(w, data_array, data_array_d, chosen_file_name, is_train, is_test):
    train_or_test = ""
    if is_train == 1 and is_test == 0: train_or_test = "Training Data"
    elif is_train == 0 and is_test == 1: train_or_test = "Testing Data"
    
    d_max, d_min = np.amax(data_array_d), np.amin(data_array_d)

    if d_max == 1.0 and d_min == 0.0:
        for i in range(data_array.shape[0]):
            if data_array_d[i][0] == 0.0:
                plt.scatter(data_array[i][0], data_array[i][1], c = 'blue', s = 10)
            else:
                plt.scatter(data_array[i][0], data_array[i][1], c = 'red', s = 10)
    elif d_max == 2.0 and d_min == 1.0:
        for i in range(data_array.shape[0]):
            if data_array_d[i][0] == 1.0:
                plt.scatter(data_array[i][0], data_array[i][1], c = 'blue', s = 10)
            else:
                plt.scatter(data_array[i][0], data_array[i][1], c = 'red', s = 10)
    
    # find maximum, minimum in index 0(value x) of data_array
    max_x, min_x = np.max(data_array[..., 0]), np.min(data_array[..., 0])

    # linear eq: w1x+w2y-w0=0 (w0 is bias) ---> y = -(w1x-w0)/w2
    max_y, min_y = -1*(w[1]*max_x - w[0])/w[2], -1*(w[1]*min_x - w[0])/w[2]

    chosen_file_name = chosen_file_name.split('.')[0] + f' ({train_or_test})'
    save_path = "D:\\大三上\\類神經網路\\hw1\\hw1_截圖"
    plt.plot([min_x, max_x], [min_y, max_y])
    plt.title(chosen_file_name)
    plt.xlabel('weight  ---  w0(bias):{:.3f}   w1:{:.3f}   w2:{:.3f}'.format(w[0], w[1], w[2]))
    plt.savefig(save_path + '\\' + chosen_file_name)
    plt.show()
    plt.close()


def is_match_d(n):
    if n > 0: return 1
    else: return 0

def Train():
    start_time = time.time()
    # get the chosen txt file
    file_name = selected_file.get()
    print(f'目前 {file_name} 執行結果:')
    file_name_path = os.path.join(data_path, file_name)
    dataset = np.loadtxt(file_name_path, delimiter = ' ') # delimiter default = space
    data_train, data_train_d, data_test, data_test_d = reform_data(dataset)

    # initialize weight, max_iter and learning rate
    w = np.random.rand(3) 
    X = np.array([-1, np.random.rand(), np.random.rand()]) # init X[0] = -1

    # Entry.get() is a string, change it to integer or floating point
    max_iter = int(Max_training_iter_entry.get())
    learning_rate = float(learning_rate_entry.get())

    # find max, min in data_train_d
    data_train_d_max, data_train_d_min = np.amax(data_train_d), np.amin(data_train_d)
    # find max, min in data_test_d
    data_test_d_max, data_test_d_min = np.amax(data_test_d), np.amin(data_test_d)

    N = max_iter

    while N != 0:
        for i in range(data_train.shape[0]):
            X = np.array([-1, data_train[i][0], data_train[i][1]])
            cur_D = data_train_d[i][0]
            if (data_train_d_max == 2.0 and data_train_d_min == 1.0):
                cur_D -= 1
            dot_product = np.dot(w.T, X)
            # check whether need to adjust "current_weight" or not
            if is_match_d(dot_product) != int(cur_D):
                if is_match_d(dot_product) == 1 and int(cur_D) == 0:
                    w = w - learning_rate * X
                elif is_match_d(dot_product) == 0 and int(cur_D) == 1:
                    w = w + learning_rate * X
                # from data_train[0] to retrain the weight
                i = 0 
                N -= 1 # iteration_number minus 1
                continue
            # iteration_number minus 1
            N -= 1 
            if N == 0:
                break
    weight_result_label.configure(text = "weight_result : " + '\nw0(bias):{:.3f}, w1:{:.3f}, w2:{:.3f}'.format(w[0], w[1], w[2]))
    print(f'鍵結值{w}')
    end_time = time.time()
    time_label.configure(text = f'elapsed time : {round(end_time - start_time, 3)} seconds')

    # cal train_accuracy and round to third decimal place
    train_success_num = 0
    for j in range(data_train.shape[0]):
        X = np.array([-1, data_train[j][0], data_train[j][1]])
        cur_D = data_train_d[j][0]
        if (data_train_d_max == 2.0 and data_train_d_min == 1.0):
            cur_D -= 1
        cur_dot = X.dot(w.T)
        if is_match_d(cur_dot) == int(cur_D):
            train_success_num += 1
    train_accuracy = round((train_success_num / data_train.shape[0]) * 100, 3)
    training_accuracy_label.configure(text = "training_accuracy : " + f'{train_accuracy}%')
    print(f'訓練資料正確率:{train_accuracy}%')

    # cal test_accuracy and round to third decimal place
    test_success_num = 0
    for j in range(data_test.shape[0]):
        X = np.array([-1, data_test[j][0], data_test[j][1]])
        cur_D = data_test_d[j][0]
        if (data_test_d_max == 2.0 and data_test_d_min == 1.0):
            cur_D -= 1
        cur_dot = X.dot(w.T)
        if is_match_d(cur_dot) == int(cur_D):
            test_success_num += 1
    test_accuracy = round((test_success_num / data_test.shape[0]) * 100, 3)
    testing_accuracy_label.configure(text = "testing_accuracy : " + f'{test_accuracy}%')
    print(f'測試資料正確率:{test_accuracy}%')

    # plot training result
    plot_data(w, data_train, data_train_d, file_name, 1, 0)
    # plot testing result
    plot_data(w, data_test, data_test_d, file_name, 0, 1)


# GUI interface
if __name__ == "__main__":

    window = Tk()
    window.title("HW1_Perceptron")
    window.geometry("500x380")

    combo_box_frame = Frame(window)
    combo_box_label = Label(window, text = "Choose a data file", font = ("Arial", 11, "normal"))
    combo_box_label.pack()
    # ttk combobox, contents = hw1_dataset_list
    combo_box = ttk.Combobox(window, values = hw1_dataset_list)
    combo_box.pack(side = TOP)
    # binding event, trigger "on_select" function when the selection changes
    combo_box.bind("<<ComboboxSelected>>", combo_box_on_select)
    selected_file = StringVar() # combobox string buffer, store file_name

    Max_training_iter_frame = Frame(window)
    Max_training_iter_frame.pack()
    Max_training_iter_label = Label(Max_training_iter_frame, text = "max_train_rounds", font = ("Arial", 11, "normal"))
    Max_training_iter_label.pack(side = LEFT)
    Max_training_iter_entry = Entry(Max_training_iter_frame)
    Max_training_iter_entry.pack(side = LEFT)
    Max_training_iter_entry.config(width = 15)

    learning_rate_frame = Frame(window)
    learning_rate_frame.pack(side = TOP)
    learning_rate_label = Label(learning_rate_frame, text = "learning rate", font = ("Arial", 11, "normal"))
    learning_rate_label.pack(side = LEFT)
    learning_rate_entry = Entry(learning_rate_frame)
    learning_rate_entry.pack(side = LEFT)
    learning_rate_entry.config(width = 20)

    training_accuracy_label = Label(window, text = "training_accuracy : ", font = ("Arial", 11, "normal"))
    training_accuracy_label.pack()
    testing_accuracy_label = Label(window, text = "testing_accuracy : ", font = ("Arial", 11, "normal"))
    testing_accuracy_label.pack()
    weight_result_label = Label(window, text = "weight_result : ", font = ("Arial", 11, "normal"))
    weight_result_label.pack()

    # execution time(determine weight in Train())
    time_label = Label(window, text = "elapsed time : ", font = ("Arial", 11, "normal"))
    time_label.pack()

    button_start_train = Button(window, text = "Start training", font = ("Arial", 10, "bold"), command = Train)
    button_start_train.pack()

    window.mainloop()

