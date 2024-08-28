import numpy as np
import matplotlib.patches as patch
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from tkinter import ttk
from Kmeans import Kmeans
from Training_Model import Training_Model
from new_simulate import Car
import time
import os

def combo_box_on_select(event):
    global selected_file
    selected_file.set(combo_box.get())
    print(f'The chosen file is {selected_file.get()}')

def reform_data(dataset):
    global split_num
    split_num = dataset.shape[1]

    X_train, label_train = dataset[..., :split_num-1], dataset[..., split_num-1]
    # normalization
    label_train = (label_train - np.amin(label_train)) / (np.amax(label_train) - np.amin(label_train))

    return X_train, label_train

data_path = os.path.join(os.getcwd(), "NN_HW2_DataSet")
hw2_dataset_list, track_coordinate = (os.listdir(data_path))[0:2], os.listdir(data_path)[2]

def plot_track(backgroud, figure):
    global figure_plot, init_x, init_y, init_phi
    figure.clear()
    figure_plot = figure.add_subplot(111)
    figure_plot.clear()

    track_path = os.path.join(data_path, track_coordinate)
    temp_track_data = []
    with open(track_path, 'r') as f:
        for line in f: 
            temp_track_data.append(list(map(float, line.strip().split(','))))
    track_data = temp_track_data[3:]
    init_x, init_y, init_phi = temp_track_data[0][0], temp_track_data[0][1], temp_track_data[0][2]
    final_line_p1, final_line_p2 = temp_track_data[1:3][0], temp_track_data[1:3][1]

    for i in range(len(track_data)-1):
        x_1, y_1 = track_data[i][0], track_data[i][1]
        x_2, y_2 = track_data[i+1][0], track_data[i+1][1]
        figure_plot.plot([x_1, x_2], [y_1, y_2], 'indigo')
    # starting line
    figure_plot.plot([-6, 6], [0, 0], 'lightcoral', linewidth = 2)
    # final line (rectangle)
    final_line = patch.Rectangle(final_line_p1, final_line_p2[0] - final_line_p1[0], final_line_p2[1] - final_line_p1[1], fill = True, facecolor = 'lightcoral')
    figure_plot.add_patch(final_line)
    backgroud.draw()

def Train(learning_rate, epoch, is_training_label, window):
    learning_rate = float(learning_rate_entry.get())
    epoch = int(Epoch_entry.get())

    global training_model, K
    start_time_train = time.time()
    file_name = selected_file.get()
    file_name_path = os.path.join(data_path, file_name)
    dataset = np.loadtxt(file_name_path, delimiter = ' ') # delimiter default = space
    X_train, label_train = reform_data(dataset)
    K = X_train.shape[1]
    kmeans = Kmeans(X_train, K)
    m, sigma = kmeans.k_means()

    print(f'k-means cluster center : {m}')
    print(f'sigma : {sigma}')

    training_model = Training_Model(learning_rate, epoch, m, sigma, K)
    training_model.RBFN(X_train, label_train, is_training_label, window)
    end_time_train = time.time()
    time_train_label.configure(text = f'Training elapsed time : {round(end_time_train - start_time_train, 5)}')

def start_simulate(front_dis_label, right_dis_label, left_dis_label, background, window):
    start_time_simulate = time.time()
    txt_name = 'default'
    if split_num == 4: 
        four_six = 4
        txt_name = 'track4D.txt'
    elif split_num == 6: 
        four_six = 6
        txt_name = 'track6D.txt'
    # print(init_x, init_y, init_phi)
    car = Car(init_x, init_y, init_phi)
    simulate_result = car.Start(training_model, four_six, front_dis_label, right_dis_label, left_dis_label, background, figure_plot, window)
    # save simulate_result as track(4||6)D.txt
    with open (txt_name, 'w') as file:
        for i in range (len(simulate_result)): 
            temp_str = ""
            for j in range(len(simulate_result[i])): 
                if j == 0: temp_str += (f'{simulate_result[i][j]}')
                else: temp_str += (f' {simulate_result[i][j]}')
            if i == 0: 
                file.write(temp_str)
            else: 
                file.write(f'\n{temp_str}')
    end_time_simulate = time.time()
    time_simulate_label.configure(text = f'Simulating elapsed time : {round(end_time_simulate - start_time_simulate, 5)}')


if __name__ == "__main__":

    window = Tk()
    window.title("HW2_AutoCar")
    window.geometry("800x800")

    combo_box_frame = Frame(window)
    combo_box_label = Label(window, text = "Choose a data file", font = ("Arial", 11, "normal"))
    combo_box_label.pack()
    # ttk combobox, contents = hw2_dataset_list
    combo_box = ttk.Combobox(window, values = hw2_dataset_list)
    combo_box.pack(side = TOP)
    # binding event, trigger "on_select" function when the selection changes
    combo_box.bind("<<ComboboxSelected>>", combo_box_on_select)
    selected_file = StringVar() # combobox string buffer, store file_name

    figure = Figure(figsize = (5, 4))
    plot_frame = Frame(window)
    plot_frame.pack(side = TOP)
    background = FigureCanvasTkAgg(figure, plot_frame)
    background.get_tk_widget().pack(side = TOP, expand = 1)

    learning_rate_frame = Frame(window)
    learning_rate_frame.pack(side = TOP)
    learning_rate_label = Label(learning_rate_frame, text = "Learning rate", font = ("Arial", 11, "normal"))
    learning_rate_label.pack(side = LEFT)
    learning_rate_entry = Entry(learning_rate_frame)
    # default learning rate
    learning_rate = 0.01 
    learning_rate_entry.pack(side = LEFT)
    learning_rate_entry.config(width = 20)

    Epoch_frame = Frame(window)
    Epoch_frame.pack()
    Epoch_label = Label(Epoch_frame, text = "Epoch", font = ("Arial", 11, "normal"))
    Epoch_label.pack(side = LEFT)
    Epoch_entry = Entry(Epoch_frame)
    # default epoch
    epoch = 0 
    Epoch_entry.pack(side = LEFT)
    Epoch_entry.config(width = 25)

    is_training_label = Label(window, text = f"Training iter : {0}", font = ("Arial", 11, "normal"))
    is_training_label.pack()
    front_dis_label = Label(window, text = f"front distance : {0}", font = ("Arial", 11, "normal"))
    front_dis_label.pack()
    right_dis_label = Label(window, text = f"right distance : {0}", font = ("Arial", 11, "normal"))
    right_dis_label.pack()
    left_dis_label = Label(window, text = f"left distance : {0}", font = ("Arial", 11, "normal"))
    left_dis_label.pack()

    time_train_label = Label(window, text = f"Training elapsed time : {0}", font = ("Arial", 11, "normal"))
    time_train_label.pack()
    time_simulate_label = Label(window, text = f"Simulating elapsed time : {0}", font = ("Arial", 11, "normal"))
    time_simulate_label.pack()

    button_plot_track = Button(window, text = "Start plotting", font = ("Arial", 10, "bold"), command = lambda: plot_track(background, figure))
    button_plot_track.pack()
    button_start_train = Button(window, text = "Start training", font = ("Arial", 10, "bold"), command = lambda: Train(learning_rate, epoch, is_training_label, window))
    button_start_train.pack()
    button_start_simulate = Button(window, text = "Start simulating", font = ("Arial", 10, "bold"), command = lambda: start_simulate(front_dis_label, right_dis_label, left_dis_label, background, window))
    button_start_simulate.pack()

    window.mainloop()

