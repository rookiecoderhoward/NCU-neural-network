from Hopfield import Hopfield
from tkinter import ttk
from tkinter import *
import numpy as np
import os

def combo_box_on_select(event):
    global selected_header
    selected_header.set(combo_box.get())
    print(f'The chosen header_file is {selected_header.get()}.')

unsigned_int = np.uint64
data_path = os.path.join(os.getcwd(), "Hopfield_dataset")
hw3_dataset_list = ['Basic', 'Bonus', 'Noise']

def plot_2D_data(arr, row, col):
    global figure
    figure = ''
    for i in range(row):
        temp = ''
        for j in range(col):
            idx = i * col + j
            # chage token '1' to '*'(more comfortable to look)
            if arr[idx] == 1.0: temp += '*'
            else: temp += ' '
        figure += (f'{temp}\n')
        print(temp)


def convert_to_bit(arr, row, col):
    temp_image_idx, temp_col_ct = 0, 0
    for token in data_read:
        if token == '\n': continue
        elif token == '1' or token == ' ':
            if token == '1': arr[temp_image_idx][temp_col_ct] = 1
            else: arr[temp_image_idx][temp_col_ct] = 0
            temp_col_ct += 1
        else: pass

        if temp_col_ct == row * col:
            temp_image_idx += 1
            temp_col_ct = 0
    
    return arr

# load data and convert to 0/1 bit
def load_data(file_name):
    global data_read
    file_name_path = os.path.join(data_path, file_name)
    with open(file_name_path, 'r') as f:
        data_read = f.read()

    image_num = len(data_read.split('\n\n')) # how many input data(# of image)
    col_num = len(data_read.split('\n')[0])
    row_num = int((len(data_read.split('\n\n')[0]) + 1) / (col_num + 1))
    
    train_arr = np.zeros((image_num, row_num * col_num), dtype = unsigned_int)
    train_arr = convert_to_bit(train_arr, row_num, col_num)

    return row_num, col_num, train_arr


def store_result(fig_buffer):
    with open(f'hw3_{selected_header.get()}_output.txt', 'w') as f:
        for i in range(len(fig_buffer)):
            if i != 0:
                print('---------------------------', file = f)
            print(f'Training data:\n\n{fig_buffer[i][0]}', file = f)
            print(f'Testing data:\n\n{fig_buffer[i][1]}', file = f)
            print(f'Recall result:\n\n{fig_buffer[i][2]}', file = f)
            # check this image's recall T or F
            print(f'Recall success? : {fig_buffer[i][0] == fig_buffer[i][2]}', file = f)


def Hop_Run():
    header = selected_header.get()
    fig_buffer = []
    # input training data to train hopfield
    row_num, col_num, train_arr = load_data(f'{header}_Training.txt')
    hop = Hopfield(train_arr.shape[0], train_arr.shape[1]) # image_num, row_num * col_num
    hop.hop_train(train_arr)
    # input testing data
    row_num, col_num, test_arr = load_data(f'{header}_Testing.txt')
    for i in range(hop.image_num):
        Training, Testing = train_arr[i], test_arr[i]
        recall = hop.hop_run(Testing)
        # show original data and the recall data
        print('---------------------------')
        print('Training data:\n')
        plot_2D_data(Training, row_num, col_num)
        temp_train_figure = figure
        print('\nTesting data:\n')
        plot_2D_data(Testing, row_num, col_num)
        temp_test_figure = figure
        print('\nRecall result:\n')
        plot_2D_data(recall, row_num, col_num)
        temp_recall_figure = figure
        # check this image's recall T or F
        print(f'\nRecall success? : {temp_train_figure == temp_recall_figure}')
        fig_buffer.append([temp_train_figure, temp_test_figure, temp_recall_figure])

    store_result(fig_buffer)


def main():
    Hop_Run()


# GUI
if __name__ == "__main__":

    window = Tk()
    window.title("HW3_Hopfield")
    window.geometry("500x250")

    combo_box_frame = Frame(window)
    combo_box_label = Label(window, text = "Choose file header", font = ("Arial", 11, "normal"))
    combo_box_label.pack()
    # ttk combobox, contents = hw3_dataset_list
    combo_box = ttk.Combobox(window, values = hw3_dataset_list)
    combo_box.pack(side = TOP)
    # binding event, trigger "on_select" function when the selection changes
    combo_box.bind("<<ComboboxSelected>>", combo_box_on_select)
    selected_header = StringVar() # combobox string buffer, store file_header

    button_run = Button(window, text = 'Run Hopfield', font = ("Arial", 10, "bold"), command = main)
    button_run.place(x = 200, y = 150)

    window.mainloop()

