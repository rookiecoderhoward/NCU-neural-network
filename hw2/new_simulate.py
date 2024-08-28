import numpy as np
import matplotlib.patches as patch
from Training_Model import *
from Kmeans import *
from sympy import *
from sympy import Segment, Point

class Car:
    def __init__(self, x, y, phi):
        self.x, self.y = x, y
        self.phi = phi
        self.length = 3
        self.simulate_result = []
        self.coordinate = np.array([
            [-6,-3], [-6,22], [18,22], [18,50], [30,50], [30,10], [6,10], [6,-3], [-6,-3]
        ])
    
    def euclidean_distance(self, point_1, point_2):
        dis = np.linalg.norm(point_1 - point_2, axis = 0)
        return dis

    def renew_pos(self, theta):
        self.x = self.x + np.cos(np.radians(self.phi + theta)) + (np.sin(np.radians(theta)) * np.sin(np.radians(self.phi)))
        self.y = self.y + np.sin(np.radians(self.phi + theta)) - (np.sin(np.radians(theta)) * np.cos(np.radians(self.phi)))
        self.phi = self.phi - (np.arcsin(2 * np.sin(np.radians(theta)) / self.length) * (180 / np.pi))  # change radian to degree   

    def line_intersection(self, sensor_point):
        coord = self.coordinate
        line_1 = Segment(Point(self.x, self.y), Point(sensor_point[0], sensor_point[1]))
        # initialize min_dis to infinte
        min_dis = np.inf
        for i in range(coord.shape[0]-1):
            line_2 = Segment(Point(coord[i][0], coord[i][1]), Point(coord[i+1][0], coord[i+1][1]))
            intersection_result = line_1.intersection(line_2)
            # Check whether there is an intersection or not
            if intersection_result:
                inter_point = intersection_result[0] # find the intersection point
                inter_x, inter_y = float(inter_point[0]), float(inter_point[1])
                point_1 = np.array([self.x, self.y])
                point_2 = np.array([inter_x, inter_y])
                cur_dis = self.euclidean_distance(point_1, point_2)
                min_dis = min(min_dis, cur_dis)
                # print(f'min_dis {min_dis}')

        return min_dis

    def rotate_car(self, rotate_theta):
        rotate_theta = np.radians(rotate_theta)
        cosine, sine = np.cos(rotate_theta), np.sin(rotate_theta)
        x = 250 * cosine - 0 * sine
        y = 250 * sine + 0 * cosine
        return np.array([x, y])

    def Draw(self, background, figure_plot):
        figure_plot.scatter(self.x, self.y, c = 'green', s = 11)
        car = patch.Circle((self.x, self.y), radius = 3, linewidth = 0.6, fill = False, color = 'saddlebrown')
        figure_plot.add_patch(car)
        background.draw()

    def Start(self, training_model, four_six, front_dis_label, right_dis_label, left_dis_label, background, figure_plot, window):
        border = 37 - self.length
        while self.y < border:
            front_p = self.rotate_car(self.phi)
            front_dis = self.line_intersection(front_p)
            right_p = self.rotate_car(self.phi - 45)
            right_dis = self.line_intersection(right_p)
            left_p = self.rotate_car(self.phi + 45)
            left_dis = self.line_intersection(left_p)
            # Check whether it hits the track || finished
            if front_dis < 3 or right_dis < 3 or left_dis < 3: break

            if four_six == 4:
                active, Fx = training_model.predict_output(np.array([front_dis, right_dis, left_dis]))
                Fx = Fx * 70 - 32
                self.simulate_result.append([front_dis, right_dis, left_dis, Fx])
            elif four_six == 6:
                active, Fx = training_model.predict_output(np.array([self.x, self.y, front_dis, right_dis, left_dis]))
                Fx = Fx * 70 - 32
                self.simulate_result.append([self.x, self.y, front_dis, right_dis, left_dis, Fx])
            
            self.renew_pos(Fx)
            print(self.x, self.y, self.phi)

            front_dis_label.configure(text = f'front distance : {front_dis}')
            right_dis_label.configure(text = f'right distance : {right_dis}')
            left_dis_label.configure(text = f'left distance : {left_dis}')
            window.update()
            self.Draw(background, figure_plot)

        return self.simulate_result

