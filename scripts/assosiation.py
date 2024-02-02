
import sys
import random
import time
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import linear_sum_assignment
from numpy import array

from itertools import combinations
import copy

VIEWPOINT_BEFORE_START = 111
VIEWPOINT_AFTER_FINISH = 999
VIEWPOINT_ID_START  = 0
VIEWPOINT_ID_END    = 53



def calc_distance(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)**0.5
    

class KalmanFilteredFruit:
    def __init__(self, detected_fruit, detection_id, fruit_id):
        self.initial_detected_fruit = detected_fruit
        self.n_fruit = detected_fruit.n_fruit

        self.state              = self.initial_detected_fruit.xyz
        self.uncertainty        = np.identity(3)/100 
        self.process_noise      = np.diag([0.001, 0.001, 0.001])
        self.measurement_noise  = np.diag([0.05, 0.05, 0.05])  #
        
        self.cnt_observed       = 1
        self.detection_id       = detection_id
        self.fruit_id           = fruit_id

    def predict(self):
        self.uncertainty += self.process_noise
        self.cnt_observed += 1

    def update(self, measurement):
        measurement = np.array(measurement)
        S = self.uncertainty + self.measurement_noise
        kalman_gain = np.dot(self.uncertainty, np.linalg.inv(S))
        self.state = self.state + np.dot(kalman_gain, (measurement - self.state))
        self.uncertainty = np.dot((np.identity(3) - kalman_gain), self.uncertainty)

class FruitTracker:
    def __init__(self):
        # managing detected fruits
        self.cur_fruit_id = 0
        self.detection_cnt_at_current_viewpoint = 1 # viewpoint
        self.viewpoint_infos = {}
        
        for viewpoint_id in range(VIEWPOINT_ID_START, VIEWPOINT_ID_END+1):
            self.viewpoint_info[viewpoint_id] = {'image_captured':0, 'tracked_fruits': {}}
        
        self.threshold_DA = 0.12
        self.thresh_fruit_cluster_dist = 0.7

    def update_tracked_fruits(self, detected_fruits, target_viewpoint_info):
        target_viewpoint, target_viewpoint_valid = target_viewpoint_info
        if target_viewpoint == VIEWPOINT_BEFORE_START:
            return
        if not VIEWPOINT_ID_START <= target_viewpoint <= VIEWPOINT_ID_END:
            raise ValueError(f"[Detector] target_viewpoint must be between {VIEWPOINT_ID_START} and {VIEWPOINT_ID_END}")
        if not target_viewpoint_valid:
            return
        

        self.detection_cnt_dict[target_viewpoint]['image_captured'] += 1   
        if len(detected_fruits) == 0:
            return

        for i in range(len(detected_fruits)):
            detected_fruit = detected_fruits[i]

            fruit_pos = detected_fruit.xyz

            min_distance = float('inf')
            matched_id = None

            for tracked_fruits in self.detection_cnt_dict[target_viewpoint]['tracked_fruits']:
                distance = calc_distance(tracked_fruits.state, fruit_pos) 
                # if any fruit is close to preobserved fruit, update the preobserved fruit
                if distance < min_distance:
                    min_distance = distance
                    matched_id = tracked_fruits.fruit_id

            if min_distance < self.threshold_DA: ## twtw
                self.tracked_fruits[matched_id].predict() # ok for static model
                self.tracked_fruits[matched_id].update(fruit_pos)
            else:
                new_kalman_filtered_fruit = KalmanFilteredFruit(detected_fruit, 
                                                                target_viewpoint,
                                                                self.cur_fruit_id)
                self.detection_cnt_dict[target_viewpoint]['tracked_fruits'][self.cur_fruit_id] \
                    = new_kalman_filtered_fruit

                self.cur_fruit_id += 1 #

            # self.last_detection_time = fruit_time
        
    def clean_fruits(self):
        valid_detection_ratio = 0.6
        
        # find fruits that are not detected for a long time
        for viewpoint_id, viewpoint_info in self.viewpoint_infos.items():
            tracked_fruits = viewpoint_info['tracked_fruits']

            for fruit_id, tracked_fruit in tracked_fruits.items():
                n_fruit_detected = tracked_fruit.cnt_observed
                if n_fruit_detected < viewpoint_info['image_captured']*valid_detection_ratio:
                    # remove key from dict
                    tracked_fruits.pop(fruit_id)
            
            viewpoint_info['tracked_fruits'] = tracked_fruits


        # find fruits that are close to each other
        for viewpoint_id, viewpoint_info in self.viewpoint_infos.items():
            tracked_fruits = viewpoint_info['tracked_fruits']
            clusters = []

            while len(tracked_fruits):
                ks = tracked_fruits.keys()
                clusters.append(tracked_fruits[ks[0]])
                tracked_fruits.pop(ks[0])

                cur_cluster = tracked_fruits[-1]
                for fruit_id, tracked_fruit in tracked_fruits.items():
                    fruit_pos_1 = cur_cluster.state
                    fruit_pos_2 = tracked_fruit.state

                    distance = calc_distance(fruit_pos_1, fruit_pos_2)
                    
                    if distance < self.threshold_DA+0.01:
                        cur_cluster.state = list((np.array(fruit_pos_1) + np.array(fruit_pos_2))/2)
                        tracked_fruits.pop(fruit_id)
            
            viewpoint_info['tracked_fruits'] = tracked_fruits
            

        # cluster fruits in a plant            
        for viewpoint_id, viewpoint_info in self.viewpoint_infos.items():
            fruits_to_be_clustered = copy.deepcopy(viewpoint_info['tracked_fruits'])
            representative_fruits = []

            while len(fruits_to_be_clustered):
                ks = fruits_to_be_clustered.keys()
                representative_fruits.append(fruits_to_be_clustered[ks[0]])
                fruits_to_be_clustered.pop(ks[0])

                cur_cluster = fruits_to_be_clustered[-1]
                for fruit_id, tracked_fruit in fruits_to_be_clustered.items():
                    fruit_pos_1 = cur_cluster.state
                    fruit_pos_2 = tracked_fruit.state

                    distance = calc_distance(fruit_pos_1, fruit_pos_2)
                    
                    if distance < self.thresh_fruit_cluster_dist:
                        cur_cluster.state = list((np.array(fruit_pos_1) + np.array(fruit_pos_2))/2)
                        cur_cluster.n_fruit += tracked_fruit.n_fruit
                        fruits_to_be_clustered.pop(fruit_id)
            viewpoint_info['representative_fruits'] = representative_fruits
        
        # rearrange fruits to return
        plants_infos = []
        for viewpoint_id, viewpoint_info in self.viewpoint_infos.items():
            representative_fruits = copy.deepcopy(viewpoint_info['representative_fruits'])
            if not 0 <= representative_fruits <= 3:
                raise ValueError(f"[Tracker] # of plants must be between 0 and 3")
            
            # sort with y coordinate
            representative_fruits.sort(key=lambda x: x.state[1])

            # count total # of fruits
            total_n_fruit = 0
            for representative_fruit in representative_fruits:
                total_n_fruit += representative_fruit.n_fruit
            
            plants_infos.append((viewpoint_id, total_n_fruit, representative_fruits))

        # sort with viewpoint id
        plants_infos.sort(key=lambda x: x[0])
        
        return plants_infos
        # list of (viewpoint id, total # of plant, (plants))