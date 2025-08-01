#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from multiprocessing import Process, Queue
import queue
import sys
import numpy as np
from spatialmath import SE3
from scripts.mcmc import mcmc_callback
import signal
import magnum as mn



def mcmc_process(input_queue, output_queue):

    while True:

        msg = input_queue.get()
        while not input_queue.empty():
            msg = input_queue.get_nowait()  # overwrite with newer messages

        
        response = mcmc_callback(msg)
        output_queue.put(response)  # send response back



def get_osmm_arrows(basis_vectors, rot_angle, scale = 3, threshold=0.2):

    
    arrows = []

    for i, basis in enumerate(basis_vectors):
        if i >=3:
            sign = -1
            
        else:
            sign = 1


        magnitude = basis*scale*sign

        #Clip to threshold
        if abs(magnitude) > threshold:
            magnitude = magnitude/abs(magnitude)*threshold

        if i%3 == 0:    #x
            color = mn.Color3(0.1, 0.1, 0.9)
            vec = mn.Vector3(magnitude, 0, 0)

        elif i%3 == 1:  #y
            color = mn.Color3(0.1, 0.9, 0.1)
            vec = mn.Vector3(0, magnitude, 0)

        else:       #z
            color = mn.Color3(0.9, 0.1, 0.1)
            vec = mn.Vector3(0, 0, magnitude)


        vec = rot_angle.transform_vector(vec)

        arrows.append((vec, color))

    return arrows


def get_movement_feasability_vector(xr_translation_in_arm_frame, basis_vectors, rot_angle, scale=3, threshold=0.2):
    
    
    norm = np.linalg.norm([xr_translation_in_arm_frame[0], xr_translation_in_arm_frame[1], xr_translation_in_arm_frame[2]])

    # xr_position should not be at the end-effector position
    if norm < 0.1:
        return None

    #normalize the displacement vector
    xr_translation_in_arm_frame = xr_translation_in_arm_frame / norm * threshold

    xr_pose_transformed = rot_angle.transform_vector(xr_translation_in_arm_frame)

    x = max(xr_pose_transformed[0]*basis_vectors[0], -xr_pose_transformed[0]*basis_vectors[3])
    y = max(xr_pose_transformed[1]*basis_vectors[1], -xr_pose_transformed[1]*basis_vectors[4])
    z = max(xr_pose_transformed[2]*basis_vectors[2], -xr_pose_transformed[2]*basis_vectors[5])


    if xr_pose_transformed[0] < 0:
        x = -x
    if xr_pose_transformed[1] < 0:
        y = -y
    if xr_pose_transformed[2] < 0:
        z = -z
    
    color_norm = np.linalg.norm([x,y,z])
    
    
    green = color_norm*100/2
    red = 1 - (color_norm*100/2)

    return (xr_translation_in_arm_frame, mn.Color3(red, green, 0.0))





class MCMC_Subprocess():
    def __init__(self):
        
        #initialize basis vectors for 
        self.left_basis_vectors = np.array([0]*6)
        self.right_basis_vectors = np.array([0]*6)

        self.left_previous_seed = np.array([0.0]*7)
        self.right_previous_seed = np.array([0.0]*7)

        self.SE3 = SE3()

        self.to_child_left = Queue()
        self.from_child_left = Queue()
        self.to_child_right = Queue()
        self.from_child_right = Queue()

        self.process_left = Process(target=mcmc_process, args=(self.to_child_left, self.from_child_left))
        self.process_right = Process(target=mcmc_process, args=(self.to_child_right, self.from_child_right))
        self.process_left.start()
        self.process_right.start()

        signal.signal(signal.SIGINT, self._make_signal_handler())

    
    def _make_signal_handler(self):
        def handler(sig, frame):
            print("SIGINT received. Terminating child processes...")
            self.process_left.terminate()
            self.process_right.terminate()
            self.process_left.join()
            self.process_right.join()
            sys.exit(0)
        return handler

    def update_basis(self):

        try:
            basis_vectors = self.from_child_right.get_nowait()

            basis_vectors = [-basis_vectors[1], basis_vectors[2], -basis_vectors[0], 
                     -basis_vectors[4], basis_vectors[5], -basis_vectors[3]]
            
            self.right_basis_vectors = np.array(basis_vectors)

        except queue.Empty:
            pass
        
        try:
            basis_vectors = self.from_child_left.get_nowait()

            basis_vectors = [-basis_vectors[1], basis_vectors[2], -basis_vectors[0], 
                            -basis_vectors[4], basis_vectors[5], -basis_vectors[3]]
            
            self.left_basis_vectors = np.array(basis_vectors)
            
        except queue.Empty:
            pass


    def _send_seed(self, q, prev_q, child):
        q = np.array(q)
        if np.any(np.abs(q - prev_q) > np.deg2rad(1.5)):
            child.put(q)
            return q
        else:
            return prev_q

    def set_seed(self, q, arm='right_arm'):

        if arm == 'left_arm':
            self.left_previous_seed = self._send_seed(q, self.left_previous_seed, self.to_child_left)
        elif arm == 'right_arm':
            self.right_previous_seed = self._send_seed(q, self.right_previous_seed, self.to_child_right)

