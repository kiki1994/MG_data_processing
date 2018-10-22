# -*- coding: utf-8 -*-

import h5py
import numpy
import os
import os.path
import torch.utils.data as data

def load_all_h5(root):
    left_list = []
    right_list = []
    for person in [root + a + '/' for a in os.listdir(root)]:
        for day in [person + a + '/' + 'left/'for a in os.listdir(person)]:
            for left_file in os.listdir(day):
                if(left_file.split('.')[-1] == 'h5'):
                   left_list.append(day + left_file)
        for day in [person + a + '/' + 'right/'for a in os.listdir(person)]:
            for right_file in os.listdir(day):
                if(right_file.split('.')[-1] == 'h5'):
                    right_list.append(day + right_file)
    return left_list,right_list


class train_gaze_dataset(data.Dataset):
    def __init__(self,img_list_left,img_list_right):
        self.h5_path_list_left= img_list_left[:200000]
        self.h5_path_list_right= img_list_right[:200000]

    def __getitem__(self, index):
        filename_l = self.h5_path_list_left[index]
        f_l = h5py.File(filename_l, 'r')
        img_l = f_l['data'].value
        head_pose_l = f_l['labels'].value[0, 2:]
        label_l = f_l['labels'].value[0, :2]
    
        filename_r = self.h5_path_list_right[index]
        f_r = h5py.File(filename_r, 'r')
        img_r = f_r['data'].value
        head_pose_r = f_r['labels'].value[0, 2:]
        label_r = f_r['labels'].value[0, :2]
        return img_l, head_pose_l, label_l, img_r, head_pose_r, label_r

    def __len__(self):
        return len(self.h5_path_list_left)
        
class test_gaze_dataset(data.Dataset):
    def __init__(self,img_list_left,img_list_right):
        self.h5_path_list_left = img_list_left[200000:]
        self.h5_path_list_right = img_list_right[200000:]
    def __getitem__(self,index):
        filename_l = self.h5_path_list_left[index]
        f_l = h5py.File(filename_l, 'r')
        img_l = f_l['data'].value
        head_pose_l = f_l['labels'].value[0, 2:]
        label_l = f_l['labels'].value[0, :2]
        
        filename_r = self.h5_path_list_right[index]
        f_r = h5py.File(filename_r, 'r')
        img_r = f_r['data'].value
        head_pose_r = f_r['labels'].value[0, 2:]
        label_r = f_r['labels'].value[0, :2]
        return img_l, head_pose_l, label_l, img_r, head_pose_r, label_r

    def __len__(self):
        return len(self.h5_path_list_left)
        
