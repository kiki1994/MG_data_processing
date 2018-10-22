import h5py
import numpy
import os
import os.path
import torch.utils.data as data

H5_path = r'F:\datebase\zhang_xucong_paper\MPIIGaze\MPIIGaze\Data\h5/'


def load_all_h5(root):
    left_list = []
    right_list = []
    for person in [root + a + '/' for a in os.listdir(root)]:
        for day in [person + a + '/' + 'left/'for a in os.listdir(person)]:
            for left_file in os.listdir(day):
                if(left_file.split('.')[-1] == 'h5'):
                    list.append(day + left_file)
        for day in [person + a + '/' + 'right/'for a in os.listdir(person)]:
            for right_file in os.listdir(day):
                if(right_file.split('.')[-1] == 'h5'):
                    right_list.append(day + right_file)
    return left_list,right_list

def main():
    img_list_left, img_list_right = load_all_h5(H5_path)
    print(img_list_left[2])