# -*- coding: utf-8 -*-



import torch
import dataset
import model
import loss_func

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

#import matplotlib.pyplot as plt
#plt.switch_backend('agg')



H5_address ='/home/lyq/caffe-master/data_gaze/gaze_detection/h5/'
Save_model_address = '/disks/disk0/linyuqi/model/gaze_2eye_ARE/'
BatchSize = 128
EPOCH = 30



AR_model = model.AR_Net()
#if torch.cuda.device_count() > 1:
#    print("Let us use ", torch.cuda.device_count(),"GPUs!")
AR_model = nn.DataParallel(AR_model)
#if torch.cuda.is_available():
AR_model.cuda()

E_model = model.E_Net()
E_model = nn.DataParallel(E_model)
E_model.cuda()

img_list_left,img_list_right = dataset.load_all_h5(H5_address)
#print(img_list_left[1])
#print(img_list_right[1])
train_Dataset = dataset.train_gaze_dataset(img_list_left,img_list_right)
test_Dataset = dataset.test_gaze_dataset(img_list_left,img_list_right)

train_loader = torch.utils.data.DataLoader(train_Dataset,shuffle=True,batch_size=BatchSize,num_workers=6)
test_loader = torch.utils.data.DataLoader(test_Dataset,shuffle=True,batch_size=BatchSize,num_workers=6)


l1_loss = nn.SmoothL1Loss().cuda()
l1_loss = nn.MSELoss().cuda()
#optimizer = torch.optim.Adam(gaze_model.parameters(),lr=0.01)

optimizer_AR = torch.optim.SGD(AR_model.parameters(),lr=0.001,momentum=0.9)
optimizer_E = torch.optim.SGD(E_model.parameters(),lr=0.001,momentum=0.9)


def d_3(result):
    data = torch.zeros([128, 3])
    if i >=128:
        rang = BatchSize
    else:rang = i
    for i in range(0, BatchSize):

        data[i,0] = (-1) * (torch.cos(result[i][0])) * (torch.sin(result[i][1]))

        data[i][1] = (-1) * (torch.sin(result[i][0]))

        data[i][2] = (-1) * (torch.cos(result[i][0])) * (torch.cos(result[i][1]))

    #tens_3 = torch.cat([data[:, 0], data[:, 1], data[:, 2]], 0)
    #tens1_3 = torch.unsqueeze(tens_3, 0)

    return data   ##size 128*3

def normal_d(data_3d):
    data_3D = d_3(data_3d)

    for i in range(0, BatchSize):

        normal_data = torch.sqrt(torch.pow(data_3D[i][0],2)+torch.pow(data_3D[i][1],2)+torch.pow(data_3D[i][2],2))

    return normal_data



def accuracy_text(result, label):
    accuracy = 0

    for i in range(0, BatchSize):
        result_3D = d_3(result)
        norm_data = np.sqrt(data_x * data_x + data_y * data_y + data_z * data_z)

        label_x = (-1) * (np.cos(label[i][0])) * (np.sin(label[i][1]))
        label_y = (-1) * (np.sin(label[i][0]))
        label_z = (-1) * (np.cos(label[i][0])) * (np.cos(label[i][1]))
        norm_label = np.sqrt(label_x * label_x + label_y * label_y + label_z * label_z)

        angle_value = (data_x * label_x + data_y * label_y + data_z * label_z) / (norm_data * norm_label)
        accuracy += (np.arccos(angle_value) * 180) / 3.1415926

    accuracy_avg = accuracy / BatchSize
    return accuracy_avg

def train():
    AR_model.train()
    for i ,(image_left,head_pose_left,label_left,image_right,head_pose_right,label_right) in enumerate(train_loader):
        image_left = image_left.squeeze(1)
        image_left = image_left.cuda()
        head_pose_left = head_pose_left.cuda()
        label_left = label_left.cuda()

        image_right = image_right.squeeze(1)
        image_right = image_right.cuda()
        head_pose_right = head_pose_right.cuda()
        label_right = label_right.cuda()

#Turn to 3D
        label_left = d_3(label_left)
        label_right = d_3(label_right)
        head_pose_left = d_3(head_pose_left)
        head_pose_right = d_3(head_pose_right)

#        print(label_left.size())
#        print(head_pose_left.size())

        optimizer_AR.zero_grad() 
        optimizer_E.zero_grad()

        result_AR = AR_model(image_left, image_right,head_pose_left,head_pose_right)       ##output 128 x 4
        result_E = E_model(image_left, image_right)                                     ##output 128 x 2
#        print(result_AR.size())
#        print(result_E.size())
#        label = torch.cat([label_left,label_right],1)

        loss_AR,loss_E,loss_AR2 = loss_func.loss_f(result_AR[:,:3],result_AR[:,3:],label_left,label_right,result_E[:,0],result_E[:,1])
        loss = (loss_AR + loss_AR2 +loss_E)/3
        loss.backward()
        optimizer_AR.step()
        optimizer_E.step()
#        label = (label_right+label_left)/2
#        loss_l = l1_loss(result[:,:2],label_left)
#        loss_r = l1_loss(result[:,2:],label_right)
#        print(loss_l)
#        optimizer.zero_grad()
#        loss = l1_loss(result,label)
#        loss_l.backward()
#        loss.backward()
#        optimizer_AR.step()


        if i%20 ==0:
            print('num_of_batch = {}    train_loss={}'.format(i,loss))

def test():
    AR_model.eval()
    for i ,(image_left,head_pose_left,label_left,image_right,head_pose_right,label_right) in enumerate(test_loader):
        image_left = image_left.squeeze(1)
        image_left = image_left.cuda()
        head_pose_left = head_pose_left.cuda()
        label_left = label_left.cuda()

        image_right = image_right.squeeze(1)
        image_right = image_right.cuda()
        head_pose_right = head_pose_right.cuda()
        label_right = label_right.cuda()
        
        test_AR = AR_model()
        test_AR.load_state_dict(os.path.join(Save_model_address, 'checkpoint_{}.tar'.format(main.epoch)))
        prediction = test_AR(image_left, image_right,head_pose_left,head_pose_right)

#        with torch.no_grad():
#           result =AR_model(image_left, image_right,head_pose_left,head_pose_right)
#        label = torch.cat([label_left,label_right],1)
#        loss = l1_loss(result,label)
        
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
        
        if i%20 ==0:
            print('left predit:{:+5.4},{:5.4}   right predit:{:+5.4},{:5.4}').format(prediction[i][0],prediction[i][1],prediction[i][2],prediction[i][3])

            print('label_left:{:5.4}{:5.4}     label_right:{:5.4}{:5.4}').format(label_left,label_right)
#            acc = accuracy_text(prediction,label)
#            print('num_of_batch = {}    test_loss={}    accuracy={}   '.format(i,loss,acc))

 #       if i%100 ==0:
 #           for i in range(0,20):
 #               print('image_number:{:2}   process answer {:+5.4},{:+5.4})   label: {:+5.4},{:+5.4})'.format(i,result[i][0],result[i][1],label[i][0],label[i][1]))

def main():
    for epoch in range(1,EPOCH):
        print('epoch:' + str(epoch) + '\n')
        train()
#        test()
        torch.save({'epoch':epoch,
                    'state_dict':AR_model.state_dict(),},os.path.join(Save_model_address, 'checkpoint_{}.tar'.format(epoch)))
        # save only the parameters
        test()


if __name__ =='__main__':
    main()       
