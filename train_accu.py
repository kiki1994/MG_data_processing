# -*- coding: utf-8 -*-



import torch
import dataset
import model_ns
import loss_func

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


H5_address ='/disks/disk0/linyuqi/dataset/data_gaze/gaze_detection/h5/'
Save_model_address = '/disks/disk0/linyuqi/model/gaze_2eye_ARE/'
BatchSize = 128 
EPOCH = 80



AR_model = model_ns.AR_Net()
#if torch.cuda.device_count() > 1:
#    print("Let us use ", torch.cuda.device_count(),"GPUs!")
AR_model = nn.DataParallel(AR_model)
#if torch.cuda.is_available():
AR_model.cuda()

E_model = model_ns.E_Net()
E_model = nn.DataParallel(E_model)
E_model.cuda()

AR_down_model = model_ns.AR_Net_down()
AR_down_model = nn.DataParallel(AR_down_model)
AR_down_model.cuda()

AR_up_model = model_ns.AR_Net_up()
AR_up_model = nn.DataParallel(AR_down_model)
AR_up_model.cuda()

loss_f = loss_func.loss_f()
loss_f.cuda()

img_list_left,img_list_right = dataset.load_all_h5(H5_address)
#print(img_list_left[1])
#print(img_list_right[1])
train_Dataset = dataset.train_gaze_dataset(img_list_left,img_list_right)
test_Dataset = dataset.test_gaze_dataset(img_list_left,img_list_right)

train_loader = torch.utils.data.DataLoader(train_Dataset,shuffle=True,batch_size=BatchSize,num_workers=6)
test_loader = torch.utils.data.DataLoader(test_Dataset,shuffle=True,batch_size=BatchSize,num_workers=6)


L1_loss = nn.SmoothL1Loss().cuda()
#l1_loss = nn.MSELoss().cuda()
#optimizer = torch.optim.Adam(gaze_model.parameters(),lr=0.01)

optimizer_AR = torch.optim.SGD(AR_model.parameters(),lr=0.0001,momentum=0.9)
optimizer_E = torch.optim.SGD(E_model.parameters(),lr=0.0001,momentum=0.9)


def d_3(result):
    data = torch.zeros([result.size()[0], 3]) 
        
    for i in range(0, result.size()[0]):

        data[i][0] = (-1) * (torch.cos(result[i][0])) * (torch.sin(result[i][1]))

        data[i][1] = (-1) * (torch.sin(result[i][0]))

        data[i][2] = (-1) * (torch.cos(result[i][0])) * (torch.cos(result[i][1]))

    #tens_3 = torch.cat([data[:, 0], data[:, 1], data[:, 2]], 0)
    #tens1_3 = torch.unsqueeze(tens_3, 0)

    return data   ##size 128*3
    
def accuracy_text(result_l,result_r,label_l,label_r):
    accuracy_l = 0
    accuracy_r = 0
 
    norm_data_l = torch.sqrt(torch.sum(torch.pow(result_l,2),1))    #[128]
    norm_label_l = torch.sqrt(torch.sum(torch.pow(label_l,2),1))
    angle_value_l = torch.sum(torch.mul(result_l,label_l),1) / (norm_data_l * norm_label_l)
    accuracy_l += (torch.acos(torch.clamp(angle_value_l,min = -1,max = 1)) * 180)/3.1415926     #[128]
        
    norm_data_r = torch.sqrt(torch.sum(torch.pow(result_r,2),1))    #[128]
    norm_label_r = torch.sqrt(torch.sum(torch.pow(label_r,2),1))
    angle_value_r = torch.sum(torch.mul(result_r,label_r),1) / (norm_data_r * norm_label_r)
    accuracy_r += (torch.acos(torch.clamp(angle_value_r,min=-1,max=1)) * 180)/3.1415926     #[128]
    
    accuracy = (accuracy_l + accuracy_r)/2     #left and right average
    accuracy_avg = accuracy /result_l.size()[0]                      #[128]
    return accuracy_avg 
    
def train():
    AR_model.train()
    start = time.time()
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
        label_left = d_3(label_left).cuda()
        label_right = d_3(label_right).cuda()
        head_pose_left = d_3(head_pose_left).cuda()
        head_pose_right = d_3(head_pose_right).cuda()

#        print(label_left)
#        print(head_pose_left)
#        print(label_right)
#        print(head_pose_right)
#        print(image_left)
#        print(image_right)
        optimizer_AR.zero_grad() 
        optimizer_E.zero_grad()

        result_AR = AR_model(image_left, image_right,head_pose_left,head_pose_right)       ##output 128 x 4
        result_E = E_model(image_left, image_right)                                        ##output 128 x 2
#        result_d = AR_down_model(image_left, image_right)      
#        result_u = AR_up_model(image_left, image_right)                          
#        print(result_AR)
#        print(result_E)
 #       print(result_u)
#        print(result_d)
#        label = torch.cat([label_left,label_right],1)

        L_E, L_AR, L_AR2 = loss_f(result_AR[:,:3],result_AR[:,3:],label_left,label_right,result_E[:,0],result_E[:,1])
        loss = (L_E)
#        print(loss_AR)
        
#        print("--------")
        print(L_E)        
        print(L_AR)
        print(L_AR2)
        print("++++++")
#        label = torch.cat((label_left,label_right),1)
#        loss = L1_loss(result_AR,label)
        loss.backward()
        optimizer_AR.step()
        optimizer_E.step()
        
        elapsed = time.time() - start

        if i%20 ==0:
            print('num_of_batch = {}    train_loss={}      time = {:.2f}'.format(i,loss,elapsed))

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
        
        label_left = d_3(label_left).cuda()
        label_right = d_3(label_right).cuda()
        head_pose_left = d_3(head_pose_left).cuda()
        head_pose_right = d_3(head_pose_right).cuda()
        
#        print(label_left)
#        print(head_pose_left)
#        print(label_right)
#        print(head_pose_right)
        
#        test_AR = AR_model()
#        test_AR.load_state_dict(torch.load(os.path.join(Save_model_address, 'checkpoint_{}.tar'.format(main.epoch))))
#        prediction = test_AR(image_left, image_right,head_pose_left,head_pose_right)

        with torch.no_grad():
           result_AR = AR_model(image_left, image_right,head_pose_left,head_pose_right)       ##output 128 x 6
           result_E = E_model(image_left, image_right)                                     ##output 128 x 2
           
        loss_AR,loss_E,loss_AR2 = loss_f(result_AR[:,:3],result_AR[:,3:],label_left,label_right,result_E[:,0],result_E[:,1])
        loss = (loss_AR + loss_AR2 +loss_E)
        
#        label = torch.cat((label_left,label_right),1)
#        loss = L1_loss(result_AR,label)
#        print(result_AR)
#        print(result_E)
                
        if i%20 ==0:
            acc = accuracy_text(result_AR[:,:3],result_AR[:,3:],label_left,label_right)
            accuracy_avg = acc.mean()
            print('num_of_batch = {}    test_loss={}    accuracy={}   '.format(i,loss,accuracy_avg))
            
        #if i%100 ==0:
            #for i in range(0,20):
            print('left predit:({:5.4},{:5.4},{:5.4})   right predit:({:5.4},{:5.4},{:5.4})'.format(result_AR[i][0],result_AR[i][1],result_AR[i][2],result_AR[i][3],result_AR[i][4],result_AR[i][5]))

            print('label_left:({:5.4},{:5.4},{:5.4})     label_right:({:5.4},{:5.4},{:5.4})'.format(label_left[i][0],label_left[i][1],label_left[i][2],label_right[i][0],label_right[i][1],label_right[i][2]))
            
        
 #       if i%100 ==0:
 #           for i in range(0,20):
 #               print('image_number:{:2}   process answer {:+5.4},{:+5.4})   label: {:+5.4},{:+5.4})'.format(i,result[i][0],result[i][1],label[i][0],label[i][1]))

def main():
    for epoch in range(1,EPOCH):
        print('epoch:' + str(epoch) + '\n')
        train()
        test()
        torch.save({'epoch':epoch,
                    'state_dict':AR_model.state_dict(),},os.path.join(Save_model_address, 'checkpoint_{}.tar'.format(epoch)))
        # save only the parameters
#        test()


if __name__ =='__main__':
    main()       
