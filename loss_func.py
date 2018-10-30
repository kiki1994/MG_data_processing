
import torch
import torch.nn as nn
import torch.nn.functional as func



class loss_f(nn.Module):
    def __init__(self):
        super(loss_f, self).__init__()
        return
    def forward(self, result_l,result_r,label_l,label_r,prob_l,prob_r):
        
        error_up_l = torch.sum(torch.mul(result_l,label_l),1)        ##torch.Size([128])
        error_d_l = torch.mul(torch.sqrt(torch.sum(torch.pow(result_l,2),1)),     ##torch.Size([128])
                             torch.sqrt(torch.sum(torch.pow(label_l,2),1)))
        angle128_l = torch.acos(torch.clamp((error_up_l/error_d_l),min=-1,max=1))
               

        error_up_r = torch.sum(torch.mul(result_r,label_r),1)
        error_d_r = torch.mul(torch.sqrt(torch.sum(torch.pow(result_r, 2), 1)),
                              torch.sqrt(torch.sum(torch.pow(label_r, 2), 1)))
        angle128_r = torch.acos(torch.clamp((error_up_r / error_d_r),min=-1,max=1))
        
        L_AR_ = 2 * torch.mul(angle128_r, angle128_l) / (angle128_r + angle128_l + 1e-4)      ####torch.Size([128])
        L_AR = torch.sum(L_AR_,0)                                                                ####scale
        
        mat_b = angle128_l <= angle128_r
        mat_b = mat_b.float().cuda()
 
        L_E128_up = torch.sum(torch.mul(result_l,result_r),1)                                ##torch.Size([128])
        L_E128_down = torch.mul(torch.sqrt(torch.sum(torch.pow(result_l,2),1)),           ##torch.Size([128])
                             torch.sqrt(torch.sum(torch.pow(result_r,2),1)))
                             
#        L_E128 = -(mat_b * torch.mul(torch.acos(L_E128_up/L_E128_down),torch.log(prob_l)) + (1 - mat_b) * torch.mul(torch.acos(L_E128_up/L_E128_down),(torch.log(prob_r))))
        L_E128 = -(mat_b * torch.mul(torch.acos(L_E128_up),torch.log(prob_l)) + (1 - mat_b) * torch.mul(torch.acos(L_E128_up),(torch.log(prob_r))))
        L_E = torch.sum(L_E128, 0) 

        omega = (1 + (2 * mat_b -1) * prob_l + (1 - 2 * mat_b) * prob_r) / 2
        
        L_AR2_ = torch.mul(omega, L_AR_) + 0.1*torch.mul((1 - omega), ((angle128_r + angle128_l) / 2))  ### The new one
        L_AR2 = torch.sum(L_AR2_,0)
         
        return L_E,L_AR,L_AR2
