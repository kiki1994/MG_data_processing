
import torch
import torch.nn as nn
import torch.nn.functional as func



class loss_f(nn.Module):
    def __init__(self):
        super(loss_f, self).__init__()
        return
    def forward(self, result_l,result_r,label_l,label_r,prob_l,prob_r):
        error_up_l = torch.unsqueeze(torch.sum(torch.mul(result_l,label_l),1),1)        ##torch.Size([128, 1])
        error_d_l = torch.mul(torch.unsqueeze(torch.sum(torch.pow(result_l,2),1),1),
                              torch.unsqueeze(torch.sum(torch.pow(label_l,2),1),1))
        angle_l = torch.sum(torch.acos(error_up_l/error_d_l),0)/128

        error_up_r = torch.unsqueeze(torch.sum(torch.mul(result_r,label_r),1),1)
        error_d_r = torch.mul(torch.unsqueeze(torch.sum(torch.pow(result_r, 2), 1), 1),
                              torch.unsqueeze(torch.sum(torch.pow(label_r, 2), 1), 1))
        angle_r = torch.sum(torch.acos(error_up_r / error_d_r),0)/128                     ##torch.Size([1])

#        angle_r = torch.acos(torch.mm((result_r), torch.t(label_r))/(torch.norm(result_r) * torch.norm(label_r)))


        L_AR = 2 * torch.mul(angle_r, angle_l) / (angle_r + angle_l)      ####torch.Size([1])

        if angle_l <= angle_r:
            L_E128 = -(torch.mul(torch.acos(torch.sum(torch.mm(result_l, result_r),1)),(torch.log(prob_l))))   ##torch.Size([128, 1])
            L_E = torch.sum(L_E128,0)/128                                                         ##torch.Size([1])

            prob_l = torch.sum(prob_l,0)/128
            prob_r = torch.sum(prob_r,0)/128
            omega = (1 + prob_l - prob_r) / 2                                             ##torch.Size([128, 1])

        else:
            L_E128 = -(torch.mul(torch.acos(torch.sum(torch.mm(result_l, result_r),1)),(torch.log(prob_r))))
            L_E = torch.sum(L_E128, 0) / 128

            prob_l = torch.sum(prob_l,0)/128
            prob_r = torch.sum(prob_r,0)/128
            omega = (1 - prob_l + prob_r) / 2

        L_AR2 = omega * L_AR + (1 - omega) * 0.1 * ((angle_r + angle_l) / 2)  ### The new one

        return L_AR, L_E, L_AR2

