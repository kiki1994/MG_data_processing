import torch
BatchSize = 128


def accuracy_text(result, label):
    accuracy = 0

    for i in range(0, BatchSize):
        norm_data = torch.sqrt(torch.sum(torch.pow(result,2),1))    #[128,1]
        norm_label = torch.sqrt(torch.sum(torch.pow(label,2),1))

        angle_value = torch.sum(torch.mul(result,label),1) / (norm_data * norm_label)
        accuracy += (torch.acos(angle_value) * 180) / 3.1415926

    accuracy_avg = accuracy / BatchSize
    return accuracy_avg


