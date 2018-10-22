def d_3 (result):
    data[:,0] = (-1) *(torch.cos(result[i][0])) *(np.sin(result[i][1]))

    data[:,1] = (-1) *(np.sin(result[i][0]))

    data[:,2] = (-1) *(np.cos(result[i][0])) * (np.cos(result[i][1]))

    tens_3 =torch.cat([data[:,0],data[:,1],data[:,2]],0)
    tens1_3 = torch.unsqueeze(tens_3,0)

    return  tens1_3


def accuracy_text(result, label):
    accuracy = 0

    for i in range(0, BatchSize):
        data_x = (-1) * (np.cos(result[i][0])) * (np.sin(result[i][1]))
        data_y = (-1) * (np.sin(result[i][0]))
        data_z = (-1) * (np.cos(result[i][0])) * (np.cos(result[i][1]))
        norm_data = np.sqrt(data_x * data_x + data_y * data_y + data_z * data_z)

        label_x = (-1) * (np.cos(label[i][0])) * (np.sin(label[i][1]))
        label_y = (-1) * (np.sin(label[i][0]))
        label_z = (-1) * (np.cos(label[i][0])) * (np.cos(label[i][1]))
        norm_label = np.sqrt(label_x * label_x + label_y * label_y + label_z * label_z)

        angle_value = (data_x * label_x + data_y * label_y + data_z * label_z) / (norm_data * norm_label)
        accuracy += (np.arccos(angle_value) * 180) / 3.1415926

    accuracy_avg = accuracy / BatchSize
    return accuracy_avg