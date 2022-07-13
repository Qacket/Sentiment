import random

import numpy as np
import torch

from sen import SEN
from train import train_a_vae, train_t_vae, train_mymodel

seed = 1
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)



model_dir = './model/'

epochs = 500
batch_size = 64
lr = 0.001
annotator_dim = 203

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Load the data
train_data = SEN(data_dir="./datasets", create_data=False, max_sequence_length=70)
# Batchify the data
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)


task_voc_size = train_data.vocab_size



if __name__ == '__main__':


    # 训练工人vae
    # train_a_vae(device, lr, epochs, train_loader, model_dir, annotator_dim)
    # 训练任务vae
    # train_t_vae(device, lr, epochs, train_loader, model_dir, task_voc_size, batch_size)

    a_vae = torch.load('/mnt/4T/scj/sentiment/Mymodel_generate/a_vae')
    t_vae = torch.load('/mnt/4T/scj/sentiment/Mymodel_generate/t_vae')

    a_vae.trainable = False
    t_vae.trainable = False

    train_mymodel(a_vae, t_vae, device, lr, epochs, train_loader, model_dir, annotator_dim, batch_size)
