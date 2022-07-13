# import torch
#
# from data import deal_data
# from utils import idx_to_word
#
# device, train_iter, task_vocab, task_voc_size, task_pad_idx, task_sos_idx, task_eos_idx, task_unk_idx = deal_data()
#
# mymodel = torch.load('/mnt/4T/scj/sentiment/Mymodel_generate/task_vae_13train2016')
# mymodel.to(device)
# mymodel.eval()
# samples, z = mymodel.inference()
# print('----------SAMPLES----------')
# print("---------------trg")
# print(samples, z)
# for j in range(10):
#     print("--------1111")
#     trg = idx_to_word(samples[j], task_vocab, task_eos_idx, task_pad_idx)
#     print(trg)
#     print("--------11112222")
import torch

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

model = torch.load('./model/mymodel3')
z1 = torch.randn((1, 1, 200)).to(device)

print(z1)
sos = "<sos>"
sample1 = model.t_vae.inference(10, sos, z1)

print(sample1)


