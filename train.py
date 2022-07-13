import numpy as np
import torch
from torch import optim, nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
from loss import T_VAE_Loss, A_VAE_Loss
from mymodel import T_VAE, A_VAE, my_model
from utils import get_batch

def train_a_vae(device, lr, epochs, train_loader, model_dir, annotator_dim):
    # # 初始化工人vae
    a_vae = A_VAE(
        E_in=annotator_dim,
        hidden_size=20,
        latent_size=8,
        D_out=annotator_dim,
        device=device
    )
    a_vae = a_vae.to(device)
    A_loss = A_VAE_Loss()
    optimizer = optim.Adam(a_vae.parameters(), lr=lr)
    writer = SummaryWriter('logs/events.out.a_vae')
    for epoch in range(epochs):
        for batch in tqdm(train_loader):
            # 获取数据
            annotator_id, answer, task, target, task_lengths = get_batch(batch)
            # 工人
            annotator_id = np.array(annotator_id).astype(dtype=int)
            annotator_id = torch.from_numpy(annotator_id).to(device)
            annotator_inputs = F.one_hot(annotator_id, annotator_dim).type(torch.float32)  # 工人input
            # 工人vae
            a_output, a_mean, a_logv, a_z = a_vae(annotator_inputs)

            # 工人 loss
            a_mloss, a_KL_loss, a_recon_loss = A_loss(mu=a_mean, logvar=a_logv, recon_x=a_output, x=annotator_inputs)

            a_mloss.backward()

            optimizer.step()

            optimizer.zero_grad()

        writer.add_scalar(tag='a_KL_loss', scalar_value=a_KL_loss.data.item(), global_step=epoch)
        writer.add_scalar(tag='a_recon_loss', scalar_value=a_recon_loss.data.item(), global_step=epoch)
        writer.add_scalar(tag='a_mloss', scalar_value=a_mloss.data.item(), global_step=epoch)
        print(epoch, a_mloss.data.item())

    torch.save(a_vae, model_dir + 'a_vae')


def train_t_vae(device, lr, epochs, train_loader, model_dir, task_voc_size, batch_size):

    # 初始化任务vae
    t_vae = T_VAE(
        task_voc_size=task_voc_size,
        embedding_size=300,
        hidden_size=200,
        latent_size=200,
        device=device
    )
    t_vae = t_vae.to(device)
    T_loss = T_VAE_Loss()
    optimizer = optim.Adam(t_vae.parameters(), lr=lr)
    writer = SummaryWriter('logs/events.out.t_vae')
    for epoch in range(epochs):
        states = t_vae.init_hidden(batch_size)
        for batch in tqdm(train_loader):
            # 获取数据
            annotator_id, answer, task, target, task_lengths = get_batch(batch)

            # 任务
            task = task.to(device)
            target = target.to(device)
            # 任务vae
            t_output, t_mean, t_logv, t_z, states = t_vae(task, task_lengths, states)
            # detach hidden states
            states = states[0].detach(), states[1].detach()

            # 任务loss
            t_mloss, t_KL_loss, t_recon_loss = T_loss(mu=t_mean, log_var=t_logv, z=t_z, x_hat_param=t_output, x=target)

            t_mloss.backward()

            optimizer.step()

            optimizer.zero_grad()

        writer.add_scalar(tag='t_KL_loss', scalar_value=t_KL_loss.data.item(), global_step=epoch)
        writer.add_scalar(tag='t_recon_loss', scalar_value=t_recon_loss.data.item(), global_step=epoch)
        writer.add_scalar(tag='t_mloss', scalar_value=t_mloss.data.item(), global_step=epoch)
        print(epoch, t_mloss.data.item())

    torch.save(t_vae, model_dir + 't_vae')

def train_mymodel(annotator_vae, task_vae, device, lr, epochs, train_loader, model_dir, annotator_dim, batch_size):

    mymodel = my_model(annotator_vae, task_vae).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mymodel.parameters(), lr=lr)
    writer = SummaryWriter('logs/events.out.mymodel3')

    for epoch in range(epochs):
        states = mymodel.t_vae.init_hidden(batch_size)

        for batch in tqdm(train_loader):
            # 获取数据
            annotator_id, answer, task, target, task_lengths = get_batch(batch)

            # 工人
            annotator_id = np.array(annotator_id).astype(dtype=int)
            annotator_id = torch.from_numpy(annotator_id).to(device)
            annotator_inputs = F.one_hot(annotator_id, annotator_dim).type(torch.float32)  # 工人input
            # 工人vae
            a_output, a_mean, a_logv, a_z = mymodel.a_vae(annotator_inputs)

            # 任务
            task = task.to(device)
            # 任务vae
            t_output, t_mean, t_logv, t_z, states = mymodel.t_vae(task, task_lengths, states)
            states = states[0].detach(), states[1].detach()

            # label
            answer = np.array(answer).astype(dtype=int)
            answer = torch.from_numpy(answer).to(device)
            label_tensor = answer
            z = torch.cat((a_z, t_z.squeeze()), 1)  # z1 z2 结合
            dev_label = mymodel(z)

            # 监督loss
            loss = loss_fn(dev_label, label_tensor)

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

        writer.add_scalar(tag='loss', scalar_value=loss.data.item(), global_step=epoch)
        print(epoch, loss.data.item())

    torch.save(mymodel, model_dir + 'mymodel3')



