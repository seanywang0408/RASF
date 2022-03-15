import os
import time
from tqdm import tqdm

import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import _init_path
from config import cfg
from RASF import RASF
from pointclouds.datasets.shapenetpart import ShapenetPartDataset, to_categorical
from utils.training_utils import backup_terminal_outputs, backup_code, set_seed
from utils.chamfer_distance import ChamferDistance


save_path = os.path.join('./log/recon', time.strftime("%y%m%d_%H%M%S"))
os.makedirs(save_path, exist_ok=True)
print('save_path', save_path)

backup_terminal_outputs(save_path)
backup_code(save_path)

batch_size = 64
num_workers = 0
num_epochs = 150

num_input_points = 24

rasf_resolution = cfg.rasf_resolution
rasf_channel = cfg.rasf_channel
num_local_points = 64 # total_points = 2048


data_path = cfg.ShapeNetPart_path
train_set = ShapenetPartDataset(data_path, npoints=2048, split='trainval')
test_set = ShapenetPartDataset(data_path, npoints=2048, split='test')

train_loader = DataLoader(train_set,
                          batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)

val_loader = DataLoader(test_set,
                        batch_size=batch_size,
                        num_workers=num_workers, pin_memory=True)

class Generator(nn.Module):
    def __init__(self, rasf_channel):

        super().__init__()
        self.conv1 = nn.Conv1d(rasf_channel+3, rasf_channel*2, 1)
        self.conv2 = nn.Conv1d(rasf_channel*2, rasf_channel*4, 1)
        self.conv3 = nn.Conv1d(rasf_channel*4, rasf_channel*8, 1)

        self.fc1 = nn.Linear(rasf_channel*8, rasf_channel*8*2)
        self.fc2 = nn.Linear(rasf_channel*8*2, 1024*3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, negative_slope=0.02, inplace=True)
        x = self.conv2(x)
        x = F.leaky_relu(x, negative_slope=0.02, inplace=True)
        x = self.conv3(x)
        x = F.leaky_relu(x, negative_slope=0.02, inplace=True)
         
        x = x.max(-1)[0]

        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=0.02, inplace=True)
        x = self.fc2(x)
        x = x.view(x.shape[0], -1, 3)
        return x



model = Generator(rasf_channel=rasf_channel).cuda()
field = RASF(resolution=(rasf_resolution, rasf_resolution, rasf_resolution), channel=rasf_channel, num_local_points=num_local_points).cuda()

optimizer = torch.optim.Adam(list(model.parameters())+list(field.parameters()), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100], gamma=0.2)

start_time = time.time()
best_loss = 20


chamfer_dist = ChamferDistance()

for e in range(num_epochs):

    print('###################')
    print('Epoch:', e)
    print('###################')

    train_loss = 0.
    train_accuracy = 0.
    num_batches = 0

    model.train()
    field.train()
    scheduler.step()

    for idx, (data, category, seg) in enumerate(tqdm(train_loader)):
        category = category.cuda()

        data = data.cuda()


        points = data
        data = torch.cat([data.transpose(2,1), field.batch_samples(data)], 1)
        select_points = torch.ones(data.shape[0], data.shape[2]).multinomial(num_samples=num_input_points).cuda()
        data = data.gather(-1, select_points.unsqueeze(1).expand(-1, data.shape[1], -1))
        
        output = model(data)

        d1, d2 = chamfer_dist(output, points)
        loss = (d1.mean() + d2.mean())
        train_loss += loss.item()
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        num_batches += 1
        print(train_loss/num_batches)

    os.makedirs(os.path.join(save_path, 'epoch_%d'%e))
    for i, (y_points, pred_points) in enumerate(zip(points.cpu().detach(), output.cpu().detach())):
        trimesh.PointCloud(y_points.numpy(), colors=np.zeros(y_points.shape)).export(os.path.join(save_path, 'epoch_%d'%e, 'train_%d_y.ply'%i))
        trimesh.PointCloud(pred_points.numpy(), colors=np.zeros(pred_points.shape)).export(os.path.join(save_path, 'epoch_%d'%e, 'train_%d_pred.ply'%i))


    print('Train loss:', train_loss / num_batches)

    val_loss = 0.
    val_accuracy = 0.
    num_batches = 0

    model.eval()
    field.eval()

    with torch.no_grad():
        for idx, (data, category, seg) in enumerate(tqdm(val_loader)):
            category = category.cuda()
            data = data.cuda()

            points = data
            data = torch.cat([data.transpose(2,1), field.batch_samples(data)], 1)
            select_points = torch.ones(data.shape[0], data.shape[2]).multinomial(num_samples=num_input_points).cuda()
            data = data.gather(-1, select_points.unsqueeze(1).expand(-1, data.shape[1], -1))
            # data = data.max(-1)[0]

            output = model(data)

            d1, d2 = chamfer_dist(output, points)
            loss = (d1.mean() + d2.mean())
            val_loss += loss.item()

            num_batches += 1

    for i, (y_points, pred_points) in enumerate(zip(points.cpu().detach(), output.cpu().detach())):
        # points.shape == [n_points, 3]
        trimesh.PointCloud(y_points.numpy(), colors=np.zeros(y_points.shape)).export(os.path.join(save_path, 'epoch_%d'%e, 'test_%d_y.ply'%i))
        trimesh.PointCloud(pred_points.numpy(), colors=np.zeros(pred_points.shape)).export(os.path.join(save_path, 'epoch_%d'%e, 'test_%d_pred.ply'%i))

    print('Val loss:', val_loss / num_batches)
    # print('Val accuracy:', val_accuracy / num_batches)
    if best_loss >= val_loss / num_batches:
        best_loss = val_loss / num_batches
        torch.save(field.state_dict(), os.path.join(save_path, "recon_weights.pt"))


end_time = time.time()
print('Training time: {}'.format(end_time - start_time))
print('best loss: ', best_loss)


