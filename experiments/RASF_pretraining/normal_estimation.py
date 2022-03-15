import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import _init_path
from config import cfg
from RASF import RASF
from pointclouds.datasets.modelnet40 import ModelNetDataLoader
from utils.training_utils import backup_terminal_outputs, backup_code, set_seed
from utils.chamfer_distance import ChamferDistance



save_path = os.path.join('./log/normal_estimation', time.strftime("%y%m%d_%H%M%S"))
os.makedirs(save_path, exist_ok=True)
print('save_path', save_path)

backup_terminal_outputs(save_path)
backup_code(save_path)

batch_size = 32
num_workers = 0
num_epochs = 150

num_input_points = 32

rasf_resolution = 32
rasf_channel = 32
num_local_points = 32 # total points is 1024
withnormal = False


data_path = cfg.ModelNet40_path
train_set = ModelNetDataLoader(data_path, split='train', normal_channel=withnormal)
test_set = ModelNetDataLoader(data_path, split='test', normal_channel=withnormal)

train_loader = DataLoader(train_set,
                          batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)

val_loader = DataLoader(test_set,
                        batch_size=batch_size,
                        num_workers=num_workers, pin_memory=True)

"""
normal estimation
"""
class Generator(nn.Module):
    def __init__(self, rasf_channel):

        super().__init__()
        self.conv1 = nn.Conv1d(rasf_channel+3, rasf_channel*2, 1)
        self.conv2 = nn.Conv1d(rasf_channel*2, rasf_channel*4, 1)
        self.conv3 = nn.Conv1d(rasf_channel*4, rasf_channel*8, 1)

        self.conv4 = nn.Conv1d(rasf_channel*8, rasf_channel*4, 1)
        self.conv5 = nn.Conv1d(rasf_channel*4, rasf_channel*2, 1)
        self.conv6 = nn.Conv1d(rasf_channel*2, 3, 1)


    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, negative_slope=0.02, inplace=True)
        x = self.conv2(x)
        x = F.leaky_relu(x, negative_slope=0.02, inplace=True)
        x = self.conv3(x)
        x = F.leaky_relu(x, negative_slope=0.02, inplace=True)
         
        x = self.conv4(x)
        x = F.leaky_relu(x, negative_slope=0.02, inplace=True)
        x = self.conv5(x)
        x = F.leaky_relu(x, negative_slope=0.02, inplace=True)
        x = self.conv6(x)

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
    scheduler.step()
    for idx, (data, category) in enumerate(tqdm(train_loader)):
        category = category.cuda()
        data = data.cuda()
        points = data[:,:,:3]
        normals=data[:,:,3:]

        input_data = torch.cat([points.transpose(2,1), field.batch_samples(points)], 1)
        #print(input_data.shape)
        output = model(input_data)

        cosinesim=F.cosine_similarity(output.transpose(2,1), normals, dim=2, eps=1e-8)
        
        loss = 1-cosinesim.mean()
        #print(loss)
        train_loss += loss.item()
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        num_batches += 1

 


    print('Train loss:', train_loss / num_batches)
    # print('Train accuracy:', train_accuracy / num_batches)

    val_loss = 0.
    val_accuracy = 0.
    num_batches = 0

    model.eval()

    with torch.no_grad():
        for idx, (data, category) in enumerate(tqdm(val_loader)):
            category = category.cuda()
            points = data[:,:,:3]
            normals = data[:,:,3:]

            input_data = torch.cat([points.transpose(2,1), field.batch_samples(points)], 1)
            #print(input_data.shape)
            output = model(input_data)
            # local field
            cosinesim=F.cosine_similarity(output.transpose(2,1), normals, dim=2, eps=1e-8)
            loss = 1-cosinesim.mean()
            val_loss += loss.item()

            num_batches += 1


    print('Val loss:', val_loss / num_batches)
    # print('Val accuracy:', val_accuracy / num_batches)
    if best_loss >= val_loss / num_batches:
        best_loss = val_loss / num_batches
        torch.save(field.state_dict(), os.path.join(save_path, "normalestim_weights.pt"))

end_time = time.time()
print('Training time: {}'.format(end_time - start_time))
print('best loss: ', best_loss)