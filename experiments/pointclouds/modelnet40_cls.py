import os
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import _init_path
from config import cfg
from RASF import RASF
from datasets.modelnet40 import ModelNetDataset
from utils.training_utils import backup_terminal_outputs, backup_code, set_seed

save_path = os.path.join('./log/modelnet40_cls', time.strftime("%y%m%d_%H%M%S"))

with_RASF = cfg.with_RASF
sd_path = cfg.rasf_weights_path
backbone = cfg.backbone # pointnet or pointnet2_msg
withnormal = False
optimize_field = False


print('save_path:', save_path)    
os.makedirs(save_path, exist_ok=True)
backup_terminal_outputs(save_path)
backup_code(save_path, marked_in_parent_folder=['utils',])

if with_RASF:
    print('RASF weights:', sd_path)
else:
    print('Trained without RASF.')

writer = SummaryWriter(save_path)

num_workers = 0
num_epochs = 150

rasf_resolution = cfg.rasf_resolution
rasf_channel = cfg.rasf_channel
num_local_points = 32
data_path = cfg.ModelNet40_path

if backbone == 'pointnet':
    from backbones.pointnet import get_model
    batch_size = 128
elif backbone == 'pointnet2_msg':
    from backbones.pointnet2 import get_model_msg as get_model
    batch_size = 16 # reduce batch size due to the limit of gpu memory
    set_seed(14)
else:
    raise ValueError('Invalid Model Name')


train_set = ModelNetDataset(data_path, split='train', normal_channel=withnormal)
test_set = ModelNetDataset(data_path, split='test', normal_channel=withnormal)

train_loader = DataLoader(train_set,
                          batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)

val_loader = DataLoader(test_set,
                        batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)

if with_RASF:
    field = RASF(resolution=(rasf_resolution, rasf_resolution, rasf_resolution), channel=rasf_channel, num_local_points=num_local_points).cuda()
    if sd_path is not None:
        field_state_dict = torch.load(sd_path)
        field.load_state_dict(field_state_dict)
        print('field state dict loaded from: ', sd_path)
    else:
        print('random field')
    field.eval()

    model = get_model(40, channel=rasf_channel+3).cuda()
else:
    model = get_model(40, channel=3).cuda()




if optimize_field:
    optimized_param = list(model.parameters()) + list(field.parameters())
else:
    optimized_param = model.parameters()
optimizer = torch.optim.Adam(optimized_param, lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100], gamma=0.2)



criterion = torch.nn.CrossEntropyLoss()
start_time = time.time()
best_acc = 0

for e in range(num_epochs):

    print('###################')
    print('Epoch:', e)
    print('###################')

    train_loss = 0.
    train_accuracy = 0.
    num_batches = 0
    total_num = 0.

    model.train()
    scheduler.step()

    for idx, (data, category) in enumerate(tqdm(train_loader)):
        bz = data.shape[0]
        category = category.cuda()

        data = data.cuda()
        if with_RASF:
            data = torch.cat([data.transpose(2,1), field.batch_samples(data[:,:,:3])], 1)
        else:
            data = data.transpose(2, 1)
        # print(data.shape)
        if backbone=='pointnet':
            output, trans_feat = model(data)
            loss = criterion(output, category.view(-1))
        else:
            output = model(data)
            loss = criterion(output, category.view(-1)) 



        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pred_label = torch.argmax(output, dim=1)
        train_accuracy += (pred_label == category.view(-1)).sum().float().detach().cpu().item()
        total_num += bz
        
        num_batches += 1
        # print(train_accuracy/total_num)
        

    print('Train loss:', train_loss / num_batches)
    print('Train accuracy:', train_accuracy / total_num)
    writer.add_scalar('Loss/train', train_loss / num_batches, e)
    writer.add_scalar('Acc/train', train_accuracy / total_num, e)


    val_loss = 0.
    val_accuracy = 0.
    num_batches = 0
    total_num = 0.

    model.eval()
    
    with torch.no_grad():
        for idx, (data, category) in enumerate(tqdm(val_loader)):
            bz = data.shape[0]
            category = category.cuda()

            data = data.cuda()
            if with_RASF:
                data = torch.cat([data.transpose(2,1), field.batch_samples(data[:,:,:3])], 1)
            else:
                data = data.transpose(2, 1)
            if backbone=='pointnet':
                pred, _ = model(data)
            else:
                pred = model(data)
            loss = criterion(pred, category.view(-1))
            val_loss += loss.item()

            # Compute accuracy
            pred_label = torch.argmax(pred, dim=1)
            val_accuracy += (pred_label == category.view(-1)).sum().float().detach().cpu().item()
            total_num += bz
            num_batches += 1
    
    if best_acc <= val_accuracy / total_num:
        best_acc = val_accuracy / total_num
    
    writer.add_scalar('Loss/Test', val_loss / num_batches, e)
    writer.add_scalar('Acc/Test', val_accuracy / total_num, e)
    print('Val loss:', val_loss / num_batches)
    print('Val accuracy:', val_accuracy / total_num)
    print('Best accuracy:', best_acc)
end_time = time.time()
print('Training time: {}'.format(end_time - start_time))
print('best acc:', best_acc)
print('sd_path:', sd_path, '\n', 'save_path:', save_path)
