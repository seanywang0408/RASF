import os
import time
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import _init_path
from config import cfg
from RASF import RASF
from datasets.shapenetpart import ShapenetPartDataset, to_categorical
from utils.training_utils import backup_terminal_outputs, backup_code, set_seed


save_path = os.path.join('./log/shapenetpart_seg', time.strftime("%y%m%d_%H%M%S"))

with_RASF = cfg.with_RASF
sd_path = cfg.rasf_weights_path
backbone = cfg.backbone # pointnet or pointnet2_msg

print('save_path:', save_path)
os.makedirs(save_path, exist_ok=True)
backup_terminal_outputs(save_path)
backup_code(save_path)
set_seed(100)
if with_RASF:
    print('RASF weights:', sd_path)
else:
    print('Trained without RASF.')

writer = SummaryWriter(save_path)

num_workers = 0
num_epochs = 150

rasf_resolution = cfg.rasf_resolution
rasf_channel = cfg.rasf_channel

num_points = 2048
num_part = 50
num_classes = 16


if backbone == 'pointnet':
    from backbones.pointnet_part_seg import get_model
    batch_size = 32
elif backbone == 'pointnet2_msg':
    from backbones.pointnet2_part_seg import get_model_msg as get_model
    batch_size = 16
else:
    raise ValueError('Invalid Model Name')

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


data_path = cfg.ShapeNetPart_path
train_set = ShapenetPartDataset(data_path, npoints=num_points, split='trainval', normal_channel=False)
test_set = ShapenetPartDataset(data_path, npoints=num_points, split='test', normal_channel=False)

train_loader = DataLoader(train_set,
                          batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)

val_loader = DataLoader(test_set,
                        batch_size=batch_size, 
                          num_workers=num_workers, pin_memory=True)

if with_RASF:
    field = RASF(resolution=(rasf_resolution, rasf_resolution, rasf_resolution), channel=rasf_channel, num_local_points=32).cuda()
    field_state_dict = torch.load(sd_path)
    field.load_state_dict(field_state_dict)
    field.eval()

    model = get_model(part_num=num_part, channel=rasf_channel+3, num_points=num_points).cuda()
else:
    model = get_model(part_num=num_part, channel=3, num_points=num_points).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100], gamma=0.2)



criterion = torch.nn.CrossEntropyLoss()
start_time = time.time()
best_acc = 0
best_class_avg_iou = 0
best_inctance_avg_iou = 0
 
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

    for idx, (data, category, seg) in enumerate(tqdm(train_loader)):
        bz = data.shape[0]
        category = category.cuda()
        seg = seg.cuda()
        seg = seg.view(-1, 1)[:,0]

        data = data.cuda()

        if with_RASF:
            data = torch.cat([data.transpose(2,1), field.batch_samples(data[:,:,:3])], 1)
        else:
            data = data.transpose(2, 1)
        if backbone=='pointnet':
            output, trans_feat = model(data, to_categorical(category, num_classes))
            output = output.contiguous().view(-1, num_part)
            loss = criterion(output, seg)
        else:
            output = model(data, to_categorical(category, num_classes))
            output = output.contiguous().view(-1, num_part)
            loss = criterion(output, seg) 


        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Compute accuracy
        pred_choice = output.data.max(1)[1]
        correct = pred_choice.eq(seg.data).cpu().sum() / num_points
        train_accuracy += correct
        total_num += bz
        
        num_batches += 1
        print(train_accuracy/total_num)
        
    train_instance_acc = train_accuracy / total_num
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
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        for idx, (data, category, seg) in enumerate(tqdm(val_loader)):
            # category = attributes['category'].cuda()
            cur_batch_size, NUM_POINT, _ = data.size()
            category = category.cuda()
            seg = seg.cuda()
            data = data.cuda()
            if with_RASF:
                data = torch.cat([data.transpose(2,1), field.batch_samples(data[:,:,:3])], 1)
            else:
                data = data.transpose(2, 1)
            if backbone=='pointnet':
                output, trans_feat = model(data, to_categorical(category, num_classes))
            else:
                output = model(data, to_categorical(category, num_classes))
            cur_pred_val = output.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            # print(seg.shape)
            target = seg.cpu().numpy()
            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
        for cat in sorted(shape_ious.keys()):
            print('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

        print('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
                 e+1, test_metrics['accuracy'],test_metrics['class_avg_iou'],test_metrics['inctance_avg_iou']))
        if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
            print('Save model...')
            model_savepath = str(save_path) + '/best_model.pth'
            print('Saving at %s'% model_savepath)
            state = {
                'epoch': model,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, model_savepath)
            print('Saved model....')

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        print('Best accuracy is: %.5f'%best_acc)
        print('Best class avg mIOU is: %.5f'%best_class_avg_iou)
        print('Best inctance avg mIOU is: %.5f'%best_inctance_avg_iou)            
            
        writer.add_scalar('Class_avg_iou/Test', test_metrics['class_avg_iou'], e)
        writer.add_scalar('Inctance_avg_iou/Test', test_metrics['inctance_avg_iou'], e)
        writer.add_scalar('Acc/Test', test_metrics['accuracy'], e)

end_time = time.time()
print('Training time: {}'.format(end_time - start_time))
print('Best accuracy is: %.5f'%best_acc)
print('Best class avg mIOU is: %.5f'%best_class_avg_iou)
print('Best inctance avg mIOU is: %.5f'%best_inctance_avg_iou)         
print('sd_path:', sd_path, '\n', 'save_path:', save_path)
