import numpy as np
import warnings
import os
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')
import random


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc



class ModelNetDataset(Dataset):
    def __init__(self, root, npoint=1024, split='train', normal_channel=True, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel


        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d'%(split,len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, catogory = self.cache[index]
        else:
            fn = self.datapath[index]
            catogory = self.classes[self.datapath[index][0]]
            catogory = np.array([catogory]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            point_set = point_set[0:self.npoints,:]

            point_set = pc_normalize(point_set)            
            normals = point_set[:, 3:]
            point_set = point_set[:, 0:3]


            if self.normal_channel:
                point_set = np.concatenate([point_set, normals], -1)

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, catogory)
        # print(point_set.dtype)
        return point_set, catogory.astype(np.int64)

    def __getitem__(self, index):
        return self._get_item(index)


