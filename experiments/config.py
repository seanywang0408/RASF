from numpy import False_
from yacs.config import CfgNode as CN

cfg = CN()

'''
RASF configs
'''
cfg.with_RASF = True
cfg.rasf_resolution = 16
cfg.rasf_channel = 32


'''
downstream backbones:
pointnet, pointnet2_msg
'''
cfg.backbone = 'pointnet'



'''
paths
'''
cfg.rasf_weights_path = '../weights/recon_weights_16res32dim64neigh32input.pt'
cfg.ModelNet40_path ='./data/modelnet40_normal_resampled'
cfg.ShapeNetPart_path = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal'
