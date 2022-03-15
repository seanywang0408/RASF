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
cfg.backbone = 'pointnet2_msg'



'''
paths
'''
cfg.rasf_weights_path = '../weights/recon_weights_16res32dim64neigh32input.pt'
cfg.ModelNet40_path = '/media/sdb1/Data/modelnet40_normal_resampled'#'./data/modelnet40_normal_resampled'
cfg.ShapeNetPart_path = '/data1/hxy/data/shapenetcore_partanno_segmentation_benchmark_v0_normal'
