from yacs.config import CfgNode as CN

cfg = CN()

'''
RASF configs
'''
cfg.rasf_resolution = 16
cfg.rasf_channel = 32


'''
paths
'''
cfg.rasf_weights_path = '../weights/recon_weights_16res32dim64neigh32input.pt'
cfg.ModelNet40_path = '/media/sdb1/Data/modelnet40_normal_resampled'#'./data/modelnet40_normal_resampled'
cfg.ShapeNetPart_path = '/data1/hxy/data/shapenetcore_partanno_segmentation_benchmark_v0_normal'
