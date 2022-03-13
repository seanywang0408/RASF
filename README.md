# Representation-Agnostic Shape Fields 

This is the official code of ICLR'22 paper **Representation-Agnostic Shape Fields** written in PyTorch ([openreview](https://openreview.net/forum?id=-ngwPqanCEZ)) ([arxiv]()).

# Features

Representation-Agnostic Shape Fields (RASF) is a generalizable and computation-efficient shape embedding layer for 3D deep learning. Shape embeddings for various 3D shape representations (point clouds, meshes and voxels) are retrieved by coordinates indexing. We provide two effective schemes for RASF pre-training, that is shape reconstruction and normal estimation, to enable RASF to learn robust and general shape embeddings.  Once trained, RASF could be plugged into any 3D neural network with negligible cost. RASF widely boosts the performance for various 3D representations, neural backbones and applications.

Since a large parts of our downstream experiments are modified based on other codebases, directly releasing all the code would be a bit messy. We are stilling working on refactoring the code to make it more graceful. Currently we only release the experiemnts of PointNet and PointNet++ for classfication and segmentation for point clouds data.

# Code Structure

- [RASF.py](./RASF.py): The basic module of RASF
- [experiments/](./experiments):
    - [pointclouds](./experiments/pointclouds): Downstream tasks on point clouds.
    - meshes (TBD): Downstream tasks on meshes.
    - voxels (TBD): Downstream tasks on voxels.
    - [RASF_pretraining](./experiments/pointclouds): Pretrain RASF on pretext tasks.
    - [utils](./experiments/utils): Utils for training.
    - [config.py](./experiments/config.py): Configurations for paths and settings.
    - [weights](./experiments/weights): Directory to place the pretrained RASF weights.
    - [data](./experiments/data): Directory to place the downloaded datasets.

# Requirements
Only basic PyTorch packages are needed.
```
pip install -r requirements.txt
```

# Usage
We show the usage of RASF on point clouds data. Demos on meshes and voxels would be presented with the rest experiments.

```
# inference for a batch of point clouds
rasf = RASF()
rasf.load_state_dict(torch.load('./weights/recon_weights.pt'))

pcd = torch.rand(16,1000,3) # shape: [batch, num_p, 3]
RASF_embedding = rasf.batch_samples(pcd) # shape: [batch, rasf_channel, num_p]
pcd_with_RASF = torch.cat([pcd.transpose(2,1), RASF_embedding], 1) # shape: [batch, (rasf_channel+3), num_p]

```


# Download

## Data

Download ModelNet40 and ShapeNetPart from this [link](https://github.com/AnTao97/PointCloudDatasets). Put them in ``./experiments/data`` as below:
```
./experiments/data  
    |--- modelnet40_normal_resampled
    |--- shapenetcore_partanno_segmentation_benchmark_v0_normal
```

## Pretrained Weights

Download the pretrained weights from [Onedrive](https://1drv.ms/u/s!Ajsnj0gOimMfi40WA_2UoQmHnLerBw?e=ggaDbv) and put them in ``./experiments/weights``. Or you can run the pretraining code yourself and put the weights in the above directory. In this case, modify the weights path in ``./experiments/config.py``.


# Evaluation

To evaluate RASF performance on point clouds, run ``./experiments/pointclouds/modelnet40_cls.py`` and ``./experiments/pointclouds/shapenetpart_seg.py``.

# Acknowledgement and Citation

This codebase borrows a lot from [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch). Thanks for their helps!

If you find this project useful in your research, please cite the following papers:
``` bibtex
@inproceedings{huang2022representation,
  title={Representation-Agnostic Shape Fields},
  author={Huang, Xiaoyang and Yang, Jiancheng and Wang, Yanjun and Chen, Ziyu and Li, Linguo and Li, Teng and Ni, Bingbing and Zhang, Wenjun},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```