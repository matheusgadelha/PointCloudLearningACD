# Self-supervision for Point Clouds using Approximate Convex Decompositions 

This repository contains code for the paper *Label-Efficient Learning on Point Clouds using Approximate Convex Decompositions*, published at ECCV 2020. 

Thanks to yanx27 for an excellent PyTorch PointNet++ implementation [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch); our model implementation is based off that codebase.

## Environment setup

CUDA setup:
```
CUDA: '9.2.148'    # torch.version.cuda
CuDNN: 7603        # torch.backends.cudnn.version()
```

Conda environment:
```
conda create -n acd-env python=3.6
pip install numpy six protobuf>=3.2.0
pip install torch torchvision
pip install matplotlib tqdm tensorboard_logger trimesh
```

For reference, we also tested using CUDA 10.1, and the corresponding torch and torchvision can be installed using `pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html`.


## Data setup

Download part segmentation dataset **ShapeNet** [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip) and save in `data/shapenetcore_partanno_segmentation_benchmark_v0_normal/`.

Download the pre-computed ACD components for the unlabeled ShapeNet core shapes from [here](http://maxwell.cs.umass.edu/zezhou/visualization/acd/ACDShapeNetSegPartAnno.zip) and extract its content in `data`.

Download the aligned and resampled **ModelNet40** dataset for shape classication from [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `data/modelnet40_normal_resampled/`.


## Few-shot segmentation on ShapeNet

From the project root, the following code snippet trained a model jointly on semantic segmentation on ShapeNetSeg, using 5 samples per shape category (i.e. 5 * 16 = 80 labeled training samples) and a pairwise contrastive loss over ACD components of the unlabeled ShapeNet Core data (for 9 epochs, decaying the learning rate at every epoch, with a batchsize of 24 shapes). 

```
python train_partseg_shapenet_multigpu.py --seed 2001 \
        --k_shot 5 --batch_size 24 --selfsup --step_size 1  --epoch 9 \
        --ss_path data/ACDv2/
```

The models are stored in the experiment output folder, under `checkpoints` sub-folder. Tensorboard logs and console output as txt file are saved under sub-folder `logs`. The test performance is evaluated at the end of the training epochs (i.e. epoch 9 in this case) and written to the logfile.



## Pretrain on ACD and test on ModelNet

**Pretraining on ACD:**

The following example command trains a PointNet++ network on the ACD task. The `seed` is an integer that serves as an identifier for multiple runs of the same experiment. Random rotations around the "up" or Z axis is done as data augmentation during training(`--rotation_z`). Only the best performing model based on the **validation ACD loss** is stored under the experiment output folder, under `checkpoints` sub-folder. Tensorboard logs and console output as txt file are saved under sub-folder `logs`.

```
python pretrain_partseg_shapenet.py --rotation_z --seed 1001 --gpu 0 \
                                    --model pointnet2_part_seg_msg  \
                                    --batch_size 16 --step_size 1  \
                                    --selfsup  --retain_overlaps \
                                    --ss_path data/ACDv2
```


**Evaluate pre-trained model on ModelNet40:**

* Evaluating on ModelNet with cross-validation of SVM (takes a while): `python test_acdfeat_modelnet.py --gpu 0  --sqrt  --model pointnet2_part_seg_msg   --log_dir $LOG_DIR  --cross_val_svm`
* Avoiding the cross-validation for the SVM C, one can also explicitly put the value as a runtime argument: `python test_acdfeat_modelnet.py --gpu 0  --sqrt  --model pointnet2_part_seg_msg   --log_dir $LOG_DIR --svm_c 220.0`
* Examples of `LOG_DIR` can be found at the top of the `test_acdfeat_modelnet.py` code file. Basically it points to wherever the ACD pre-training script dumps its outputs.


