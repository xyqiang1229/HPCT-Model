# HPCT-Model
1. Introduction 
The accompanying materials are provided along with the paper “Hierarchical Point Cloud Transformer: A Unified Vegetation Semantic Segmentation Model for Multisource Point Clouds Based on Deep Leaning”. The materials mainly include the source code of hierarchical point cloud Transformer (HPCT) model, the point cloud dataset mentioned in the paper, and the descriptions and instructions of the code operation. Need to note:
(1)	The model code includes code for testing with other models (such as PointNet, PointNet++, PCT, Point CNN and DGCNN etc.) and the corresponding indexes for model’s evaluation.
(2)	Due to the large amount of the involved dataset, it is not possible to upload it to GitHub under the existing condition. Please go to the following cloud disk address to obtain it. URL: https://pan.baidu.com/s/1gTWYa94oiYGEqwYKbNouaA?pwd=ptfg, and the extracting code is “ptfg”.

2. Data Preprocessing
Store point cloud data with tree label 1 and non-tree label 0 in .txt file format, and then run txt_to_npy.py to convert the .txt file into a program readable .npy file.

3. Requirements of Operating Environment
Python>=3.8, pytorch>=1.13, CUDA11.7, tqdm>=4.66.1, numpy>=1.24.3

4. Model Training
python train_semseg.py 
--dataset 
--model 
--only_xyz True 
--num_workers 16 --batch_size 100 --epoch 50 --block_size 10.0 --smple_rate 1.0
--log_dir 
--gpu 0

Among these parameters:
--dataset: It has four options: Reconstruction, Airborne, Vehicle and MultiSource, which are corresponding to the independent HPCT training with single type of point clouds and a unified HPCT training with multi-source data.
--model: It has four options: HPCT_sem_seg, pointnet_sem_seg.py, pointnet2_sem_seg.py，PCT_sem_seg，PointCNN_sem_seg_pyg etc. 
--only_xyz: True, indicates only using xyz coordinates, ignoring the color information. It is mandatory for multi-source data training.
--num_workers 16 --batch_size 100 --epoch 50 --block_size 10.0 --smple_rate 1.0:  Theses training parameters, specific meanings can be found in the paper.
--log_dir DIR: the directory name where the model and training records are stored, which is usually used the “Model Date Sequence Number” format.
--gpu: Specify which GPU card the program runs on. For example, “--gpu:0” refers to running on GPU number 0.

Note: When training DGCNN model, it is necessary to replace the training script with DGCNN_train_semseg.py, --model should select DGCNN_sem_seg, and others remain unchanged.

5. Results Prediction
python test_semseg.py
--dataset 
--log_dir 
--only_xyz True
--block_size 10.0 --gpu 0 --batch_size 256
--visual

Noet:
(1) The trained weights can be found in “best_model.pth”.
(2) Specify dataset and the experiment directory with “--dataset” and “--log_dir DIR”, the prediction process can be completed. “--visual” means saving the visualization results to “/visual”, which needs to be opened using MeshLab software.
(3) When predicting DGCNN model, it is necessary to replace the testing script with DGCNN_test_semseg.py, and others remain unchanged.
