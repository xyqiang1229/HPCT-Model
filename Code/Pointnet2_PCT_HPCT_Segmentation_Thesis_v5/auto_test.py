import os

# 数据集
for dataset in ['Own_Thesis_v2_Reconstruction', 'Own_Thesis_v2_Airborne', 'Own_Thesis_v2_Vehicle', 'Own_Thesis_v2_MultiSource']:
    for log_dir in ['pointnet_230512_1']:
        cmd = 'python test_semseg.py --dataset ' + dataset + ' --log_dir ' + log_dir + ' --block_size 10.0 --gpu 0 --batch_size 40 --visual --only_xyz True'
        os.system(cmd)