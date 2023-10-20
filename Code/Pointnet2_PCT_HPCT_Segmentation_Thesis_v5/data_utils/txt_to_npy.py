import os
import glob
import numpy as np


# 读取文本文件，返回np
def txt_to_npy(pointcloud_filename):
    txt_file = open(pointcloud_filename, "r") #只读模式打开
    lines = txt_file.readlines()
    np_lines = np.loadtxt(lines)
    txt_file.close()
    print(pointcloud_filename + ' is read')
    return np_lines


if __name__ == '__main__':
    # 读取pointcloud_folder文本文件，转为npy后存在save_folder
    pointcloud_folder = '../../../Data/Own_Thesis_v2/Vehicle/PointNet++/tmp'
    save_folder = '../../../Data/Own_Thesis_v2/Vehicle/PointNet++/tmp'
    pointcloud_filenames = glob.glob(os.path.join(pointcloud_folder, '*.txt'))
    lable_order = 6
    for pointcloud_filename in pointcloud_filenames:
        pure_name = os.path.split(os.path.splitext(pointcloud_filename)[0])[1]
        pointcloud_npy = txt_to_npy(pointcloud_filename)
        pointcloud_npy2 = np.append(pointcloud_npy[:, 0:3], pointcloud_npy[:, lable_order].reshape(-1, 1), axis=1)
        np.save(os.path.join(save_folder, pure_name), pointcloud_npy2)
        print(pointcloud_filename + ' is saved')

