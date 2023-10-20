import os
import numpy as np
from data_utils.DatasetParameters import DatasetParameters
from tqdm import tqdm
from torch.utils.data import Dataset


class PointcloudDataset(Dataset):
    def __init__(self, only_xyz=False, data_root='', dataset='', num_point=4096, block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        self.only_xyz = only_xyz

        #初始化数据集
        dataset_parameters = DatasetParameters(dataset)
        num_class = dataset_parameters.num_class
        # [255, 255, 255]和[0, 0, 0]
        self.feature_max = dataset_parameters.feature_max
        self.feature_min = dataset_parameters.feature_min

        # 读取文件夹中所有文件名，存在rooms_split列表中
        all_rooms = sorted(os.listdir(data_root))
        rooms = []
        for room in all_rooms:
            if os.path.splitext(room)[1] == '.npy':
                rooms.append(room)
        # 有点多余...
        rooms_split = [room for room in rooms]

        # 依据文件名读取点云和标签，存在room_points、room_labels
        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(num_class)

        for room_name in tqdm(rooms_split, total=len(rooms_split)):
            # room_path 数据集的绝对路径
            room_path = os.path.join(data_root, room_name)
            if self.only_xyz:
                room_data = np.load(room_path)  # xyzl or xyzrgbl
                points, labels = room_data[:, 0:3], room_data[:, room_data.shape[1] - 1]  # xyz, N*3; l, N
            else:
                room_data = np.load(room_path)  # xyzrgbl, N*7
                points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
            tmp, _ = np.histogram(labels, range(num_class+1))
            # 树和非树的点数统计
            labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            # 把五个数据集的点全加入self.room_points
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)

        # 五个npy文件中的树和非树的点数 labelweights
        labelweights = labelweights.astype(np.float32) # 0.56，0.44
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print(self.labelweights) # 1，1.088
        # 五个场景，每个场景被采样到的概率 sample_prob
        sample_prob = num_point_all / np.sum(num_point_all)
        # num_iter 理论上可以划分样本的数  11353
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point) # sample_rate is 1.0，num_point is 4096
        room_idxs = []
        for index in range(len(rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        # romm_idxs 五个数据集，理论迭代次数中、每个每个应该被访问的次数 （11354次）
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples.".format(len(self.room_idxs)))

    def __getitem__(self, idx):
        # idx，是 0~11353 中的某个
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]   # only_xyz ->  N * 3 or N * 6
        labels = self.room_labels[room_idx]   # N
        N_points = points.shape[0]

        # 这段代码的作用是不断地随机选择一个点作为块的中心，然后根据中心点和 self.block_size 定义一个块，在块内查找点的索引，直到找到一个包含超过 1024 个点的块为止
        while(True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        # 确保为 4096。point_index 是索引
        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        if self.only_xyz:
            selected_points = points[selected_point_idxs, :]  # num_point * 3
            current_points = np.zeros((self.num_point, 6))  # num_point * 6
            current_points[:, :3] = selected_points

            selected_min, selected_max = np.amin(selected_points, axis=0)[:3], np.amax(selected_points, axis=0)[:3]
            current_points[:, 3] = (selected_points[:, 0] - selected_min[0]) / (selected_max[0] - selected_min[0])
            current_points[:, 4] = (selected_points[:, 1] - selected_min[1]) / (selected_max[1] - selected_min[1])
            current_points[:, 5] = (selected_points[:, 2] - selected_min[2]) / (selected_max[2] - selected_min[2])

            current_labels = labels[selected_point_idxs]




        else:
            selected_points = points[selected_point_idxs, :]  # num_point * 6
            current_points = np.zeros((self.num_point, 9))  # num_point * 9

            selected_min, selected_max = np.amin(selected_points, axis=0)[:3], np.amax(selected_points, axis=0)[:3]
            current_points[:, 6] = (selected_points[:, 0] - selected_min[0]) / (selected_max[0] - selected_min[0])
            current_points[:, 7] = (selected_points[:, 1] - selected_min[1]) / (selected_max[1] - selected_min[1])
            current_points[:, 8] = (selected_points[:, 2] - selected_min[2]) / (selected_max[2] - selected_min[2])

            for i in range(3, 6):
                selected_points[:, i] = (selected_points[:, i] - self.feature_min[i - 3]) / (
                            self.feature_max[i - 3] - self.feature_min[i - 3])

            current_points[:, 0:6] = selected_points
            current_labels = labels[selected_point_idxs]

        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)

        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)


class ScannetDatasetWholeScene():
    # prepare to give prediction on each points
    def __init__(self, only_xyz=False, block_points=4096, dataset='', block_size=0.2):
        # set dataset
        dataset_parameters = DatasetParameters(dataset)
        num_class = dataset_parameters.num_class
        self.only_xyz = only_xyz
        self.root = dataset_parameters.test_root
        self.feature_max = dataset_parameters.feature_max
        self.feature_min = dataset_parameters.feature_min
        # other parameters
        self.block_points = block_points
        self.block_size = block_size
        self.padding = block_size * 0.1
        self.stride = block_size
        self.scene_points_num = []

        self.scene_points_list = []
        self.semantic_labels_list = []
        self.room_coord_min, self.room_coord_max = [], []

        self.file_list = []
        file_list_all = [d for d in os.listdir(self.root)]
        for room in file_list_all:
            if os.path.splitext(room)[1] == '.npy':
                self.file_list.append(room)

        for file in self.file_list:
            data = np.load(self.root + file)
            points = data[:, :3]
            if only_xyz:
                self.scene_points_list.append(data[:, :3])
                self.semantic_labels_list.append(data[:, data.shape[1] - 1])
            else:
                self.scene_points_list.append(data[:, :6])
                self.semantic_labels_list.append(data[:, 6])
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        labelweights = np.zeros(num_class)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(num_class+1))
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        if self.only_xyz:
            points = point_set_ini[:,:3]
        else:
            points = point_set_ini[:,:6]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]),  np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                point_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (points[:, 1] >= s_y - self.padding) & (
                                points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size <= self.block_points:
                    continue

                num_batch = int(np.ceil(point_idxs.size / self.block_points))
                point_size = int(num_batch * self.block_points)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]

                normlized_xyz = np.zeros((point_size, 3))

                selected_min, selected_max = np.amin(data_batch, axis=0)[:3], np.amax(data_batch, axis=0)[:3]
                normlized_xyz[:, 0] = (data_batch[:, 0] - selected_min[0]) / (selected_max[0] - selected_min[0])
                normlized_xyz[:, 1] = (data_batch[:, 1] - selected_min[1]) / (selected_max[1] - selected_min[1])
                normlized_xyz[:, 2] = (data_batch[:, 2] - selected_min[2]) / (selected_max[2] - selected_min[2])
                # data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                # data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)

                if not self.only_xyz:
                    for i in range(3, 6):
                        data_batch[:, i] = (data_batch[:, i] - self.feature_min[i - 3]) / (self.feature_max[i - 3] - self.feature_min[i - 3])

                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]

                data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
                label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if label_room.size else batch_weight
                index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs

        data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))
        label_room = label_room.reshape((-1, self.block_points))
        sample_weight = sample_weight.reshape((-1, self.block_points))
        index_room = index_room.reshape((-1, self.block_points))
        return data_room, label_room, sample_weight, index_room

    def __len__(self):
        return len(self.scene_points_list)