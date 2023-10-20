import math

class DatasetParameters():

    def __init__(self, dataset):

        if dataset == 'Vaihingen':
            self.classes = ['Powerline', 'Low vegetation', 'Impervious surfaces', 'Car', 'Fence/Hedge', 'Roof', 'Facade',
                       'Shrub', 'Tree']
            self.num_class = self.classes.__len__()
            self.train_root = '../../Data/Vaihingen_PointNet++/Training_Validation'
            self.test_root = '../../Data/Vaihingen_PointNet++/Test/'
            self.feature_max = [255, 4, 4]
            self.feature_min = [0, 1, 1]
            self.g_class2color = {'Powerline': [0, 255, 0],
                             'Low vegetation': [0, 0, 255],
                             'Impervious surfaces': [0, 255, 255],
                             'Car': [255, 255, 0],
                             'Fence/Hedge': [255, 0, 255],
                             'Roof': [100, 100, 255],
                             'Facade': [200, 200, 100],
                             'Shrub': [170, 120, 200],
                             'Tree': [255, 0, 0],
                             }


        if dataset == 'S3DIS':
            self.classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa',
                       'bookcase',
                       'board', 'clutter']
            self.num_class = self.classes.__len__()
            self.train_root = '../../Data/S3DIS_PointNet++/Training_Validation'
            self.test_root = '../../Data/S3DIS_PointNet++/Test/'
            self.feature_max = [255, 255, 255]
            self.feature_min = [0, 0, 0]
            self.g_class2color = {'ceiling': [0, 255, 0],
                             'floor': [0, 0, 255],
                             'wall': [0, 255, 255],
                             'beam': [255, 255, 0],
                             'column': [255, 0, 255],
                             'window': [100, 100, 255],
                             'door': [200, 200, 100],
                             'table': [170, 120, 200],
                             'chair': [255, 0, 0],
                             'sofa': [200, 100, 100],
                             'bookcase': [10, 200, 100],
                             'board': [200, 200, 200],
                             'clutter': [50, 50, 50]}


        if dataset == 'Vehicle-InnovationCenter':
            self.classes = ['unlabeled points', 'tree']
            self.num_class = self.classes.__len__()
            self.train_root = '../../Data/Own_Thesis/Vehicle/PointNet++/InnovationCenter/Training_Validation'
            self.test_root = '../../Data/Own_Thesis/Vehicle/PointNet++/InnovationCenter/Test/'
            self.feature_max = [255, 255, 255]
            self.feature_min = [0, 0, 0]
            self.g_class2color = {'unlabeled points': [0, 0, 255],
                             'tree': [0, 255, 0]}


        if dataset == 'Reconstruction-InnovationCenter':
            self.classes = ['unlabeled points', 'tree']
            self.num_class = self.classes.__len__()
            self.train_root = '../../Data/Own_Thesis/Reconstruction/PointNet++/InnovationCenter/Training_Validation'
            self.test_root = '../../Data/Own_Thesis/Reconstruction/PointNet++/InnovationCenter/Test/'
            self.feature_max = [255, 255, 255]
            self.feature_min = [0, 0, 0]
            self.g_class2color = {'unlabeled points': [0, 0, 255],
                             'tree': [0, 255, 0]}

        if dataset == 'Airborne-InnovationCenter':
            self.classes = ['unlabeled points', 'tree']
            self.num_class = self.classes.__len__()
            self.train_root = '../../Data/Own_Thesis/Airborne/PointNet++/InnovationCenter/Training_Validation'
            self.test_root = '../../Data/Own_Thesis/Airborne/PointNet++/InnovationCenter/Test/'
            self.feature_max = [255, 255, 255]
            self.feature_min = [0, 0, 0]
            self.g_class2color = {'unlabeled points': [0, 0, 255],
                             'tree': [0, 255, 0]}

        if dataset == 'MultiSource-InnovationCenter':
            self.classes = ['unlabeled points', 'tree']
            self.num_class = self.classes.__len__()
            self.train_root = '../../Data/Own_Thesis/MultiSource/PointNet++/InnovationCenter/Training_Validation'
            self.test_root = '../../Data/Own_Thesis/MultiSource/PointNet++/InnovationCenter/Test/'
            self.feature_max = [255, 255, 255]
            self.feature_min = [0, 0, 0]
            self.g_class2color = {'unlabeled points': [0, 0, 255],
                             'tree': [0, 255, 0]}

        if dataset == 'Own_Thesis_v2_MultiSource':
            self.classes = ['unlabeled points', 'tree']
            self.num_class = self.classes.__len__()
            self.train_root = '../../Data/Own_Thesis_v2/MultiSource/PointNet++/Training_Validation'
            self.test_root = '../../Data/Own_Thesis_v2/MultiSource/PointNet++/Test/'
            self.feature_max = [255, 255, 255]
            self.feature_min = [0, 0, 0]
            self.g_class2color = {'unlabeled points': [0, 0, 255],
                             'tree': [0, 255, 0]}

        if dataset == 'Own_Thesis_v2_Airborne':
            self.classes = ['unlabeled points', 'tree']
            self.num_class = self.classes.__len__()
            self.train_root = '../../Data/Own_Thesis_v2/Airborne/PointNet++/Training_Validation'
            self.test_root = '../../Data/Own_Thesis_v2/Airborne/PointNet++/Test/'
            self.feature_max = [255, 255, 255]
            self.feature_min = [0, 0, 0]
            self.g_class2color = {'unlabeled points': [0, 0, 255],
                             'tree': [0, 255, 0]}

        if dataset == 'Own_Thesis_v2_Reconstruction':
            self.classes = ['unlabeled points', 'tree']
            self.num_class = self.classes.__len__()
            self.train_root = '../../Data/Own_Thesis_v2/Reconstruction/PointNet++/Training_Validation'
            self.test_root = '../../Data/Own_Thesis_v2/Reconstruction/PointNet++/Test/'
            self.feature_max = [255, 255, 255]
            self.feature_min = [0, 0, 0]
            self.g_class2color = {'unlabeled points': [0, 0, 255],
                             'tree': [0, 255, 0]}

        if dataset == 'Own_Thesis_v2_Vehicle':
            self.classes = ['unlabeled points', 'tree']
            self.num_class = self.classes.__len__()
            self.train_root = '../../Data/Own_Thesis_v2/Vehicle/PointNet++/Training_Validation'
            self.test_root = '../../Data/Own_Thesis_v2/Vehicle/PointNet++/Test/'
            self.feature_max = [255, 255, 255]
            self.feature_min = [0, 0, 0]
            self.g_class2color = {'unlabeled points': [0, 0, 255],
                             'tree': [0, 255, 0]}

        self.g_class2label = {cls: i for i, cls in enumerate(self.classes)}
        self.g_label2color = {self.classes.index(cls): self.g_class2color[cls] for cls in self.classes}

if __name__ == '__main__':
    dataset_parameters = DatasetParameters('Vaihingen')
    print(dataset_parameters.feature_max)