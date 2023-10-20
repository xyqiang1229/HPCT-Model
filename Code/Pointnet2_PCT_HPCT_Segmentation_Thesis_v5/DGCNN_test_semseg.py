"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.PointcloudLoader import ScannetDatasetWholeScene
from data_utils.DatasetParameters import DatasetParameters
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np
# from thop import profile

from torch_geometric.data import (
    Data,
    InMemoryDataset,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--block_size', type=float, default=1.0, help='Smple space per whole space')
    parser.add_argument('--only_xyz', type=bool, default=False, help='batch size in testing [default: 32]')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting [default: 5]')
    parser.add_argument('--dataset', type=str, default='Vaihingen', help='Dataset')

    return parser.parse_args()


# 对全场景的点，进行三次预测。vote_label_pool 大小为 场景点数 * 类别数2
# 对每次预测的结果进行统计（对应位置加1）
def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    # 对每个BATCH_SIZE
    for b in range(B):
        # 对每个点
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


# pyg独有的数据处理方式
def pyg_processor(points, target, batch_size):
    points = points.data.numpy()
    # 旋转数据
    points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
    points = torch.Tensor(points)
    points, target = points.float().cuda(), target.long().cuda()

    data_list = []
    for batch_idx in range(points.size(0)):
        # 提取当前批次的点云数据
        current_batch_points = points[batch_idx]

        # 提取xyz坐标信息（后三列）
        pos = current_batch_points[:, 3:]
        x = torch.zeros_like(current_batch_points[:, 3:])

        # 创建Data对象并添加到列表中
        data = Data(pos=pos, x=x, y=target[batch_idx])
        data_list.append(data)

    data, slices = InMemoryDataset.collate(data_list)

    # my_labels 是batch的标签
    my_labels = torch.zeros(4096 * batch_size, dtype=torch.long).cuda()
    for j in range(1, len(slices["x"])):
        if j == 1:
            my_labels[0:slices["x"][j]] = j - 1
        else:
            my_labels[slices["x"][j - 1]:slices["x"][j]] = j - 1
    return data, my_labels, target

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''Set Dataset'''
    dataset_parameters = DatasetParameters(args.dataset)
    classes = dataset_parameters.classes
    num_class = dataset_parameters.num_class
    g_label2color = dataset_parameters.g_label2color
    class2label = {cls: i for i, cls in enumerate(classes)}
    seg_classes = class2label
    seg_label_to_cat = {}
    for i, cat in enumerate(seg_classes.keys()):
        seg_label_to_cat[i] = cat

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = '../../Predict/Own_Thesis/' + args.dataset  + '/' + args.log_dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('\nPARAMETER ...')
    log_string(args)

    NUM_CLASSES = num_class
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(only_xyz=args.only_xyz, dataset=args.dataset, block_points=NUM_POINT, block_size=args.block_size)
    log_string("The number of test data is: %d" % len(TEST_DATASET_WHOLE_SCENE))

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    # 计算参数量和运算量
    # if args.only_xyz:
    #     test_data = torch.randn((1, 6, NUM_POINT))
    # else:
    #     test_data = torch.randn((1, 9, NUM_POINT))
    # test_data = test_data.cuda()
    # flops, params = profile(classifier, inputs=(test_data,))
    # log_string("FLOPs: %d" % flops)
    # log_string("Params: %d" % params)

    with torch.no_grad():
        scene_id = TEST_DATASET_WHOLE_SCENE.file_list
        scene_id = [x[:-4] for x in scene_id]
        num_batches = len(TEST_DATASET_WHOLE_SCENE)

        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        log_string('---- EVALUATION WHOLE SCENE----')

        # 三个场景
        for batch_idx in range(num_batches):
            print("Inference [%d/%d] %s ..." % (batch_idx + 1, num_batches, scene_id[batch_idx]))
            total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]
            if args.visual:
                fout = open(os.path.join(visual_dir, scene_id[batch_idx] + '_pred.obj'), 'w')
                fout_gt = open(os.path.join(visual_dir, scene_id[batch_idx] + '_gt.obj'), 'w')
                fout_confusion = open(os.path.join(visual_dir, scene_id[batch_idx] + '_confusion.obj'), 'w')


            whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
            whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]

            # 投票3次
            vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
            for _ in tqdm(range(args.num_votes), total=args.num_votes):
                # 6627 *4096 * 6
                scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
                # 该场景划分成了num blocks (6627)块测试集
                num_blocks = scene_data.shape[0]
                # 该场景需要多少个batch
                s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
                if args.only_xyz:
                    batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 6))
                else:
                    batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))

                # BATCH_SIZE * 4096 * 2
                batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))

                # 开始对场景进行测试
                for sbatch in range(s_batch_num):
                    # 该批次开始和结束的batch编号
                    start_idx = sbatch * BATCH_SIZE
                    end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)

                    # 拷贝场景数据到局部
                    real_batch_size = end_idx - start_idx
                    batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                    batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                    # 每个点都有独一无二的编号
                    batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                    batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                    batch_data[:, :, 3:6] /= 1.0

                    torch_data = torch.Tensor(batch_data)
                    data, my_labels, _ = pyg_processor(torch_data, torch.Tensor(batch_label), BATCH_SIZE)
                    # BATCH_SIZE * 4096 * 2
                    seg_pred, _ = classifier(data, my_labels)

                    # 对每个点，做出预测。BATCH_SIZE·4096 * 1
                    seg_pred = seg_pred.contiguous().cpu().data.max(1)[1]
                    # 改格式
                    seg_pred = seg_pred.view(BATCH_SIZE, 4096)
                    batch_pred_label = seg_pred.numpy()

                    vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                               batch_pred_label[0:real_batch_size, ...],
                                               batch_smpw[0:real_batch_size, ...])

            # 三次投票完成，寻找最大值（投票最多的地方），作为最终预测结果
            pred_label = np.argmax(vote_label_pool, 1)

            for l in range(NUM_CLASSES):
                # 每个类别的总点数
                total_seen_class_tmp[l] += np.sum((whole_scene_label == l))
                # 每个类别中，预测正确的总点数（本来是树，预测也是树的点数）
                total_correct_class_tmp[l] += np.sum((pred_label == l) & (whole_scene_label == l))
                # 每个类别中，预测是树，和本身是树的总点数
                total_iou_deno_class_tmp[l] += np.sum(((pred_label == l) | (whole_scene_label == l)))
                # 统计三个场景的总点数
                total_seen_class[l] += total_seen_class_tmp[l]
                # 统计三个场景中预测正确的总点数
                total_correct_class[l] += total_correct_class_tmp[l]
                # 统计三个场景中，预测是树，和本身是树的总点数
                total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

            # 记录混淆矩阵 -bfge
            TN = total_correct_class_tmp[0]
            TP = total_correct_class_tmp[1]
            FP = total_seen_class_tmp[0] - TN
            FN = total_seen_class_tmp[1] - TP

            # 加入平滑项，防止在机器学习中的数值不稳定性
            IoU = float(TP) / (TP + FN + FP + 1e-6)
            prec_tree = float(TP) / (TP + FP + 1e-6)
            recall = float(TP) / (TP + FN + 1e-6)
            OA = (float(TP) + TN) / (TN + TP + FP + FN + 1e-6)
            F1_score = 2 * prec_tree * recall / (prec_tree + recall + 1e-6)
            prec_no_tree = TN / (TN + FN + 1e-6)
            mAcc = (prec_tree + prec_no_tree) / 2

            log_string('-------' + scene_id[batch_idx] + '--------')

            log_string("evaluation index: ")
            log_string(f'tree IoU: {IoU}')
            log_string(f'tree Precision: {prec_tree}, Recall: {recall}')
            log_string(f'tree F1 score: {F1_score}')
            log_string(f'OA: {OA}, mAcc: {mAcc}')

            print('----------------------------\n')

            filename = os.path.join(visual_dir, scene_id[batch_idx] + '.txt')
            with open(filename, 'w') as pl_save:
                for i in pred_label:
                    pl_save.write(str(int(i)) + '\n')
                pl_save.close()
            for i in range(whole_scene_label.shape[0]):
                color = g_label2color[pred_label[i]]
                color_gt = g_label2color[whole_scene_label[i]]
                if args.visual:
                    fout.write('v %f %f %f %d %d %d\n' % (
                        whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color[0], color[1],
                        color[2]))
                    fout_gt.write(
                        'v %f %f %f %d %d %d\n' % (
                            whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color_gt[0],
                            color_gt[1], color_gt[2]))
                    if pred_label[i] == 1 and whole_scene_label[i] == 1:
                        fout_confusion.write(
                            'v %f %f %f %d %d %d\n' % (
                                whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], 0,
                                255, 0))
                    elif pred_label[i] == 0 and whole_scene_label[i] == 0:
                        fout_confusion.write(
                            'v %f %f %f %d %d %d\n' % (
                                whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], 0,
                                0, 255))
                    elif pred_label[i] == 1 and whole_scene_label[i] == 0:
                        fout_confusion.write(
                            'v %f %f %f %d %d %d\n' % (
                                whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], 255,
                                0, 0))
                    elif pred_label[i] == 0 and whole_scene_label[i] == 1:
                        fout_confusion.write(
                            'v %f %f %f %d %d %d\n' % (
                                whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], 0,
                                0, 0))


            if args.visual:
                fout.close()
                fout_gt.close()
                fout_confusion.close()

        print("Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)

    print("over")
