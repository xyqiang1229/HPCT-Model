import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MLP, DynamicEdgeConv

# 定义模型
class get_model(nn.Module):
    def __init__(self, out_channels, k=30, aggr='max'):
        super().__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * 6, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)

        self.mlp = MLP([3 * 64, 1024, 256, 128, out_channels], dropout=0.5,
                       norm=None)

    def forward(self, data, batch):
        x, pos = data.x, data.pos
        x0 = torch.cat([pos, x], dim=-1)
        x1 = self.conv1(x0, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        out = self.mlp(torch.cat([x1, x2, x3], dim=1))
        return F.log_softmax(out, dim=1), x3


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss


if __name__ == '__main__':
    import torch
    model = get_model(9, 1)
    xyz = torch.rand(6, 6, 2048)
    tem = model(xyz)
    print('test')