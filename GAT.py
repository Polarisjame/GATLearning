import argparse

import dgl
import dgl.function as fn
from torch import nn
from torch import FloatTensor
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, Softmax
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from dgl.data import CoraGraphDataset
from time import sleep
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def my_add_self_loop(graph):
    # add self loop
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    return graph


def evaluate(model, features, graph, labels, mask, lossF):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        loss = lossF(logits, labels)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels), loss.item()


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return param_size, param_sum, buffer_size, buffer_sum, all_size


class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.res_linear = nn.Linear(in_dim, out_dim)
        self.rate = torch.nn.Parameter(torch.tensor(0, dtype=torch.float32))

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        z = self.fc(h)  # eq. 1
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)  # eq. 2
        self.g.update_all(self.message_func, self.reduce_func)  # eq. 3 and 4
        h0 = self.res_linear(h)
        self.g.ndata['h'] += h0 * self.rate
        return self.g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h


# --------------------------------------------
class GCN_Attention(nn.Module):
    def __init__(self, g, in_feat, out_feat, args):
        super(GCN_Attention, self).__init__()
        self.outlinear = nn.Linear(in_feat, out_feat)
        self.attn_fc = nn.Linear(out_feat * 2, 1, bias=False)
        self.attn_activ = nn.GELU()
        self.softmax = Softmax(1)
        self.graph = g
        self.res = args.res_add
        self.rezero = args.re_zero
        if args.res_add:
            if args.re_zero:
                self.rate = torch.nn.Parameter(torch.tensor(0, dtype=torch.float32))
            self.res_linear = nn.Linear(in_feat, out_feat)

    def edge_attn(self, edges):
        z = torch.cat((edges.src['z'], edges.dst['z']), 1)
        a = self.attn_fc(z)
        return {'e': self.attn_activ(a)}

    def msg_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reducer_func(self, nodes):
        a = self.softmax(nodes.mailbox['e'])
        h = torch.sum(a * nodes.mailbox['z'], 1)
        return {'h': h}

    def forward(self, feature):
        z = self.outlinear(feature)
        self.graph.ndata['z'] = z
        self.graph.apply_edges(self.edge_attn)
        self.graph.update_all(self.msg_func, self.reducer_func)
        if self.res:
            h0 = self.res_linear(feature)
            if self.rezero:
                self.graph.ndata['h'] += h0 * self.rate
            else:
                self.graph.ndata['h'] += h0
        return self.graph.ndata.pop('h')


class MultiHead_GAT_Attention(nn.Module):
    def __init__(self, g, in_feat, out_feat, num_heads, args, mode='concate'):
        super(MultiHead_GAT_Attention, self).__init__()
        self.mul_attn = nn.ModuleList()
        for head in range(num_heads):
            self.mul_attn.append(GCN_Attention(g, in_feat, out_feat, args))
        if mode == "concate":
            self.head_linear = GCN_Attention(g, out_feat * num_heads, out_feat, args)
        else:
            self.head_linear = nn.Linear(out_feat, out_feat)
        self.mode = mode

    def forward(self, feature):
        outs = [gat_attn(feature) for gat_attn in self.mul_attn]
        if self.mode == "concate":
            catch_outs = torch.cat(outs, 1)
        else:
            sum_outs = torch.mean(torch.stack(outs), dim=0)
            return self.head_linear(sum_outs)
        return self.head_linear(catch_outs)


class GCNNet(nn.Module):
    def __init__(self, g, in_feat, args):
        super(GCNNet, self).__init__()
        self.layers = nn.ModuleList()
        out_dim = 7
        layers = args.num_layers
        hid_dim = args.hid_dim
        num_heads = args.num_heads
        if args.dense_net:
            dense_dim = args.dense_dim
            self.layers.append(MultiHead_GAT_Attention(g, in_feat, dense_dim, num_heads, args))
            for lay in range(layers - 1):
                self.layers.append(MultiHead_GAT_Attention(g, dense_dim * (lay + 1), dense_dim, num_heads, args))
            self.outlayer = MultiHead_GAT_Attention(g, dense_dim * layers, hid_dim, num_heads, args)
            self.trans_layer = MultiHead_GAT_Attention(g, hid_dim, out_dim, num_heads, args)
        else:
            self.layers.append(MultiHead_GAT_Attention(g, in_feat, hid_dim, num_heads, args))
            for lay in range(layers - 1):
                self.layers.append(MultiHead_GAT_Attention(g, hid_dim, hid_dim, num_heads, args))
            self.outlayer = MultiHead_GAT_Attention(g, hid_dim, out_dim, num_heads, args, mode='mean')
        self.dropout = nn.Dropout(args.dropout)
        self.in_drop = args.in_drop
        self.softmax = nn.Softmax(dim=1)
        self.res = args.dense_net

    def forward(self, feature):
        used_feature = None
        for i, layer in enumerate(self.layers):
            if i == 0 and self.in_drop:
                feature = self.dropout(feature)
            elif i > 0:
                feature = self.dropout(feature)
            out_feature = layer(feature)
            if used_feature is None:
                used_feature = out_feature
            else:
                used_feature = torch.cat((used_feature, out_feature), 1)
            if self.res:
                feature = used_feature
            else:
                feature = out_feature
        out = self.outlayer(feature)
        if self.res:
            out = self.trans_layer(out)
        out = self.softmax(out)
        return out


def main(args):
    if not args.res_add:
        args.re_zero = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = CoraGraphDataset()
    graph = data[0]
    graph = graph.to(device)
    features = graph.ndata['feat']
    labels = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    graph = my_add_self_loop(graph)
    net = GCNNet(graph, features.shape[1], args)
    print(getModelSize(net))
    # net = GAT(graph,features.shape[1],args.hid_dim,7,args.num_heads)
    if args.re_zero:
        rate_params = []
        other_params = []
        for name, parameters in net.named_parameters():
            if name.endswith("rate"):
                rate_params += [parameters]
            else:
                other_params += [parameters]
        optimizer = Adam(
            [
                {'params': other_params},
                {'params': rate_params, 'lr': args.re_zero_lr}
            ],
            lr=args.lr
        )
    else:
        optimizer = Adam(net.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=3)
    lossF = CrossEntropyLoss()
    cur_lr_list = []
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []
    net.to(device)
    for m in net.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
            torch.nn.init.xavier_uniform_(m.weight)
    epoches = args.epoch
    writer = SummaryWriter('./path/log/layer2_res_rezero_dense')
    with tqdm(total=epoches) as t:
        for epoch in range(epoches):
            t.set_description('Epoch {%d}' % (epoch + 1))
            cur_lr = optimizer.param_groups[0]['lr']
            cur_lr_list.append(cur_lr)
            net.train()
            outs = net(features)
            loss = lossF(outs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            acc, val_loss = evaluate(net, features, graph, labels, val_mask, lossF)
            train_loss_list.append(loss.item())
            val_loss_list.append(val_loss)
            val_acc_list.append(acc)
            t.set_postfix(train_loss="{:.5f}".format(loss.item()), val_loss="{:.5f}".format(val_loss), Accuracy=acc,
                          lr="{:.3e}".format(cur_lr))
            writer.add_scalar('train_loss', loss.item(), epoch)
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('Accuracy', acc, epoch)
            writer.add_scalar('lr', cur_lr, epoch)
            for name, layer in net.named_parameters():
                writer.add_histogram(name + '_grad_normal', layer.grad, epoch)
                writer.add_histogram(name + '_data_normal', layer, epoch)
            sleep(0.1)
            t.update(1)
    acc, _ = evaluate(net, features, graph, labels, test_mask, lossF)
    file_save_name = r"./path/checkpoints/acc-{:.2%},layers-{:d}_lr-{:.2e}_hid_dim-{:d}_res_rezero_dense".format(acc,
                                                                                                                 args.num_layers,
                                                                                                                 args.lr,
                                                                                                                 args.hid_dim)
    torch.save(net.state_dict(), f=file_save_name + ".pth")
    print("Test accuracy {:.2%}".format(acc))
    x_list = list(range(len(cur_lr_list)))
    plt.subplot(221)
    plt.plot(x_list, cur_lr_list)
    plt.title('lr')
    plt.subplot(222)
    plt.plot(x_list, train_loss_list)
    plt.title('train')
    plt.subplot(223)
    plt.plot(x_list, val_loss_list)
    plt.title('val_loss')
    plt.subplot(224)
    plt.plot(x_list, val_acc_list)
    plt.title('val_acc')
    plt.savefig(file_save_name + ".jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument('--hid_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--re_zero_lr', type=float, default=0.005)
    parser.add_argument('--dense_dim', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--res_add', type=bool, default=True)
    parser.add_argument('--dense_net', type=bool, default=False)
    parser.add_argument('--re_zero', type=bool, default=True)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=1e-4)
    parser.add_argument('--in_drop', type=bool, default=False)
    parser.add_argument('--epoch', type=int, default=500)
    args = parser.parse_args()
    main(args)
1
