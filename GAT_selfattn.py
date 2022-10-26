import argparse
import math

import dgl
import dgl.function as fn
from torch import nn
from torch import FloatTensor
from torch.nn import LayerNorm
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


def self_attention(q, k, v) -> torch.Tensor:
    d_k = q.size(-1)
    h = torch.matmul(q, k.transpose(-1, -2))
    p_attn = torch.softmax(h / math.sqrt(d_k), dim=-1)
    return torch.matmul(p_attn, v)


class MultiHead_SelfAttention(nn.Module):
    def __init__(self, d_model, args, num_heads=8, hid_dim=1024):
        super(MultiHead_SelfAttention, self).__init__()
        if args.attn_hid_dim:
            hid_dim = args.attn_hid_dim
        if args.num_heads:
            num_heads = args.num_heads
        self.queries = nn.Linear(d_model, hid_dim)
        self.keys = nn.Linear(d_model, hid_dim)
        self.values = nn.Linear(d_model, hid_dim)
        self.heads = num_heads
        self.d_k = int(hid_dim / num_heads)
        self.projection = nn.Linear(hid_dim,d_model)

    def forward(self, query, x):
        batch = x.size()[0]
        num_nodes = x.size()[1]
        q = self.queries(query)
        k = self.keys(x)
        v = self.values(x)
        q = q.view(batch, -1, self.d_k).unsqueeze(1).transpose(1,2)
        k, v = [x.view(batch, -1, self.heads, self.d_k).transpose(1,2) for x in (k, v)]
        out = self_attention(q, k, v)
        out = out.transpose(1,2).contiguous().view(batch, 1, self.heads * self.d_k).squeeze(1)
        return self.projection(out)


class GCN_Attention(nn.Module):
    def __init__(self, g, in_feat, out_feat, args):
        super(GCN_Attention, self).__init__()
        self.outlinear = nn.Linear(in_feat, out_feat)
        # self.attn_fc = nn.Linear(out_feat * 2, 1, bias=False)
        # self.attn_activ = nn.GELU()
        self.mult_selfattn = MultiHead_SelfAttention(out_feat, args)
        # self.softmax = Softmax(1)
        self.graph = g
        self.res = args.res_add
        self.rezero = args.re_zero
        self.layer_norm = LayerNorm(out_feat)
        self.activ = nn.GELU()
        if args.res_add:
            if args.re_zero:
                self.rate = torch.nn.Parameter(torch.tensor(0, dtype=torch.float32))
            self.res_linear = nn.Linear(in_feat, out_feat)

    # def edge_attn(self, edges):
    #     z = torch.cat((edges.src['z'], edges.dst['z']), 1)
    #     a = self.attn_fc(z)
    #     return {'e': self.attn_activ(a)}

    def msg_func(self, edges):
        return {'m': edges.src['z']}

    def reducer_func(self, nodes):
        # a = self.softmax(nodes.mailbox['e'])
        # h = torch.sum(a * nodes.mailbox['z'], 1)
        # print(nodes.data['z'].shape)
        h = self.mult_selfattn(nodes.data['z'],nodes.mailbox['m'])
        return {'h': h}

    def forward(self, feature):
        z = self.outlinear(feature)
        self.graph.ndata['z'] = z
        # self.graph.apply_edges(self.edge_attn)
        self.graph.update_all(self.msg_func, self.reducer_func)
        out_feature = self.graph.ndata.pop('h')
        out_feature = self.layer_norm(self.activ(out_feature))
        if self.res:
            h0 = self.res_linear(feature)
            if self.rezero:
                out_feature = out_feature + h0 * self.rate
            else:
                out_feature = out_feature + h0
        return out_feature


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
            self.layers.append(GCN_Attention(g, in_feat, dense_dim, args))
            for lay in range(layers - 1):
                self.layers.append(GCN_Attention(g, dense_dim * (lay + 1), dense_dim, args))
            self.outlayer = GCN_Attention(g, dense_dim * layers, hid_dim, args)
            self.trans_layer = GCN_Attention(g, hid_dim, out_dim, args)
        else:
            self.layers.append(GCN_Attention(g, in_feat, hid_dim, args))
            for lay in range(layers - 1):
                self.layers.append(GCN_Attention(g, hid_dim, hid_dim, args))
            self.outlayer = GCN_Attention(g, hid_dim, out_dim, args)
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
            if self.res:
                if used_feature is None:
                    used_feature = out_feature
                else:
                    used_feature = torch.cat((used_feature, out_feature), 1)
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
            lr=args.lr,
            weight_decay=args.l2
        )
    else:
        optimizer = Adam(net.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=3)
    lossF = CrossEntropyLoss()
    cur_lr_list = []
    train_loss_list = []
    train_acc_list = []
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
            loss = lossF(outs[train_mask], labels[train_mask])
            with torch.no_grad():
                _, indices = torch.max(outs[train_mask], dim=1)
                correct = torch.sum(indices == labels[train_mask])
                train_acc = correct.item() * 1.0 / len(labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            acc, val_loss = evaluate(net, features, graph, labels, val_mask, lossF)
            train_loss_list.append(loss.item())
            val_loss_list.append(val_loss)
            val_acc_list.append(acc)
            train_acc_list.append(train_acc)
            t.set_postfix(train_loss="{:.5f}".format(loss.item()), val_loss="{:.5f}".format(val_loss), Val_Acc=acc,
                          lr="{:.3e}".format(cur_lr), train_Acc=train_acc)
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
    file_save_name = r"./path/checkpoints/acc-{:.2%},layers-{:d}_lr-{:.2e}_hid_dim-{:d}_normal".format(acc,
                                                                                                                 args.num_layers,
                                                                                                                 args.lr,
                                                                                                                 args.hid_dim)
    torch.save(net.state_dict(), f=file_save_name + ".pth")
    print("Test accuracy {:.2%}".format(acc))
    x_list = list(range(len(cur_lr_list)))
    plt.subplot(321)
    plt.plot(x_list, cur_lr_list)
    plt.title('lr')
    plt.subplot(322)
    plt.plot(x_list, train_loss_list)
    plt.title('train_loss')
    plt.subplot(323)
    plt.plot(x_list, train_acc_list)
    plt.title('train_acc')
    plt.subplot(324)
    plt.plot(x_list, val_loss_list)
    plt.title('val_loss')
    plt.subplot(325)
    plt.plot(x_list, val_acc_list)
    plt.title('val_acc')
    plt.savefig(file_save_name + ".jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument('--hid_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=16)
    parser.add_argument('--re_zero_lr', type=float, default=0.005)
    parser.add_argument('--dense_dim', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--res_add', type=bool, default=True)
    parser.add_argument('--dense_net', type=bool, default=False)
    parser.add_argument('--re_zero', type=bool, default=True)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--in_drop', type=bool, default=False)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--attn_hid_dim', type=int, default=256)
    args = parser.parse_args()
    main(args)
1
