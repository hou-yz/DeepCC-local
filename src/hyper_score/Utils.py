import torch.nn as nn
import torch.nn.functional as F
from sampler import *
from models import *
import time
import h5py
from torch.nn import init
import scipy.io

import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt


def save_model_as_mat(path, metric_net):
    fc1_w, fc1_b = metric_net.fc1.weight.data.cpu().numpy(), metric_net.fc1.bias.data.cpu().numpy()
    fc2_w, fc2_b = metric_net.fc2.weight.data.cpu().numpy(), metric_net.fc2.bias.data.cpu().numpy()
    fc3_w, fc3_b = metric_net.fc3.weight.data.cpu().numpy(), metric_net.fc3.bias.data.cpu().numpy()
    out_w, out_b = metric_net.out_layer.weight.data.cpu().numpy(), metric_net.out_layer.bias.data.cpu().numpy()

    scipy.io.savemat(path, mdict={'fc1_w': fc1_w, 'fc1_b': fc1_b,
                                  'fc2_w': fc2_w, 'fc2_b': fc2_b,
                                  'fc3_w': fc3_w, 'fc3_b': fc3_b,
                                  'out_w': out_w, 'out_b': out_b, })


def addzero(x, insert_pos, num_zero):
    x1, x2 = x[:, 0:insert_pos], x[:, insert_pos:]
    z = torch.zeros([x.shape[0], num_zero]).cuda()
    return torch.cat((x1, z.type_as(x1), x2), dim=1)


def train(args, metric_net, train_loader, optimizer, epoch, criterion):
    # Schedule learning rate
    lr = args.lr * (0.1 ** (epoch // args.step_size))
    for g in optimizer.param_groups:
        g['lr'] = lr * g.get('lr_mult', 1)

    losses = 0
    correct = 0
    miss = 0
    metric_net.train()
    t0 = time.time()
    for batch_idx, (feat1, feat2, motion_score, target) in enumerate(train_loader):
        l = target.shape[0]
        # data = torch.cat(((feat2.cuda() - feat1.cuda()).abs().float(), motion_score.cuda().float().view(-1, 1)), dim=1)
        data = (feat2.cuda() - feat1.cuda()).abs().float()
        target = target.cuda().long()

        output = metric_net(data)
        pred = torch.argmax(output, 1)
        correct += pred.eq(target).sum().item()
        miss += l - pred.eq(target).sum().item()
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.item()
        if (batch_idx + 1) % args.log_interval == 0:
            t1 = time.time()
            t_batch = t1 - t0
            t0 = time.time()
            print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                epoch, (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_batch))

    return losses / (batch_idx + 1), correct / (correct + miss)


def test(args, metric_net, test_loader, criterion, save_result=False, epoch_max=1):
    metric_net.eval()
    losses = 0
    correct = 0
    miss = 0
    lines = torch.zeros([0]).cuda()
    t0 = time.time()
    if not save_result:
        epoch_max = 1
    for epoch in range(epoch_max):
        for batch_idx, (feat1, feat2, motion_score, target) in enumerate(test_loader):

            l = target.shape[0]
            # data = torch.cat(((feat2.cuda() - feat1.cuda()).abs().float(), motion_score.cuda().float().view(-1, 1)), dim=1)
            data = (feat2.cuda() - feat1.cuda()).abs().float()
            target = target.cuda().long()
            with torch.no_grad():
                output = metric_net(data)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += l - pred.eq(target).sum().item()
            loss = criterion(output, target)
            losses += loss.item()
            output = F.softmax(0.1 * output, dim=1)
            line = torch.cat(((output[:, 1] - output[:, 0]).view(-1, 1),
                              motion_score.view(-1, 1).cuda().float(),
                              target.view(-1, 1).float()), dim=1)
            lines = torch.cat((lines, line), dim=0)
            if (batch_idx + 1) % args.log_interval == 0:
                t1 = time.time()
                t_batch = t1 - t0
                t0 = time.time()
                print('Test on val, epoch:{}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_batch))

    lines = lines.cpu().numpy()
    if save_result:
        output_fname = args.data_path + '/results_{}_{}_train_Inf.h5'.format(args.L, args.window)
        with h5py.File(output_fname, 'w') as f:
            mat_data = np.vstack(lines)
            f.create_dataset('emb', data=mat_data, dtype=float)
            pass
    return losses / (batch_idx + 1), correct / (correct + miss)


def draw_curve(path, x_epoch, train_loss, train_prec, test_loss, test_prec):
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="prec")
    ax0.plot(x_epoch, train_loss, 'bo-', label='train')
    ax0.plot(x_epoch, test_loss, 'ro-', label='test')
    ax1.plot(x_epoch, train_prec, 'bo-', label='train')
    ax1.plot(x_epoch, test_prec, 'ro-', label='test')
    ax0.legend()
    ax1.legend()
    fig.savefig(path)
