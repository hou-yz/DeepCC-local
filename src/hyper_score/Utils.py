import torch.nn as nn
import torch.nn.functional as F
from sampler import *
import time
import h5py
from torch.nn import init
import scipy.io

import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt


class MetricNet(nn.Module):
    def __init__(self, feature_dim=256, num_class=0):
        super(MetricNet, self).__init__()
        self.num_class = num_class
        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.out_layer = nn.Linear(128, self.num_class)
        init.normal_(self.out_layer.weight, std=0.001)
        init.constant_(self.out_layer.bias, 0)

    def forward(self, x):
        # feat = x[:, 0:-1]
        # motion_score = x[:, -1].view(-1, 1)
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        # out = torch.cat((out, motion_score), dim=1)
        out = self.out_layer(out)
        return out


class AppearMotionNet(nn.Module):
    def __init__(self):
        super(AppearMotionNet, self).__init__()
        self.fc4 = nn.Linear(2, 2)

    def forward(self, x):
        out = self.fc4(x)
        return out


def save_model_as_mat(args, metric_net, appear_motion_net):
    fc1_w, fc1_b = metric_net.fc1.weight.data.cpu().numpy(), metric_net.fc1.bias.data.cpu().numpy()
    fc2_w, fc2_b = metric_net.fc2.weight.data.cpu().numpy(), metric_net.fc2.bias.data.cpu().numpy()
    fc3_w, fc3_b = metric_net.fc3.weight.data.cpu().numpy(), metric_net.fc3.bias.data.cpu().numpy()
    out_w, out_b = metric_net.out_layer.weight.data.cpu().numpy(), metric_net.out_layer.bias.data.cpu().numpy()

    if isinstance(appear_motion_net, AppearMotionNet):
        fc4_w, fc4_b = appear_motion_net.fc4.weight.data.cpu().numpy(), appear_motion_net.fc4.bias.data.cpu().numpy()
        scipy.io.savemat(args.log_dir + '/model_param_{}_{}.mat'.format(args.L, args.window),
                         mdict={'fc1_w': fc1_w, 'fc1_b': fc1_b,
                                'fc2_w': fc2_w, 'fc2_b': fc2_b,
                                'fc3_w': fc3_w, 'fc3_b': fc3_b,
                                'out_w': out_w, 'out_b': out_b,
                                'fc4_w': fc4_w, 'fc4_b': fc4_b, })
    else:
        scipy.io.savemat(args.log_dir + '/model_param_{}_{}.mat'.format(args.L, args.window),
                         mdict={'fc1_w': fc1_w, 'fc1_b': fc1_b,
                                'fc2_w': fc2_w, 'fc2_b': fc2_b,
                                'fc3_w': fc3_w, 'fc3_b': fc3_b,
                                'out_w': out_w, 'out_b': out_b, })


def addzero(x, insert_pos, num_zero):
    x1, x2 = x[:, 0:insert_pos], x[:, insert_pos:]
    z = torch.zeros([x.shape[0], num_zero]).cuda()
    return torch.cat((x1, z.type_as(x1), x2), dim=1)


def train(args, metric_net, appear_motion_net, train_loader, optimizer, epoch, criterion, train_motion=False):
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
        if not train_motion:
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
        else:
            metric_net.eval()
            l = target.shape[0]
            # data = torch.cat(((feat2.cuda() - feat1.cuda()).abs().float(), motion_score.cuda().float().view(-1, 1)), dim=1)
            data = (feat2.cuda() - feat1.cuda()).abs().float()
            target = target.cuda().long()
            output = metric_net(data)
            output = (output[:, 1] - 0.5) * 2
            data = torch.cat((output.view(-1, 1), motion_score.cuda().float().view(-1, 1)), dim=1)

            # train appear_motion_net
            appear_motion_net.train()
            output = appear_motion_net(data)
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


def test(args, metric_net, appear_motion_net, test_loader, criterion, test_motion=False, save_result=False,
         epoch_max=1):
    metric_net.eval()
    appear_motion_net.eval()
    # test_loss = 0
    # correct = 0
    # miss = 0
    # lines = torch.zeros([0]).cuda()
    # with torch.no_grad():
    #     for batch_idx in range(1, test_loader.dataset.num_spatialGroup + 1):
    #         for (feat, pid, spaGrpID) in test_loader:
    #             l = pid.shape[0]
    #             spaGrpID = int(np.unique(spaGrpID))
    #             feat1, feat2, target = feat.cuda(), feat.cuda(), pid.cuda()
    #             # if args.L2_speed == 'mid':
    #             #     pass
    #             # else:
    #             #     # iCam,centerFrame,startFrame,endFrame,startpoint, endpoint,head_velocity,tail_velocity
    #             #     seq1, seq2 = [0, 2, 6, 7, 10, 11, ], [0, 3, 4, 5, 8, 9, ]
    #             #     seq1.extend(range(12, 268)), seq2.extend(range(12, 268))
    #             #     feat1, feat2 = feat1[:, seq1], feat2[:, seq2]
    #             # data = (addzero(feat1, 4, 2).unsqueeze(0).expand(l, l, 264) -
    #             #         addzero(feat2, 6, 2).unsqueeze(1).expand(l, l, 264))
    #             data = (feat1.unsqueeze(0).expand(l, l, 256) - feat2.unsqueeze(1).expand(l, l, 256)).abs()
    #             # data[:, 0:8] = 0
    #             # data[:, :, [4, 5]] = -data[:, :, [4, 5]]
    #             target = (target.unsqueeze(0).expand(l, l) - target.unsqueeze(1).expand(l, l)) == 0
    #             target[torch.eye(l).cuda().byte()] = 1
    #             # index = torch.ones(l, l).triu().byte()
    #             data, target = data.view(-1, 256).float(), target.view(-1).long()
    #             # data = torch.cat((data[:, 0:8], torch.norm(data[:, 8:], 2, dim=1).view(-1, 1)), dim=1)
    #
    #             output = metric_net(data)
    #             test_loss += criterion(output, target).item()  # sum up batch loss
    #             _, pred = torch.max(output, 1)  # get the index of the max log-probability
    #             correct += pred.eq(target).sum().item()
    #             miss += target.shape[0] - pred.eq(target).sum().item()
    #             output = F.softmax(output, dim=1)
    #             line = (output[:, 1] - 0.5) * 2
    #             lines = torch.cat((lines, line), dim=0)
    #             pass
    #
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.
    #       format(test_loss, correct, correct + miss, 100. * correct / (correct + miss)))
    #
    #
    # lines = lines.cpu().numpy()
    # if save_result:
    #     output_fname = args.data_path + '/pairwise_dis_tmp.h5'
    #     with h5py.File(output_fname, 'w') as f:
    #         mat_data = np.vstack(lines)
    #         f.create_dataset('dis', data=mat_data, dtype=float)
    #         pass

    losses = 0
    correct = 0
    miss = 0
    lines = torch.zeros([0]).cuda()
    t0 = time.time()
    if not save_result:
        epoch_max = 1
    for epoch in range(epoch_max):
        for batch_idx, (feat1, feat2, motion_score, target) in enumerate(test_loader):
            if not test_motion:
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
                line = torch.cat((output[:, 1].view(-1, 1),
                                  motion_score.view(-1, 1).cuda().float(),
                                  target.view(-1, 1).float()), dim=1)
                lines = torch.cat((lines, line), dim=0)
                if (batch_idx + 1) % args.log_interval == 0:
                    t1 = time.time()
                    t_batch = t1 - t0
                    t0 = time.time()
                    print('Test on val, epoch:{}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                        epoch, (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_batch))
            else:
                metric_net.eval()
                l = target.shape[0]
                # data = torch.cat(((feat2.cuda() - feat1.cuda()).abs().float(), motion_score.cuda().float().view(-1, 1)), dim=1)
                data = (feat2.cuda() - feat1.cuda()).abs().float()
                target = target.cuda().long()
                output = metric_net(data)
                output = (output[:, 1] - 0.5) * 2
                data = torch.cat((output.view(-1, 1), motion_score.cuda().float().view(-1, 1)), dim=1)

                # test appear_motion_net
                output = appear_motion_net(data)
                pred = torch.argmax(output, 1)
                correct += pred.eq(target).sum().item()
                miss += l - pred.eq(target).sum().item()
                loss = criterion(output, target)
                losses += loss.item()
                if (batch_idx + 1) % args.log_interval == 0:
                    t1 = time.time()
                    t_batch = t1 - t0
                    t0 = time.time()
                    print('Test on val, epoch:{}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                        epoch, (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_batch))
    lines = lines.cpu().numpy()
    if save_result:
        output_fname = args.data_path + '/target_data.h5'
        with h5py.File(output_fname, 'w') as f:
            mat_data = np.vstack(lines)
            f.create_dataset('emb', data=mat_data, dtype=float)
            pass
    return losses / (batch_idx + 1), correct / (correct + miss)


def draw_curve(args, x_epoch, train_loss, train_prec, test_loss, test_prec, train_motion=False):
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="prec")
    ax0.plot(x_epoch, train_loss, 'bo-', label='train')
    ax0.plot(x_epoch, test_loss, 'ro-', label='test')
    ax1.plot(x_epoch, train_prec, 'bo-', label='train')
    ax1.plot(x_epoch, test_prec, 'ro-', label='test')
    ax0.legend()
    ax1.legend()
    if not train_motion:
        fig.savefig(args.log_dir + '/MetricNet_{}_{}.jpg'.format(args.L, args.window))
    else:
        fig.savefig(args.log_dir + '/AppearMotionNet_{}_{}.jpg'.format(args.L, args.window))
