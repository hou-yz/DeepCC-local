from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import *
from sampler import *
import random
import time
from torch.autograd import Variable
import h5py
from torch.nn import init
import numpy, scipy.io

import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt


class MetricNet(nn.Module):
    def __init__(self, num_class=0):
        super(MetricNet, self).__init__()
        self.num_class = num_class
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 128)
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
        epoch_max=1
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
                output = F.softmax(output, dim=1)
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


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Hyper Score')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--step-size', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, val set alone for validation")
    parser.add_argument('--momentum', type=float, default=0, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--use_AM', action='store_true')
    parser.add_argument('--save_result', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--data-path', type=str, default='~/Data/DukeMTMC/ground_truth/',
                        metavar='PATH')
    parser.add_argument('-L', type=str, default='L2', choices=['L1', 'L2'])
    parser.add_argument('--window', type=str, default='300', choices=['Inf', '150', '300', '1500'])
    parser.add_argument('--log-dir', type=str, default='', metavar='PATH')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    if '~' in args.data_path:
        args.data_path = os.path.expanduser(args.data_path)
    if args.combine_trainval:
        train_data_path = args.data_path + 'hyperGT_{}_trainval_{}.h5'.format(args.L, args.window)
    else:
        train_data_path = args.data_path + 'hyperGT_{}_train_{}.h5'.format(args.L, args.window)
    if args.save_result:
        test_data_path = args.data_path + 'hyperGT_{}_train_{}.h5'.format(args.L, args.window)
    else:
        test_data_path = args.data_path + 'hyperGT_{}_val_{}.h5'.format(args.L, args.window)
    args.log_dir = 'logs/{}/appear_only/'.format(args.L, ) + args.log_dir
    torch.manual_seed(args.seed)
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)

    trainset = SiameseHyperFeat(HyperFeat(train_data_path))
    testset = SiameseHyperFeat(HyperFeat(test_data_path),train=False)
    train_loader = DataLoader(trainset, batch_size=args.batch_size,
                              num_workers=4, pin_memory=True, shuffle=True)

    test_loader = DataLoader(testset, batch_size=args.batch_size,
                             # sampler=HyperScoreSampler(testset, 1024),
                             num_workers=4, pin_memory=True)

    metric_net = nn.DataParallel(MetricNet(num_class=2)).cuda()
    if args.resume:
        checkpoint = torch.load(args.log_dir + '/metric_net_{}_{}.pth.tar'.format(args.L, args.window))
        model_dict = checkpoint['state_dict']
        metric_net.module.load_state_dict(model_dict)

    appear_motion_net = nn.DataParallel(AppearMotionNet()).cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    if args.train:
        # Draw Curve
        x_epoch = []
        train_loss_s = []
        train_prec_s = []
        test_loss_s = []
        test_prec_s = []
        optimizer = optim.SGD(metric_net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.005)
        if not args.resume:
            for epoch in range(1, args.epochs + 1):
                train_loss, train_prec = train(args, metric_net, appear_motion_net, train_loader, optimizer, epoch,
                                               criterion)
                test_loss, test_prec = test(args, metric_net, appear_motion_net, test_loader, criterion)
                x_epoch.append(epoch)
                train_loss_s.append(train_loss)
                train_prec_s.append(train_prec)
                test_loss_s.append(test_loss)
                test_prec_s.append(test_prec)
                draw_curve(args, x_epoch, train_loss_s, train_prec_s, test_loss_s, test_prec_s)
                pass
            torch.save({'state_dict': metric_net.module.state_dict(), }, args.log_dir + '/metric_net_{}_{}.pth.tar'.
                       format(args.L, args.window))
        else:
            test(args, metric_net, appear_motion_net, test_loader, criterion, )

        x_epoch = []
        train_loss_s = []
        train_prec_s = []
        test_loss_s = []
        test_prec_s = []
        # train appear_motion_net
        optimizer = optim.SGD(metric_net.parameters(), lr=0.1 * args.lr, momentum=args.momentum)
        if args.use_AM:
            for epoch in range(1, args.epochs + 1):
                train_loss, train_prec = train(args, metric_net, appear_motion_net, train_loader, optimizer, epoch,
                                               criterion, train_motion=True)
                test_loss, test_prec = test(args, metric_net, appear_motion_net, test_loader, criterion,
                                            test_motion=True)
                x_epoch.append(epoch)
                train_loss_s.append(train_loss)
                train_prec_s.append(train_prec)
                test_loss_s.append(test_loss)
                test_prec_s.append(test_prec)
                draw_curve(args, x_epoch, train_loss_s, train_prec_s, test_loss_s, test_prec_s, train_motion=True)
                pass
            torch.save({'state_dict': appear_motion_net.module.state_dict(), },
                       args.log_dir + '/appear_motion_net_{}_{}.pth.tar'.format(args.L, args.window))
        if args.use_AM:
            save_model_as_mat(args, metric_net.module, appear_motion_net.module)
        else:
            save_model_as_mat(args, metric_net.module, [])

    checkpoint = torch.load(args.log_dir + '/metric_net_{}_{}.pth.tar'.format(args.L, args.window))
    model_dict = checkpoint['state_dict']
    metric_net.module.load_state_dict(model_dict)
    if args.use_AM:
        checkpoint = torch.load(args.log_dir + '/appear_motion_net_{}_{}.pth.tar'.format(args.L, args.window))
        model_dict = checkpoint['state_dict']
        appear_motion_net.module.load_state_dict(model_dict)
    test(args, metric_net, appear_motion_net, test_loader, criterion,
         test_motion=args.use_AM, save_result=args.save_result, epoch_max=100)


if __name__ == '__main__':
    main()
