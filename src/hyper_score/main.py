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


class Net(nn.Module):
    def __init__(self, num_class=0):
        super(Net, self).__init__()
        self.num_class = num_class
        self.fc1 = nn.Linear(264, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 128)
        if self.num_class > 0:
            if self.num_class == 1:
                self.out_layer = nn.Sequential(nn.Linear(128, self.num_class), nn.Sigmoid())
            else:
                self.out_layer = nn.Linear(128, self.num_class)
                init.normal_(self.out_layer.weight, std=0.001)
                init.constant_(self.out_layer.bias, 0)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        if self.num_class > 0:
            out = self.out_layer(out)
        return out


def save_model_as_mat(args, model):
    fc1_w, fc1_b = model.fc1.weight.data.cpu().numpy(), model.fc1.bias.data.cpu().numpy()
    fc2_w, fc2_b = model.fc2.weight.data.cpu().numpy(), model.fc2.bias.data.cpu().numpy()
    fc3_w, fc3_b = model.fc3.weight.data.cpu().numpy(), model.fc3.bias.data.cpu().numpy()
    out_w, out_b = model.out_layer.weight.data.cpu().numpy(), model.out_layer.bias.data.cpu().numpy()

    scipy.io.savemat(args.log_dir + '/model_param_{}_{}.mat'.format(args.L2_window, args.L2_speed),
                     mdict={'fc1_w': fc1_w, 'fc1_b': fc1_b,
                            'fc2_w': fc2_w, 'fc2_b': fc2_b,
                            'fc3_w': fc3_w, 'fc3_b': fc3_b,
                            'out_w': out_w, 'out_b': out_b, })


def addzero(x, insert_pos, num_zero):
    x1, x2 = x[:, 0:insert_pos], x[:, insert_pos:]
    z = torch.zeros([x.shape[0], num_zero]).cuda()
    return torch.cat((x1, z.type_as(x1), x2), dim=1)


def train(args, model, train_loader, optimizer, epoch, criterion):
    # Schedule learning rate

    if args.epochs == 20:
        step_size = 10
    else:
        step_size = args.step_size
    lr = args.lr * (0.1 ** (epoch // step_size))
    for g in optimizer.param_groups:
        g['lr'] = lr * g.get('lr_mult', 1)

    losses = 0
    correct = 0
    miss = 0
    model.train()
    t0 = time.time()
    for batch_idx, (feat1, feat2, target) in enumerate(train_loader):
        l = target.shape[0]
        if args.L2_speed == 'mid':
            pass
        else:
            # iCam,centerFrame,startFrame,endFrame,startpoint, endpoint,head_velocity,tail_velocity
            seq1, seq2 = [0, 2, 6, 7, 10, 11, ], [0, 3, 4, 5, 8, 9, ]
            seq1.extend(range(12, 268)), seq2.extend(range(12, 268))
            feat1, feat2 = feat1[:, seq1], feat2[:, seq2]
        data = (addzero(feat2.cuda(), 4, 2) - addzero(feat1.cuda(), 6, 2)).float()
        data = data.abs()
        # data[:, [4, 5]] = -data[:, [4, 5]]

        target = target.cuda().long()
        # data = torch.cat((data[:, 0:8], torch.norm(data[:, 8:], 2, dim=1).view(-1, 1)), dim=1)

        optimizer.zero_grad()
        output = model(data)
        pred = torch.argmax(output, 1)
        correct += pred.eq(target).sum().item()
        miss += l - pred.eq(target).sum().item()
        loss = criterion(output, target)
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


def test(args, model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    miss = 0
    lines = torch.zeros([0]).cuda()
    with torch.no_grad():
        for batch_idx in range(1, test_loader.dataset.num_spatialGroup + 1):
            for (feat, pid, spaGrpID) in test_loader:
                l = pid.shape[0]
                spaGrpID = int(np.unique(spaGrpID))
                feat1, feat2, target = feat.cuda(), feat.cuda(), pid.cuda()

                if args.L2_speed == 'mid':
                    pass
                else:
                    # iCam,centerFrame,startFrame,endFrame,startpoint, endpoint,head_velocity,tail_velocity
                    seq1, seq2 = [0, 2, 6, 7, 10, 11, ], [0, 3, 4, 5, 8, 9, ]
                    seq1.extend(range(12, 268)), seq2.extend(range(12, 268))
                    feat1, feat2 = feat1[:, seq1], feat2[:, seq2]
                data = (addzero(feat1, 4, 2).unsqueeze(0).expand(l, l, 264) -
                        addzero(feat2, 6, 2).unsqueeze(1).expand(l, l, 264))
                data = data.abs()
                # data[:, :, [4, 5]] = -data[:, :, [4, 5]]
                target = (target.unsqueeze(0).expand(l, l) - target.unsqueeze(1).expand(l, l)) == 0
                target[torch.eye(l).cuda().byte()] = 1
                index = torch.ones(l, l).triu().byte()
                data, target = data[index, :].view(-1, 264).float(), target[index].long()
                # data = torch.cat((data[:, 0:8], torch.norm(data[:, 8:], 2, dim=1).view(-1, 1)), dim=1)

                output = model(data)
                test_loss += criterion(output, target).item()  # sum up batch loss
                _, pred = torch.max(output, 1)  # get the index of the max log-probability
                correct += pred.eq(target).sum().item()
                miss += target.shape[0] - pred.eq(target).sum().item()
                output = F.softmax(output, dim=1)
                line = (output[:, 1] - 0.5) * 2
                lines = torch.cat((lines, line), dim=0)
                pass

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.
          format(test_loss, correct, correct + miss, 100. * correct / (correct + miss)))

    lines = lines.cpu().numpy()
    if args.save_result:
        output_fname = osp.dirname(args.data_path) + '/pairwise_dis_tmp.h5'
        with h5py.File(output_fname, 'w') as f:
            mat_data = np.vstack(lines)
            f.create_dataset('dis', data=mat_data, dtype=float)
            pass


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Hyper Score')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--step-size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--save_result', action='store_true')
    # parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--data-path', type=str, default='~/Data/DukeMTMC/ground_truth/',
                        metavar='PATH')
    parser.add_argument('--L2_window', type=int, default=300, choices=[150, 300, 1500])  # bad performance for 1200
    parser.add_argument('--L2_speed', type=str, default='mid', choices=['mid', 'head-tail'])
    parser.add_argument('--log-dir', type=str, default='logs', metavar='PATH')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    if '~' in args.data_path:
        args.data_path = os.path.expanduser(args.data_path)
    args.data_path = args.data_path + 'hyperGT_trainval_mini_{}_{}.h5'.format(args.L2_window, args.L2_speed)
    torch.manual_seed(args.seed)
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)

    dataset = HyperFeat(args.data_path, args.L2_speed)
    train_loader = DataLoader(SiameseHyperFeat(dataset), batch_size=args.batch_size,
                              num_workers=4, pin_memory=True, shuffle=True)

    test_loader = DataLoader(dataset, batch_size=args.batch_size,
                             sampler=HyperScoreSampler(dataset, 1024),
                             num_workers=0, pin_memory=True)

    model = Net(num_class=2)
    model = nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if args.train:
        # Draw Curve
        x_epoch = []
        fig = plt.figure()
        ax0 = fig.add_subplot(121, title="loss")
        ax1 = fig.add_subplot(122, title="prec")
        loss_s = []
        prec_s = []

        def draw_curve(current_epoch, train_loss, train_prec):
            x_epoch.append(current_epoch)
            ax0.plot(x_epoch, train_loss, 'bo-', label='train')
            ax1.plot(x_epoch, train_prec, 'bo-', label='train')
            if current_epoch == 0:
                ax0.legend()
                ax1.legend()
            fig.savefig(args.log_dir + '/train_{}_{}.jpg'.format(args.L2_window, args.L2_speed))

        for epoch in range(1, args.epochs + 1):
            loss, prec = train(args, model, train_loader, optimizer, epoch, criterion)
            loss_s.append(loss)
            prec_s.append(prec)
            draw_curve(epoch, loss_s, prec_s)
            pass
        torch.save({'state_dict': model.module.state_dict(), }, args.log_dir + '/checkpoint_{}_{}.pth.tar'.
                   format(args.L2_window, args.L2_speed))
        save_model_as_mat(args, model.module)

    checkpoint = torch.load(args.log_dir + '/checkpoint_{}_{}.pth.tar'.format(args.L2_window, args.L2_speed))
    model_dict = checkpoint['state_dict']
    model.module.load_state_dict(model_dict)
    test(args, model, test_loader, criterion)


if __name__ == '__main__':
    main()
