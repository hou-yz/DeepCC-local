from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import *

from Utils import *


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
    parser.add_argument('--data-path', type=str, default='1fps_train_IDE_40/',
                        metavar='PATH')
    parser.add_argument('-L', type=str, default='L2', choices=['L1', 'L2'])
    # parser.add_argument('--tracklet', type=int, default=20, choices=[20, 40])
    parser.add_argument('--window', type=str, default='300', choices=['Inf','75', '150', '300', '600', '1200'])
    parser.add_argument('--log-dir', type=str, default='', metavar='PATH')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.log_dir = 'logs/{}/appear_only/'.format(args.L, ) + args.data_path + args.log_dir
    args.data_path = os.path.expanduser('~/Data/DukeMTMC/ground_truth/') + args.data_path
    if args.combine_trainval:
        train_data_path = args.data_path + 'hyperGT_{}_trainval_{}.h5'.format(args.L, args.window)
    else:
        train_data_path = args.data_path + 'hyperGT_{}_train_{}.h5'.format(args.L, args.window)
    if args.save_result:
        test_data_path = args.data_path + 'hyperGT_{}_train_Inf.h5'.format(args.L)
    else:
        test_data_path = args.data_path + 'hyperGT_{}_val_Inf.h5'.format(args.L)
    torch.manual_seed(args.seed)
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)

    trainset = SiameseHyperFeat(HyperFeat(train_data_path))
    testset = SiameseHyperFeat(HyperFeat(test_data_path), train=True)
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
