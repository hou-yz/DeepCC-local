from __future__ import print_function
import argparse
import os
import os.path as osp
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_new import *
from Utils import *


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Hyper Score')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-j', '--num-workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=40, metavar='N')
    parser.add_argument('--step-size', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR')
    # 40epoch, lr=1e-3; 150epoch, lr=2e-4
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, val set alone for validation")
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--save_result', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--data-path', type=str, default='1fps_train_IDE_40',
                        metavar='PATH')
    parser.add_argument('-L', type=str, default='L2', choices=['L2', 'L3'])
    parser.add_argument('--window', type=str, default='75',
                        choices=['Inf', '75', '150', '300', '600', '1200', '2400', '4800', '9600', '19200'])
    parser.add_argument('--log-dir', type=str, default='GT', metavar='PATH')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=300, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--features', type=int, default=256, choices=[256, 1024, 1536])
    parser.add_argument('--fft', action='store_true')
    parser.add_argument('--pcb', action='store_true')
    parser.add_argument('--triplet', action='store_true')
    parser.add_argument('--motion', action='store_true')
    args = parser.parse_args()
    if 'L3' in args.L:
        args.weight_decay = 1e-3
        pass

    if args.triplet:
        args.data_path = '1fps_train_IDE_triplet_40'
        train_data_path = '/home/houyz/Data/DukeMTMC/L0-features/gt_features_ide_triplet_basis_train_1fps/tracklet_features.h5'
        test_data_path = '/home/houyz/Data/DukeMTMC/L0-features/gt_features_ide_triplet_basis_train_1fps/tracklet_features.h5'
    elif args.pcb:
        args.data_path = '1fps_train_PCB_40'
        train_data_path = '/home/houyz/Data/DukeMTMC/L0-features/gt_features_pcb_basis_fc64_train_1fps/tracklet_features.h5'
        test_data_path = '/home/houyz/Data/DukeMTMC/L0-features/gt_features_pcb_basis_fc64_train_1fps/tracklet_features.h5'
    else:
        train_data_path = '/home/houyz/Data/DukeMTMC/L0-features/gt_features_ide_basis_train_1fps/tracklet_features.h5'
        test_data_path = '/home/houyz/Data/DukeMTMC/L0-features/gt_features_ide_basis_train_1fps/tracklet_features.h5'

    # dataset path
    # if args.combine_trainval:
    #     train_data_path = osp.join(args.data_path, 'hyperGT_{}_trainval_{}.h5'.format(args.L, args.window))
    # else:
    #     train_data_path = osp.join(args.data_path, 'hyperGT_{}_train_{}.h5'.format(args.L, args.window))
    # if args.save_result:
    #     test_data_path = osp.join(args.data_path, 'hyperGT_{}_train_Inf.h5'.format(args.L))
    # else:
    #     if not args.motion:
    #         test_data_path = osp.join(args.data_path, 'hyperGT_{}_val_Inf.h5'.format(args.L))
    #     else:
    #         test_data_path = osp.join(args.data_path, 'hyperGT_{}_val_{}.h5'.format(args.L, args.window))

    torch.manual_seed(args.seed)
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)

    trainset = SiameseHyperFeat(HyperFeat(train_data_path, trainval='trainval' if args.combine_trainval else 'train',
                                          motion_dim=3, L=args.L, window=args.window))
    testset = SiameseHyperFeat(HyperFeat(test_data_path,
                                         motion_dim=3, trainval='val', L=args.L, window='Inf'))

    train_loader = DataLoader(trainset, batch_size=args.batch_size,
                              num_workers=args.num_workers, pin_memory=True, shuffle=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size,
                             num_workers=args.num_workers, pin_memory=True)

    metric_net = nn.DataParallel(MetricNet(feature_dim=args.features if not args.motion else 8, num_class=2)).cuda()
    if args.resume:
        checkpoint = torch.load(args.log_dir + '/metric_net_{}_{}.pth.tar'.format(args.L, args.window))
        model_dict = checkpoint['state_dict']
        metric_net.module.load_state_dict(model_dict)

    criterion = nn.CrossEntropyLoss().cuda()

    if args.train:
        # Draw Curve
        x_epoch = []
        train_loss_s = []
        train_prec_s = []
        test_loss_s = []
        test_prec_s = []
        optimizer = optim.SGD(metric_net.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
        for epoch in range(1, args.epochs + 1):
            train_loss, train_prec = train(args, metric_net, train_loader, optimizer, epoch, criterion)
            test_loss, test_prec = test(args, metric_net, test_loader, criterion, False, epoch)
            x_epoch.append(epoch)
            train_loss_s.append(train_loss)
            train_prec_s.append(train_prec)
            test_loss_s.append(test_loss)
            test_prec_s.append(test_prec)
            path = args.log_dir + '/MetricNet_{}_{}.jpg'.format(args.L, args.window)
            draw_curve(path, x_epoch, train_loss_s, train_prec_s, test_loss_s, test_prec_s)
            pass
        torch.save({'state_dict': metric_net.module.state_dict(), }, args.log_dir + '/metric_net_{}_{}.pth.tar'.
                   format(args.L, args.window))

        path = osp.join(args.log_dir, 'model_param_{}_{}.mat'.format(args.L, args.window))
        save_model_as_mat(path, metric_net.module)

    checkpoint = torch.load(args.log_dir + '/metric_net_{}_{}.pth.tar'.format(args.L, args.window))
    model_dict = checkpoint['state_dict']
    metric_net.module.load_state_dict(model_dict)
    test(args, metric_net, test_loader, criterion, save_result=args.save_result)


if __name__ == '__main__':
    main()
