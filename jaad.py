from utils.jaad_data import JAAD
from utils.jaad_preprocessing import *

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.main_model import Model
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    if not args.learn:
        seed_all(args.seed)
        data_opts = {'fstride': 1,
                    'sample_type': args.bh,  # 'beh'
                    'subset': 'default',
                    'height_rng': [0, float('inf')],
                    'squarify_ratio': 0,
                    'data_split_type': 'default',  # kfold, random, default
                    'seq_type': 'crossing',
                    'min_track_size': 15,
                    'random_params': {'ratios': None,
                                        'val_data': True,
                                        'regen_data': False},
                    'kfold_params': {'num_folds': 5, 'fold': 1},
        }
        tte = [30, 60]
        imdb = JAAD(data_path=args.set_path)
        seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
        balanced_seq_train = balance_dataset(seq_train)
        tte_seq_train, traj_seq_train = tte_dataset(balanced_seq_train, tte, 0.6, args.times_num)

        seq_valid = imdb.generate_data_trajectory_sequence('val', **data_opts)
        balanced_seq_valid = balance_dataset(seq_valid)
        tte_seq_valid, traj_seq_valid = tte_dataset(balanced_seq_valid, tte, 0, args.times_num)

        seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
        tte_seq_test, traj_seq_test = tte_dataset(seq_test, tte, 0, args.times_num)

        bbox_train = tte_seq_train['bbox']
        bbox_valid = tte_seq_valid['bbox']
        bbox_test = tte_seq_test['bbox']

        bbox_dec_train = traj_seq_train['bbox']
        bbox_dec_valid = traj_seq_valid['bbox']
        bbox_dec_test  = traj_seq_test['bbox']

        vel_train = tte_seq_train['vehicle_act']
        vel_valid = tte_seq_valid['vehicle_act']
        vel_test = tte_seq_test['vehicle_act']

        action_train = tte_seq_train['activities']
        action_valid = tte_seq_valid['activities']
        action_test = tte_seq_test['activities']

        normalized_bbox_train = normalize_bbox(bbox_train)
        normalized_bbox_valid = normalize_bbox(bbox_valid)
        normalized_bbox_test = normalize_bbox(bbox_test)

        normalized_bbox_dec_train = normalize_traj(bbox_dec_train)
        normalized_bbox_dec_valid = normalize_traj(bbox_dec_valid)
        normalized_bbox_dec_test  = normalize_traj(bbox_dec_test)

        label_action_train = prepare_label(action_train)
        label_action_valid = prepare_label(action_valid)
        label_action_test = prepare_label(action_test)

        X_train, X_valid = torch.Tensor(normalized_bbox_train), torch.Tensor(normalized_bbox_valid)
        Y_train, Y_valid = torch.Tensor(label_action_train), torch.Tensor(label_action_valid)
        X_test = torch.Tensor(normalized_bbox_test)
        Y_test = torch.Tensor(label_action_test)

        X_train_dec = torch.Tensor(pad_sequence(normalized_bbox_dec_train, 60))
        X_valid_dec = torch.Tensor(pad_sequence(normalized_bbox_dec_valid, 60))
        X_test_dec = torch.Tensor(pad_sequence(normalized_bbox_dec_test, 60))

        vel_train = torch.Tensor(vel_train)
        vel_valid = torch.Tensor(vel_valid)
        vel_test = torch.Tensor(vel_test)

        trainset = TensorDataset(X_train, Y_train, vel_train, X_train_dec)
        validset = TensorDataset(X_valid, Y_valid, vel_valid, X_valid_dec)
        testset = TensorDataset(X_test, Y_test, vel_test, X_test_dec)

        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(testset, batch_size=1)
    else: # 生成随机数据
        train_loader = [[torch.randn(size=(args.batch_size, args.times_num, args.bbox_input)), 
                         torch.randn(size=(args.batch_size, 1)), 
                         torch.randn(size=(args.batch_size, args.times_num, args.vel_input)), 
                         torch.randn(size=(args.batch_size, args.times_num, args.bbox_input))]]
        valid_loader = [[torch.randn(size=(args.batch_size, args.times_num, args.bbox_input)), 
                         torch.randn(size=(args.batch_size, 1)), 
                         torch.randn(size=(args.batch_size, args.times_num, args.vel_input)), 
                         torch.randn(size=(args.batch_size, args.times_num, args.bbox_input))]]
        test_loader = [[torch.randn(size=(args.batch_size, args.times_num, args.bbox_input)), 
                        torch.randn(size=(args.batch_size, 1)), 
                        torch.randn(size=(args.batch_size, args.times_num, args.vel_input)), 
                        torch.randn(size=(args.batch_size, args.times_num, args.bbox_input))]]
    print('Start Training Loop... \n')

    model = Model(args)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
    cls_criterion = nn.BCELoss()
    reg_criterion = nn.MSELoss()

    model_folder_name = args.set_path + '_' + args.bh
    checkpoint_filepath = 'checkpoints/{}.pt'.format(model_folder_name)
    writer = SummaryWriter('logs/{}'.format(model_folder_name))

    train(model, train_loader, valid_loader, cls_criterion, reg_criterion, optimizer, checkpoint_filepath, writer, args=args)

    #Test
    model = Model(args)
    model.to(device)

    checkpoint = torch.load(checkpoint_filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    preds, labels = test(model, test_loader)
    pred_cpu = torch.Tensor.cpu(preds)
    label_cpu = torch.Tensor.cpu(labels)

    acc = accuracy_score(label_cpu, np.round(pred_cpu))
    f1 = f1_score(label_cpu, np.round(pred_cpu))
    pre_s = precision_score(label_cpu, np.round(pred_cpu))
    recall_s = recall_score(label_cpu, np.round(pred_cpu))
    auc = roc_auc_score(label_cpu, np.round(pred_cpu))
    matrix = confusion_matrix(label_cpu, np.round(pred_cpu))

    print(f'Acc: {acc}\n f1: {f1}\n precision_score: {pre_s}\n recall_score: {recall_s}\n roc_auc_score: {auc}\n confusion_matrix: {matrix}')


if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser('Pedestrain Crossing Intention Prediction.')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('--set_path', type=str, default='JAAD')
    parser.add_argument('--bh', type=str, default='beh', help='all or beh, in JAAD dataset.')
    parser.add_argument('--balance', type=bool, default=True, help='balance or not for test dataset.')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--d_model', type=int, default=256, help='the dimension after embedding.')
    parser.add_argument('--dff', type=int, default=512, help='the number of the units.')
    parser.add_argument('--num_heads', type=int, default=8, help='number of the heads of the multi-head model.')
    parser.add_argument('--bbox_input', type=int, default=4, help='dimension of bbox.')
    parser.add_argument('--vel_input', type=int, default=1, help='dimension of velocity.')
    parser.add_argument('--time_crop', type=bool, default=False)

    parser.add_argument('--batch_size', type=int, default=64, help='size of batch.')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate to train.')

    parser.add_argument('--num_layers', type=int, default=4, help='the number of layers.')
    parser.add_argument('--times_num', type=int, default=32, help='')
    parser.add_argument('--num_bnks', type=int, default=9, help='')
    parser.add_argument('--bnks_layers', type=int, default=9, help='')
    parser.add_argument('--sta_f', type=int, default=8)
    parser.add_argument('--end_f', type=int, default=12)
    args = parser.parse_args()
    main(args)