from utils.pie_data import PIE
from utils.pie_preprocessing import *

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.main_model import Model
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    if not args.learn: # 如果args.learn为False，则真实训练， 读取真实数据
        seed_all(args.seed)
        data_opts = {
            'fstride': 1,
            'sample_type': 'all',
            'height_rng': [0, float('inf')],
            'squarify_ratio': 0,
            'data_split_type': 'random',  # kfold, random, default
            'seq_type': 'crossing',  # crossing , intention
            'min_track_size': 15,  # discard tracks that are shorter
            'kfold_params': {'num_folds': 1, 'fold': 1},
            'random_params': {'ratios': [0.7, 0.15, 0.15],
                            'val_data': True,
                            'regen_data': False},
            'tte': [30, 60],
            'batch_size': 16
        }
        imdb = PIE(data_path=args.set_path) 
        seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts) # 生成训练集
        balanced_seq_train = balance_dataset(seq_train) # 平衡数据集
        tte_seq_train, traj_seq_train = tte_dataset(balanced_seq_train, data_opts['tte'], 0.6, args.times_num) # 生成训练集的tte和轨迹

        seq_valid = imdb.generate_data_trajectory_sequence('val', **data_opts)
        balanced_seq_valid = balance_dataset(seq_valid)
        tte_seq_valid, traj_seq_valid = tte_dataset(balanced_seq_valid, data_opts['tte'], 0, args.times_num)

        seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
        tte_seq_test, traj_seq_test = tte_dataset(seq_test, data_opts['tte'], 0, args.times_num)

        bbox_train = tte_seq_train['bbox'] # 训练集的bbox
        bbox_valid = tte_seq_valid['bbox']
        bbox_test = tte_seq_test['bbox']

        bbox_dec_train = traj_seq_train['bbox'] # 训练集的轨迹
        bbox_dec_valid = traj_seq_valid['bbox']
        bbox_dec_test  = traj_seq_test['bbox']

        obd_train = tte_seq_train['obd_speed'] # 训练集的速度
        obd_valid = tte_seq_valid['obd_speed']
        obd_test = tte_seq_test['obd_speed']

        gps_train = tte_seq_train['gps_speed'] # 训练集的速度
        gps_valid = tte_seq_valid['gps_speed']
        gps_test = tte_seq_test['gps_speed']

        action_train = tte_seq_train['activities'] # 训练集的动作
        action_valid = tte_seq_valid['activities']
        action_test = tte_seq_test['activities']

        normalized_bbox_train = normalize_bbox(bbox_train) # 归一化bbox
        normalized_bbox_valid = normalize_bbox(bbox_valid)
        normalized_bbox_test = normalize_bbox(bbox_test)

        normalized_bbox_dec_train = normalize_traj(bbox_dec_train) # 归一化轨迹
        normalized_bbox_dec_valid = normalize_traj(bbox_dec_valid)
        normalized_bbox_dec_test  = normalize_traj(bbox_dec_test)

        label_action_train = prepare_label(action_train) # 准备标签
        label_action_valid = prepare_label(action_valid)
        label_action_test = prepare_label(action_test)

        X_train, X_valid = torch.Tensor(normalized_bbox_train), torch.Tensor(normalized_bbox_valid) # 转换为tensor
        Y_train, Y_valid = torch.Tensor(label_action_train), torch.Tensor(label_action_valid)
        X_test = torch.Tensor(normalized_bbox_test)
        Y_test = torch.Tensor(label_action_test)


        temp = pad_sequence(normalized_bbox_dec_train, 60) 
        X_train_dec = torch.Tensor(temp)
        X_valid_dec = torch.Tensor(pad_sequence(normalized_bbox_dec_valid, 60)) # 转换为tensor
        X_test_dec = torch.Tensor(pad_sequence(normalized_bbox_dec_test, 60))

        obd_train, gps_train = torch.Tensor(obd_train), torch.Tensor(gps_train) # 转换为tensor
        obd_valid, gps_valid = torch.Tensor(obd_valid), torch.Tensor(gps_valid)
        obd_test, gps_test = torch.Tensor(obd_test), torch.Tensor(gps_test)

        vel_train = torch.cat([obd_train, gps_train], dim=-1) # 拼接obd和gps
        vel_valid = torch.cat([obd_valid, gps_valid], dim=-1)
        vel_test = torch.cat([obd_test, gps_test], dim=-1)

        trainset = TensorDataset(X_train, Y_train, vel_train, X_train_dec) # 生成dataset
        validset = TensorDataset(X_valid, Y_valid, vel_valid, X_valid_dec)
        testset = TensorDataset(X_test, Y_test, vel_test, X_test_dec)

        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True) # 生成dataloader
        valid_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(testset, batch_size=1)
    else: # args.learn为True，不真实训练，生成随机数据。
        train_loader = [[torch.randn(size=(args.batch_size, args.times_num, args.bbox_input)), # bbox
                         torch.randn(size=(args.batch_size, 1)),                                # label
                         torch.randn(size=(args.batch_size, args.times_num, args.vel_input)),   # velocity
                         torch.randn(size=(args.batch_size, args.times_num, args.bbox_input))]] # trajectory
        valid_loader = [[torch.randn(size=(args.batch_size, args.times_num, args.bbox_input)), 
                         torch.randn(size=(args.batch_size, 1)), 
                         torch.randn(size=(args.batch_size, args.times_num, args.vel_input)), 
                         torch.randn(size=(args.batch_size, args.times_num, args.bbox_input))]]
        test_loader = [[torch.randn(size=(args.batch_size, args.times_num, args.bbox_input)), 
                        torch.randn(size=(args.batch_size, 1)), 
                        torch.randn(size=(args.batch_size, args.times_num, args.vel_input)), 
                        torch.randn(size=(args.batch_size, args.times_num, args.bbox_input))]]
    print('Start Training Loop... \n')

    model = Model(args) # 生成模型
    model.to(device) # 放到gpu上

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-6) # 生成优化器
    cls_criterion = nn.BCELoss() # 生成损失函数 binary cross entropy
    reg_criterion = nn.MSELoss() # 生成损失函数

    model_folder_name = args.set_path 
    checkpoint_filepath = 'checkpoints/{}.pt'.format(model_folder_name) # 生成checkpoint的路径
    writer = SummaryWriter('logs/{}'.format(model_folder_name)) # 生成tensorboard的路径
    #Train
    train(model, train_loader, valid_loader, cls_criterion, reg_criterion, optimizer, checkpoint_filepath, writer, args=args)

    # #Test
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
    contrix = confusion_matrix(label_cpu, np.round(pred_cpu))

    print(f'Acc: {acc}\n f1: {f1}\n precision_score: {pre_s}\n recall_score: {recall_s}\n roc_auc_score: {auc}\n confusion_matrix: {contrix}')


if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser('Pedestrain Crossing Intention Prediction.')
    
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('--set_path', type=str, default='PIE')
    parser.add_argument('--balance', type=bool, default=True, help='balance or not for test dataset.')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--d_model', type=int, default=128, help='the dimension after embedding.')
    parser.add_argument('--dff', type=int, default=256, help='the number of the units.')
    parser.add_argument('--num_heads', type=int, default=8, help='number of the heads of the multi-head model.')
    parser.add_argument('--bbox_input', type=int, default=4, help='dimension of bbox.')
    parser.add_argument('--vel_input', type=int, default=2, help='dimension of velocity.')
    parser.add_argument('--time_crop', type=bool, default=False)# 是否使用随机时间裁剪

    parser.add_argument('--batch_size', type=int, default=64, help='size of batch.')
    parser.add_argument('--lr', type=int, default=0.0005, help='learning rate to train.')
    
    parser.add_argument('--num_layers', type=int, default=4, help='the number of layers.')
    parser.add_argument('--times_num', type=int, default=16, help='')# 数据的时间维度
    parser.add_argument('--num_bnks', type=int, default=3, help='')# 瓶颈结构的单元数目
    parser.add_argument('--bnks_layers', type=int, default=7, help='')# 瓶颈结构的层数

    parser.add_argument('--sta_f', type=int, default=8)# 若采用随机时间裁剪，则从sta_f到end_f中随机选取一个时间点作为保留的时间段。
    parser.add_argument('--end_f', type=int, default=12)

    parser.add_argument('--learn', type=bool, default=True)# 是否跳过真实数据读取，生成尺寸相同的随机数据。
    # 目的如果是为了了解项目的运行过程，则可以将learn设置为True，这样可以跳过真实数据读取，生成尺寸相同的随机数据。
    args = parser.parse_args()
    main(args)
