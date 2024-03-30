import torch
import os
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_all(seed): # 初始化
    torch.cuda.empty_cache()
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def binary_acc(label, pred): # 计算准确率
    label_tag = torch.round(label)# 四舍五入
    correct_results_sum = (label_tag == pred).sum().float()# 计算正确的个数
    acc = correct_results_sum / pred.shape[0] # 计算准确率
    return acc


def end_point_loss(reg_criterion, pred, end_point):# 计算端点误差（未使用）
    for i in range(4):
        if i == 0 or i == 2:
            pred[:, i] = pred[:, i] * 1920 # 1920是视频的宽
            end_point[:, i] = end_point[:, i] * 1920 
        else:
            pred[:, i] = pred[:, i] * 1080 # 1080是视频的高
            end_point[:, i] = end_point[:, i] * 1080
    return reg_criterion(pred, end_point) 



def train(model, train_loader, valid_loader, class_criterion, reg_criterion, optimizer, checkpoint_filepath, writer,
          args):
    # best_valid_acc = 0.0# 最佳准确率
    # improvement_ratio = 0.001
    best_valid_loss = np.inf # 最佳损失
    num_steps_wo_improvement = 0 # 未提升的次数
    save_times = 0 # 保存的次数
    epochs = args.epochs # 训练的轮数
    # time_crop = args.time_crop # 是否进行时间裁剪
    if args.learn: # 调试模式： epoch = 5
        epochs = 5
    for epoch in range(epochs):
        nb_batches_train = len(train_loader) # 训练集的batch数
        train_acc = 0 # 训练集的准确率
        model.train() # 训练模式
        f_losses = 0.0 # 总损失
        cls_losses = 0.0 # 分类损失
        reg_losses = 0.0 # 回归损失

        print('Epoch: {} training...'.format(epoch + 1))
        for bbox, label, vel, traj in train_loader:
            label = label.reshape(-1, 1).to(device).float() # 标签
            bbox = bbox.to(device) 
            vel = vel.to(device)
            end_point = traj.to(device)[:, -1, :] #轨迹的最后一刻时刻的点

            # if np.random.randint(10) >= 5 and time_crop: # 随机时间裁剪
            #     crop_size = np.random.randint(args.sta_f, args.end_f)
            #     bbox = bbox[:, -crop_size:, :]
            #     # vel = vel[:, -crop_size:, :]

            pred, point, s_cls, s_reg = model(bbox, vel) # 预测值，端点，分类损失系数，回归损失系数
            cls_loss = class_criterion(pred, label) # 分类损失
            reg_loss = reg_criterion(point, end_point) # 回归损失
            f_loss = cls_loss / (s_cls * s_cls) + reg_loss / (s_reg * s_reg) + torch.log(s_cls * s_reg) 
            # 总损失

            model.zero_grad()  # 梯度清零
            f_loss.backward() # 反向传播

            f_losses += f_loss.item() # 总损失记录
            cls_losses += cls_loss.item() # 分类损失记录
            reg_losses += reg_loss.item() # 回归损失记录

            optimizer.step()  # 更新参数

            train_acc += binary_acc(label, torch.round(pred)) # 计算准确率
        

        writer.add_scalar('training full_loss',
                          f_losses / nb_batches_train,
                          epoch + 1)
        writer.add_scalar('training cls_loss',
                          cls_losses / nb_batches_train,
                          epoch + 1)
        writer.add_scalar('training reg_loss',
                          reg_losses / nb_batches_train,
                          epoch + 1)
        writer.add_scalar('training Acc',
                          train_acc / nb_batches_train,
                          epoch + 1)
        

        print(
            f"Epoch {epoch + 1}: | Train_Loss {f_losses / nb_batches_train} | Train Cls_loss {cls_losses / nb_batches_train} | Train Reg_loss {reg_losses / nb_batches_train} | Train_Acc {train_acc / nb_batches_train} ")
        valid_f_loss, valid_cls_loss, valid_reg_loss, val_acc = evaluate(model, valid_loader, class_criterion,
                                                                         reg_criterion) # 验证

        writer.add_scalar('validation full_loss',
                          valid_f_loss,
                          epoch + 1)
        writer.add_scalar('validation cls_loss',
                          valid_cls_loss,
                          epoch + 1)
        writer.add_scalar('validation reg_loss',
                          valid_reg_loss,
                          epoch + 1)
        writer.add_scalar('validation Acc',
                          val_acc,
                          epoch + 1)

        if best_valid_loss > valid_cls_loss: # 保存最佳模型
            best_valid_loss = valid_cls_loss # 更新最佳损失
            num_steps_wo_improvement = 0 # 未提升的次数清零
            save_times += 1
            print(str(save_times) + ' time(s) File saved.\n')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'Accuracy': train_acc / nb_batches_train,
                'LOSS': f_losses / nb_batches_train,
            }, checkpoint_filepath) # 保存模型
            print('Update improvement.\n')

        else: # 未提升
            num_steps_wo_improvement += 1
            print(str(num_steps_wo_improvement) + '/300 times Not update.\n')

        if num_steps_wo_improvement == 300: # 300次未提升，提前结束
            print("Early stopping on epoch:{}".format(str(epoch + 1)))
            break
    print('save file times: ' + str(save_times) + '.\n')


def evaluate(model, val_data, class_criterion, reg_criterion):
    nb_batches = len(val_data)
    val_f_losses = 0.0
    val_cls_losses = 0.0
    val_reg_losses = 0.0
    print('in Validation...')
    with torch.no_grad():
        model.eval()
        acc = 0
        for bbox, label, vel, traj in val_data:
            label = label.reshape(-1, 1).to(device).float()
            bbox = bbox.to(device)
            vel = vel.to(device)
            end_point = traj.to(device)[:, -1, :]

            pred, point, s_cls, s_reg = model(bbox, vel)
            val_reg_loss = reg_criterion(point, end_point)
            val_cls_loss = class_criterion(pred, label)
            f_loss = val_cls_loss / (s_cls * s_cls) + val_reg_loss / (s_reg * s_reg) + torch.log(s_cls * s_reg)

            val_f_losses += f_loss.item()
            val_cls_losses += val_cls_loss.item()
            val_reg_losses += val_reg_loss.item()

            acc += binary_acc(label, torch.round(pred))
    print(
        f'Valid_Full_Loss {val_f_losses / nb_batches} | Valid Cls_loss {val_cls_losses / nb_batches} | Valid Reg_loss {val_reg_losses / nb_batches} | Valid_Acc {acc / nb_batches} \n')
    return val_f_losses / nb_batches, val_cls_losses / nb_batches, val_reg_losses / nb_batches, acc / nb_batches


def test(model, test_data):
    print('Tesing...')
    with torch.no_grad():
        model.eval()
        step = 0
        for bbox, label, vel, traj in test_data:
            label = label.reshape(-1, 1).to(device).float()
            bbox = bbox.to(device)
            vel = vel.to(device)

            pred, _, _, _ = model(bbox, vel)#测试阶段只需要预测分类结果，不关心回归结果

            if step == 0:
                preds = pred
                labels = label
            else:
                preds = torch.cat((preds, pred), 0)
                labels = torch.cat((labels, label), 0)
            step += 1
    return preds, labels


def balance_dataset(dataset, flip=True): # 数据集平衡
    d = {'bbox': dataset['bbox'].copy(),
         'pid': dataset['pid'].copy(),
         'activities': dataset['activities'].copy(),
         'image': dataset['image'].copy(),
         'center': dataset['center'].copy(),
         'obd_speed': dataset['obd_speed'].copy(),
         'gps_speed': dataset['gps_speed'].copy(),
         'image_dimension': (1920, 1080)}
    gt_labels = [gt[0] for gt in d['activities']] # 标签
    num_pos_samples = np.count_nonzero(np.array(gt_labels)) # 正样本数
    num_neg_samples = len(gt_labels) - num_pos_samples # 负样本数

    if num_neg_samples == num_pos_samples: # 正负样本数相等
        print('Positive samples is equal to negative samples.')
    else: # 正负样本数不相等
        print('Unbalanced: \t Postive: {} \t Negative: {}'.format(num_pos_samples, num_neg_samples))
        if num_neg_samples > num_pos_samples:
            gt_augment = 1 # 正样本数大于负样本数，增加负样本
        else:
            gt_augment = 0 # 负样本数大于正样本数，增加正样本

        img_width = d['image_dimension'][0] # 图片宽度
        num_samples = len(d['pid']) # 样本数

        for i in range(num_samples): # 遍历样本
            if d['activities'][i][0][0] == gt_augment: # 标签与增加的标签相同
                flipped = d['center'][i].copy() # 中心点
                flipped = [[img_width - c[0], c[1]] for c in flipped] # 水平翻转
                d['center'].append(flipped) # 添加到中心点

                flipped = d['bbox'][i].copy() # 边界框
                flipped = [np.array([img_width - c[2], c[1], img_width - c[0], c[3]]) for c in flipped] # 水平翻转
                d['bbox'].append(flipped) # 添加到边界框

                d['pid'].append(dataset['pid'][i].copy()) # 添加pid

                d['activities'].append(d['activities'][i].copy()) # 添加标签
                d['gps_speed'].append(d['gps_speed'][i].copy()) # 添加gps速度
                d['obd_speed'].append(d['obd_speed'][i].copy()) # 添加obd速度

                flipped = d['image'][i].copy() # 图片
                flipped = [c.replace('.png', '_flip.png') for c in flipped] # 水平翻转

                d['image'].append(flipped) # 添加图片

        gt_labels = [gt[0] for gt in d['activities']] # 标签
        num_pos_samples = np.count_nonzero(np.array(gt_labels)) # 正样本数
        num_neg_samples = len(gt_labels) - num_pos_samples # 负样本数

        if num_neg_samples > num_pos_samples: # 负样本数大于正样本数
            rm_index = np.where(np.array(gt_labels) == 0)[0] # 删除负样本
        else:
            rm_index = np.where(np.array(gt_labels) == 1)[0] # 删除正样本

        dif_samples = abs(num_neg_samples - num_pos_samples) # 正负样本数差值

        np.random.seed(42)
        np.random.shuffle(rm_index) # 打乱索引
        rm_index = rm_index[0:dif_samples] # 间隔删除

        for k in d: # 遍历数据
            seq_data_k = d[k] # 数据
            d[k] = [seq_data_k[i] for i in range(0, len(seq_data_k)) if i not in rm_index] # 删除数据

        new_gt_labels = [gt[0] for gt in d['activities']] # 新标签
        num_pos_samples = np.count_nonzero(np.array(new_gt_labels)) # 新正样本数
        print('Balanced: Postive: %d \t Negative: %d \n' % (num_pos_samples, len(d['activities']) - num_pos_samples))
        print('Total Number of samples: %d\n' % (len(d['activities'])))

    return d


def tte_dataset(dataset, time_to_event, overlap, obs_length): # 时间到事件数据集
    d_obs = {'bbox': dataset['bbox'].copy(), 
             'pid': dataset['pid'].copy(),
             'activities': dataset['activities'].copy(),
             'image': dataset['image'].copy(),
             'gps_speed': dataset['gps_speed'].copy(),
             'obd_speed': dataset['obd_speed'].copy(),
             'center': dataset['center'].copy()
             }

    d_tte = {'bbox': dataset['bbox'].copy(),
             'pid': dataset['pid'].copy(),
             'activities': dataset['activities'].copy(),
             'image': dataset['image'].copy(),
             'gps_speed': dataset['gps_speed'].copy(),
             'obd_speed': dataset['obd_speed'].copy(),
             'center': dataset['center'].copy()}

    if isinstance(time_to_event, int):
        for k in d_obs.keys():
            for i in range(len(d_obs[k])):
                d_obs[k][i] = d_obs[k][i][- obs_length - time_to_event: -time_to_event] # 观察长度
                d_tte[k][i] = d_tte[k][i][- time_to_event:] # 时间到事件
        d_obs['tte'] = [[time_to_event]] * len(dataset['bbox']) # 观察长度
        d_tte['tte'] = [[time_to_event]] * len(dataset['bbox']) # 时间到事件

    else: # 时间到事件为列表
        olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length) # 重叠长度
        olap_res = 1 if olap_res < 1 else olap_res # 重叠长度

        for k in d_obs.keys(): # 遍历数据
            seqs = []
            seqs_tte = []
            for seq in d_obs[k]:
                start_idx = len(seq) - obs_length - time_to_event[1] # 开始索引
                end_idx = len(seq) - obs_length - time_to_event[0] # 结束索引 
                seqs.extend([seq[i:i + obs_length] for i in range(start_idx, end_idx, olap_res)]) # 观察长度
                seqs_tte.extend([seq[i + obs_length:] for i in range(start_idx, end_idx, olap_res)]) # 时间到事件
                d_obs[k] = seqs
                d_tte[k] = seqs_tte
        tte_seq = []
        for seq in dataset['bbox']:
            start_idx = len(seq) - obs_length - time_to_event[1]
            end_idx = len(seq) - obs_length - time_to_event[0]
            tte_seq.extend([[len(seq) - (i + obs_length)] for i in range(start_idx, end_idx, olap_res)])
            d_obs['tte'] = tte_seq.copy()
            d_tte['tte'] = tte_seq.copy()

    remove_index = []
    try:
        time_to_event_0 = time_to_event[0] # 时间到事件
    except:
        time_to_event_0 = time_to_event # 时间到事件 
    for seq_index, (seq_obs, seq_tte) in enumerate(zip(d_obs['bbox'], d_tte['bbox'])): # 遍历数据
        if len(seq_obs) < 16 or len(seq_tte) < time_to_event_0: # 观察长度小于16或时间到事件小于时间到事件
            remove_index.append(seq_index) # 删除索引

    for k in d_obs.keys():
        for j in sorted(remove_index, reverse=True): # 倒序删除
            del d_obs[k][j]
            del d_tte[k][j]

    return d_obs, d_tte


def normalize_bbox(dataset, width=1920, height=1080): # 归一化边界框
    normalized_set = []
    for sequence in dataset:
        if sequence == []:
            continue
        normalized_sequence = []
        for bbox in sequence:
            np_bbox = np.zeros(4)
            np_bbox[0] = bbox[0] / width # 左上角x 
            np_bbox[2] = bbox[2] / width # 右下角x 
            np_bbox[1] = bbox[1] / height # 左上角y 
            np_bbox[3] = bbox[3] / height # 右下角y 
            normalized_sequence.append(np_bbox)
        normalized_set.append(np.array(normalized_sequence))

    return normalized_set

def normalize_traj(dataset, width=1920, height=1080): # 归一化轨迹
    normalized_set = []
    for sequence in dataset:
        if sequence == []:
            continue
        normalized_sequence = []
        for bbox in sequence:
            np_bbox = np.zeros(4)
            np_bbox[0] = bbox[0]# / width
            np_bbox[2] = bbox[2]# / width
            np_bbox[1] = bbox[1]# / height
            np_bbox[3] = bbox[3]# / height
            normalized_sequence.append(np_bbox)
        normalized_set.append(np.array(normalized_sequence))

    return normalized_set


def prepare_label(dataset): # 准备标签
    labels = np.zeros(len(dataset), dtype='int64')
    for step, action in enumerate(dataset):
        if action == []:
            continue
        labels[step] = action[0][0]

    return labels

def pad_sequence(inp_list, max_len): # 填充序列
    padded_sequence = []
    for source in inp_list:
        target = np.array([source[0]] * max_len) # 填充序列
        source = source 
        target[-source.shape[0]:, :] = source # 填充序列
        
        padded_sequence.append(target)
        
    return padded_sequence
