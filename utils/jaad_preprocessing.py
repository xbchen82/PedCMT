import torch
import os
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_all(seed):
    torch.cuda.empty_cache()
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def binary_acc(label, pred):
    label_tag = torch.round(label)
    correct_results_sum = (label_tag == pred).sum().float()
    acc = correct_results_sum / pred.shape[0]
    return acc

def end_point_loss(reg_criterion, pred, end_point):
    for i in range(4):
        if i == 0 or i == 2:
            pred[:, i] = pred[:, i] * 1920
            end_point[:, i] = end_point[:, i] * 1920
        else:
            pred[:, i] = pred[:, i] * 1080
            end_point[:, i] = end_point[:, i] * 1080
    return reg_criterion(pred, end_point)


def train(model, train_loader, valid_loader, class_criterion, reg_criterion, optimizer, checkpoint_filepath, writer,
          args):
    best_valid_acc = 0.0
    improvement_ratio = 0.001
    best_valid_loss = np.inf
    num_steps_wo_improvement = 0
    save_times = 0
    epochs = args.epochs
    if args.learn: # 调试模式： epoch = 5
        epochs = 5
    time_crop = args.time_crop
    for epoch in range(epochs):
        nb_batches_train = len(train_loader)
        train_acc = 0
        model.train()
        f_losses = 0.0
        cls_losses = 0.0
        reg_losses = 0.0

        print('Epoch: {} training...'.format(epoch + 1))
        for bbox, label, vel, traj in train_loader:
            label = label.reshape(-1, 1).to(device).float()
            bbox = bbox.to(device)
            vel = vel.to(device)
            end_point = traj.to(device)[:, -1, :]

            if np.random.randint(10) >= 5 and time_crop:
                crop_size = np.random.randint(args.sta_f, args.end_f)
                bbox = bbox[:, -crop_size:, :]
                vel = vel[:, -crop_size:, :]

            pred, point, s_cls, s_reg = model(bbox, vel)

            cls_loss = class_criterion(pred, label)
            reg_loss = reg_criterion(point, end_point)
            f_loss = cls_loss / (s_cls * s_cls) + reg_loss / (s_reg * s_reg) + torch.log(s_cls) + torch.log(s_reg)

            model.zero_grad()  #
            f_loss.backward()

            f_losses += f_loss.item()
            cls_losses += cls_loss.item()
            reg_losses += reg_loss.item()

            optimizer.step()  #

            train_acc += binary_acc(label, torch.round(pred))

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
                                                                         reg_criterion)

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

        if best_valid_loss > valid_cls_loss:
            best_valid_loss = valid_cls_loss
            num_steps_wo_improvement = 0
            save_times += 1
            print(str(save_times) + ' time(s) File saved.\n')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'Accuracy': train_acc / nb_batches_train,
                'LOSS': f_losses / nb_batches_train,
            }, checkpoint_filepath)
            print('Update improvement.\n')
        else:
            num_steps_wo_improvement += 1
            print(str(num_steps_wo_improvement) + '/300 times Not update.\n')

        if num_steps_wo_improvement == 300:
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

            val_cls_loss = class_criterion(pred, label)
            val_reg_loss = reg_criterion(point, end_point)
            f_loss = val_cls_loss / (s_cls * s_cls) + val_reg_loss / (s_reg * s_reg) + torch.log(s_cls) + torch.log(
                s_reg)

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

            pred, _, _, _ = model(bbox, vel)

            if step == 0:
                preds = pred
                labels = label
            else:
                preds = torch.cat((preds, pred), 0)
                labels = torch.cat((labels, label), 0)
            step += 1

    return preds, labels


def balance_dataset(dataset, flip=True):
    d = {'bbox': dataset['bbox'].copy(),
         'pid': dataset['pid'].copy(),
         'activities': dataset['activities'].copy(),
         'image': dataset['image'].copy(),
         'center': dataset['center'].copy(),
         'vehicle_act': dataset['vehicle_act'].copy(),
         'image_dimension': (1920, 1080)}
    gt_labels = [gt[0] for gt in d['activities']]
    num_pos_samples = np.count_nonzero(np.array(gt_labels))
    num_neg_samples = len(gt_labels) - num_pos_samples

    if num_neg_samples == num_pos_samples:
        print('Positive samples is equal to negative samples.')
    else:
        print('Unbalanced: \t Postive: {} \t Negative: {}'.format(num_pos_samples, num_neg_samples))
        if num_neg_samples > num_pos_samples:
            gt_augment = 1
        else:
            gt_augment = 0

        img_width = d['image_dimension'][0]
        num_samples = len(d['pid'])

        for i in range(num_samples):
            if d['activities'][i][0][0] == gt_augment:
                flipped = d['center'][i].copy()
                flipped = [[img_width - c[0], c[1]] for c in flipped]
                d['center'].append(flipped)

                flipped = d['bbox'][i].copy()
                flipped = [np.array([img_width - c[2], c[1], img_width - c[0], c[3]]) for c in flipped]
                d['bbox'].append(flipped)

                d['pid'].append(dataset['pid'][i].copy())

                d['activities'].append(d['activities'][i].copy())
                d['vehicle_act'].append(d['vehicle_act'][i].copy())

                flipped = d['image'][i].copy()
                flipped = [c.replace('.png', '_flip.png') for c in flipped]

                d['image'].append(flipped)

        gt_labels = [gt[0] for gt in d['activities']]
        num_pos_samples = np.count_nonzero(np.array(gt_labels))
        num_neg_samples = len(gt_labels) - num_pos_samples

        if num_neg_samples > num_pos_samples:
            rm_index = np.where(np.array(gt_labels) == 0)[0]
        else:
            rm_index = np.where(np.array(gt_labels) == 1)[0]

        dif_samples = abs(num_neg_samples - num_pos_samples)

        np.random.seed(42)
        np.random.shuffle(rm_index)
        rm_index = rm_index[0:dif_samples]

        for k in d:
            seq_data_k = d[k]
            d[k] = [seq_data_k[i] for i in range(0, len(seq_data_k)) if i not in rm_index]

        new_gt_labels = [gt[0] for gt in d['activities']]
        num_pos_samples = np.count_nonzero(np.array(new_gt_labels))
        print('Balanced: Postive: %d \t Negative: %d \n' % (num_pos_samples, len(d['activities']) - num_pos_samples))
        print('Total Number of samples: %d\n' % (len(d['activities'])))

    return d


def tte_dataset(dataset, time_to_event, overlap, obs_length):
    d_obs = {'bbox': dataset['bbox'].copy(),
             'pid': dataset['pid'].copy(),
             'activities': dataset['activities'].copy(),
             'image': dataset['image'].copy(),
             'vehicle_act': dataset['vehicle_act'].copy(),
             'center': dataset['center'].copy()
             }

    d_tte = {'bbox': dataset['bbox'].copy(),
             'pid': dataset['pid'].copy(),
             'activities': dataset['activities'].copy(),
             'image': dataset['image'].copy(),
             'vehicle_act': dataset['vehicle_act'].copy(),
             'center': dataset['center'].copy()}

    if isinstance(time_to_event, int):
        for k in d_obs.keys():
            for i in range(len(d_obs[k])):
                d_obs[k][i] = d_obs[k][i][- obs_length - time_to_event: -time_to_event]
                d_tte[k][i] = d_tte[k][i][- time_to_event:]
        d_obs['tte'] = [[time_to_event]] * len(dataset['bbox'])
        d_tte['tte'] = [[time_to_event]] * len(dataset['bbox'])

    else:
        olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
        olap_res = 1 if olap_res < 1 else olap_res

        for k in d_obs.keys():
            seqs = []
            seqs_tte = []
            for seq in d_obs[k]:
                start_idx = len(seq) - obs_length - time_to_event[1]
                end_idx = len(seq) - obs_length - time_to_event[0]
                seqs.extend([seq[i:i + obs_length] for i in range(start_idx, end_idx, olap_res)])
                seqs_tte.extend([seq[i + obs_length:] for i in range(start_idx, end_idx, olap_res)])
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
        time_to_event_0 = time_to_event[0]
    except:
        time_to_event_0 = time_to_event
    for seq_index, (seq_obs, seq_tte) in enumerate(zip(d_obs['bbox'], d_tte['bbox'])):
        if len(seq_obs) < 16 or len(seq_tte) < time_to_event_0:
            remove_index.append(seq_index)

    for k in d_obs.keys():
        for j in sorted(remove_index, reverse=True):
            del d_obs[k][j]
            del d_tte[k][j]

    return d_obs, d_tte


def normalize_bbox(dataset, width=1920, height=1080):
    normalized_set = []
    for sequence in dataset:
        if sequence == []:
            continue
        normalized_sequence = []
        for bbox in sequence:
            np_bbox = np.zeros(4)
            np_bbox[0] = bbox[0] / width
            np_bbox[2] = bbox[2] / width
            np_bbox[1] = bbox[1] / height
            np_bbox[3] = bbox[3] / height
            normalized_sequence.append(np_bbox)
        normalized_set.append(np.array(normalized_sequence))

    return normalized_set

def normalize_traj(dataset, width=1920, height=1080):
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


def prepare_label(dataset):
    labels = np.zeros(len(dataset), dtype='int64')
    for step, action in enumerate(dataset):
        if action == []:
            continue
        labels[step] = action[0][0]

    return labels

def pad_sequence(inp_list, max_len):
    padded_sequence = []
    for source in inp_list:
        target = np.array([source[0]] * max_len)
        source = source
        target[-source.shape[0]:, :] = source
        
        padded_sequence.append(target)
        
    return padded_sequence
