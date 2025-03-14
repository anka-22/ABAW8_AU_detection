from base.logger import ContinuousOutputHandler, ContinuousMetricsCalculator, PlotHandler
from base.scheduler import GradualWarmupScheduler
from torch.nn.utils import clip_grad_value_
from torch import nn
from base.utils import ensure_dir

import time
import copy
import os
from tqdm import tqdm


import pandas as pd

import numpy as np
import torch
from torch import optim
from base.loss_function import compute_AU_F1

def remove_overlap(videos, videos_index):
    for video_id, frames_list in videos.items():
        index_list = videos_index[video_id]
        

        for i in range(len(frames_list) - 1):
            current_start, current_end = index_list[i]
            next_start, next_end = index_list[i + 1]
            

            if next_start < current_end:
                overlap = current_end - next_start

                videos_index[video_id][i + 1] = (current_end, next_end)
                

                frames_list[i + 1] = frames_list[i + 1][overlap:]
    # return videos, videos_index
                
class GenericTrainer(object):
    def __init__(self, **kwargs):
        self.device = kwargs['device']
        self.model_name = kwargs['model_name']
        self.model = kwargs['models'].to(self.device)
        self.save_path = kwargs['save_path']
        self.fold = kwargs['fold']
        self.min_epoch = kwargs['min_epoch']
        self.max_epoch = kwargs['max_epoch']
        self.start_epoch = 0
        self.early_stopping = kwargs['early_stopping']
        self.early_stopping_counter = self.early_stopping
        self.scheduler = kwargs['scheduler']
        self.learning_rate = kwargs['learning_rate']
        self.min_learning_rate = kwargs['min_learning_rate']
        self.patience = kwargs['patience']
        self.criterion = kwargs['criterion']
        self.factor = kwargs['factor']
        self.verbose = kwargs['verbose']
        self.milestone = kwargs['milestone']
        self.load_best_at_each_epoch = kwargs['load_best_at_each_epoch']

        self.optimizer, self.scheduler = None, None

    def train(self, **kwargs):
        kwargs['train_mode'] = True
        self.model.train()
        loss, result_dict = self.loop(**kwargs)
        print(loss,result_dict)
        return loss, result_dict

    def validate(self, **kwargs):

        kwargs['train_mode'] = False
        with torch.no_grad():
            self.model.eval()
            loss, result_dict = self.loop(**kwargs)
        return loss, result_dict

    def test(self, checkpoint_controller, predict_only=0, **kwargs):
        kwargs['train_mode'] = False

        with torch.no_grad():
            self.model.eval()

            if predict_only:
                self.predict_loop(**kwargs)
            else:
                loss, result_dict = self.loop(**kwargs)
                checkpoint_controller.save_log_to_csv(
                    kwargs['epoch'], mean_train_record=None, mean_validate_record=None, test_record=result_dict['overall'])

                return loss, result_dict

    def fit(self, **kwargs):
        raise NotImplementedError

    def loop(self, **kwargs):
        raise NotImplementedError

    def predict_loop(self, **kwargs):
        raise NotImplementedError

    def get_parameters(self):
        r"""
        Get the parameters to update.
        :return:
        """
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)
                params_to_update.append(param)
        return params_to_update
    
    def get_parameters2(self):
        r"""
        Get the parameters to update.
        :return:
        """
        params_pretrain = []
        params_train = []
        # params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)
                if 'layer4' in name:
                    params_pretrain.append(param)
                else:
                    params_train.append(param)
        return params_pretrain, params_train


class GenericVideoTrainer(GenericTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = kwargs['batch_size']
        self.emotion = kwargs['emotion']
        self.metrics = kwargs['metrics']
        self.save_plot = kwargs['save_plot']

        # For checkpoint
        self.fit_finished = False
        self.fold_finished = False
        self.resume = False
        self.time_fit_start = None

        self.train_losses = []
        self.validate_losses = []
        self.csv_filename = None
        self.best_epoch_info = None


    def fit(self, dataloader_dict, checkpoint_controller, parameter_controller):

        if self.verbose:
            print("------")
            print("Starting training, on device:", self.device)

        self.time_fit_start = time.time()
        start_epoch = self.start_epoch

        if self.best_epoch_info is None:
            self.best_epoch_info = {
                'model_weights': copy.deepcopy(self.model.state_dict()),
                'loss': 1e10,
                'ccc': -1e10
            }

        for epoch in np.arange(start_epoch, self.max_epoch):

            if self.fit_finished:
                if self.verbose:
                    print("\nEarly Stop!\n")
                break

            improvement = False

            if epoch in self.milestone :#or parameter_controller.get_current_lr() < self.min_learning_rate:
                parameter_controller.release_param(self.model.spatial, epoch)
                if parameter_controller.early_stop:
                    break

                self.model.load_state_dict(self.best_epoch_info['model_weights'])

            time_epoch_start = time.time()

            if self.verbose:
                print("There are {} layers to update.".format(len(self.optimizer.param_groups[0]['params'])))

            # Get the losses and the record dictionaries for training and validation.
            train_kwargs = {"dataloader_dict": dataloader_dict, "epoch": epoch}
            train_loss, train_record_dict = self.train(**train_kwargs)

            validate_kwargs = {"dataloader_dict": dataloader_dict, "epoch": epoch}
            validate_loss, validate_record_dict = self.validate(**validate_kwargs)

            # if epoch % 2 == 0:
            #     test_kwargs = {"dataloader_dict": dataloader_dict, "epoch": None, "train_mode": 0}
            #     validate_loss, test_record_dict = self.test(checkpoint_controller=checkpoint_controller, feature_extraction=0, **test_kwargs)
            #     print(test_record_dict['overall']['ccc'])

            if validate_loss < 0:
                raise ValueError('validate loss negative')

            self.train_losses.append(train_loss)
            self.validate_losses.append(validate_loss)

            validate_ccc = validate_record_dict['overall']['ccc']

            if validate_ccc > self.best_epoch_info['ccc']:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "model_state_dict.pth"))

                improvement = True
                self.best_epoch_info = {
                    'model_weights': copy.deepcopy(self.model.state_dict()),
                    'loss': validate_loss,
                    'ccc': validate_ccc,
                    'epoch': epoch,
                }

            if self.verbose:
                print(
                    "\n Fold {:2} Epoch {:2} in {:.0f}s || Train loss={:.3f} | Val loss={:.3f} | LR={:.1e} | Release_count={} | best={} | "
                    "improvement={}-{}".format(
                        self.fold,
                        epoch + 1,
                        time.time() - time_epoch_start,
                        train_loss,
                        validate_loss,
                        self.optimizer.param_groups[0]['lr'],
                        parameter_controller.release_count,
                        int(self.best_epoch_info['epoch']) + 1,
                        improvement,
                        self.early_stopping_counter))

                print(train_record_dict['overall'])
                print(validate_record_dict['overall'])
                print("------")

            checkpoint_controller.save_log_to_csv(
                epoch, train_record_dict['overall'], validate_record_dict['overall'])

            # Early stopping controller.
            if self.early_stopping and epoch > self.min_epoch:
                if improvement:
                    self.early_stopping_counter = self.early_stopping
                else:
                    self.early_stopping_counter -= 1

                if self.early_stopping_counter <= 0:
                    self.fit_finished = True

            self.scheduler.step(metrics=validate_loss, epoch=epoch)
            self.start_epoch = epoch + 1

            if self.load_best_at_each_epoch:
                self.model.load_state_dict(self.best_epoch_info['model_weights'])

            checkpoint_controller.save_checkpoint(self, parameter_controller, self.save_path)

        self.fit_finished = True
        checkpoint_controller.save_checkpoint(self, parameter_controller, self.save_path)

        self.model.load_state_dict(self.best_epoch_info['model_weights'])

    def loop(self, **kwargs):
        dataloader_dict, epoch, train_mode = kwargs['dataloader_dict'], kwargs['epoch'], kwargs['train_mode']

        if train_mode:
            dataloader = dataloader_dict['train']
        elif epoch is None:
            dataloader = dataloader_dict['extra']
        else:
            dataloader = dataloader_dict['validate']

        running_loss = 0.0
        total_batch_counter = 0
        inputs = {}

        num_batch_warm_up = 1500# 4000 #len(dataloader) * self.min_epoch
        all_preds = []
        all_labels = []
        
        for batch_idx, (X, trials, lengths, indices) in tqdm(enumerate(dataloader), total=len(dataloader)):
            # if batch_idx < 157:
            #     continue
            # for key, value in X.items():
            #     print(f"{key}: {value.shape}")
            if train_mode and (epoch ==0):
                self.scheduler.warmup_lr(self.learning_rate, batch_idx, num_batch_warm_up, epoch)

            total_batch_counter += len(trials)

            for feature, value in X.items():
                inputs[feature] = X[feature].to(self.device)

            if "continuous_label" in inputs:
                labels = inputs.pop("continuous_label", None)
            elif "AU_continuous_label" in inputs:
                labels = inputs.pop("AU_continuous_label", None)
                if labels.min() == -1:
                    with open('/home/data/zhangzr22/abaw/ABAW6/jilu.log', 'a') as f:  # 'a' 表示追加模式
                        f.write('label=-1\n')

            if len(torch.flatten(labels)) == self.batch_size:
                labels = torch.zeros((self.batch_size, len(indices[0]), 1), dtype=torch.float32).to(self.device)

            if train_mode:
                self.optimizer.zero_grad()

            if torch.isnan(inputs['video']).any():
                with open('/home/data/zhangzr22/abaw/ABAW6/jilu.log', 'a') as f:  # 'a' 表示追加模式
                    f.write('inputvideonan\n')

            outputs = self.model(inputs)
            if torch.isnan(outputs).any():
                with open('/home/data/zhangzr22/abaw/ABAW6/jilu.log', 'a') as f:  # 'a' 表示追加模式
                    f.write('outnan\n')

            loss, AU_pred, af_label = self.criterion(outputs, labels)
            print(loss)
            
            if torch.isnan(loss).any():
                with open('/home/data/zhangzr22/abaw/ABAW6/jilu.log', 'a') as f:  # 'a' 表示追加模式
                    f.write('lossnan\n')
            
            all_preds.append(AU_pred)
            all_labels.append(af_label)
            
            running_loss += loss.mean().item()
            clip_value = 0.5
            if train_mode:
                loss.backward()
                clip_grad_value_(self.model.parameters(), clip_value)  # 梯度裁剪
                self.optimizer.step()
                print(self.optimizer.param_groups[0]['lr'])
                
        combined_preds = torch.cat(all_preds, dim=0)
        combined_labels = torch.cat(all_labels, dim=0)
        
        _,f1_score = compute_AU_F1(combined_preds, combined_labels)
        
        if train_mode == 0:
            combined_preds_np = combined_preds.cpu().numpy()
            combined_labels_np = combined_labels.cpu().numpy()
            # 保存NumPy数组到文件
            np.save(f"/home/data/zhangzr22/abaw/ABAW6/pred_save_big/combined_preds{f1_score}.npy", combined_preds_np)
            np.save(f"/home/data/zhangzr22/abaw/ABAW6/pred_save_big/combined_labels{f1_score}.npy", combined_labels_np)
            
        epoch_loss = running_loss / total_batch_counter
        epoch_result_dict = {}
        epoch_result_dict['loss'] = epoch_loss
        epoch_result_dict['f1_score'] = f1_score
        
        return epoch_loss, epoch_result_dict

    def predict_loop(self, **kwargs):
        weights_path = '/home/data/zhangzr22/abaw/ABAW6/save/ABAW6_CAN_zzr_319_one_big_fold4_fold1_40.0001_window200_hop200_seed0.2_factor3407/model_state_dict0.5280523444457036.pth'
        self.model.load_state_dict(torch.load(weights_path))

        partition = kwargs['partition']
        dataloader = kwargs['dataloader_dict'][partition]
        inputs = {}
        records = {}
        all_preds = []
        all_labels = []
        videos = {}
        videos_index = {}
        videos_length = {}
        all_videos = {}
        # output_handler = ContinuousOutputHandler()
        for batch_idx, (X, trials, lengths, indices) in tqdm(enumerate(dataloader), total=len(dataloader)):

            if "AU_continuous_label" in inputs:
                labels = inputs.pop("AU_continuous_label", None)
                
            for feature, value in X.items():
                # if "label" in feature:
                #     continue
                inputs[feature] = X[feature].to(self.device)
                
            if "continuous_label" in inputs:
                labels = inputs.pop("continuous_label", None)

            outputs = self.model(inputs)

            bz,seq,_  = outputs.shape
            # pred = outputs.view(bz*seq,-1)
            
            # AU_pred = nn.Sigmoid()(pred)
        
            # all_preds.append(AU_pred)
            
            for i in range(bz):
                if trials[i] not in videos.keys():
                    videos[trials[i]] = []
                    videos_index[trials[i]] = []
                    videos_length[trials[i]] = lengths[i]
                videos[trials[i]].append(nn.Sigmoid()(outputs[i,:,:]))
                videos_index[trials[i]].append(indices[i])  
                # videos_length[trials[i]].append(lengths[i])  
            # records['trials'] = outputs
        #     output_handler.update_output_for_seen_trials(outputs.detach().cpu().numpy(), trials, indices, lengths)

        # output_handler.average_trial_wise_records()
        # output_handler.concat_records()
        for v in videos.keys():
            max_length = videos_index[v][-1].max()+1
            if max_length>200:
                final_length = max_length % 200
                videos[v][-1] = videos[v][-1][-final_length:,:]
                videos_index[v][-1] = videos_index[v][-1][-final_length:]
            all_videos[v] = torch.cat(videos[v])

        base_dir = '/home/data/zhangzr22/abaw/result_save_one_f4'
        # 确保基础目录存在
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        # 遍历字典并保存每个张量为.npy文件
        for key, tensor in all_videos.items():
            # 将张量转换为NumPy数组
            np_array = tensor.cpu().detach().numpy()
            # 构造保存路径
            save_path = os.path.join(base_dir, f"{key}.npy")
            # 保存NumPy数组
            np.save(save_path, np_array)
    
        combined_preds = torch.cat(all_preds, dim=0)
        combined_labels = torch.cat(all_labels, dim=0)
        _,f1_score = compute_AU_F1(combined_preds, combined_labels)
        print(f1_score)
        for trial, result in records.items():

            txt_save_path = os.path.join(self.save_path, "predict", partition, self.emotion, trial + ".txt")
            ensure_dir(txt_save_path)
            df = pd.DataFrame(data=result, index=None, columns=[self.emotion])
            df.to_csv(txt_save_path, sep=",", index=None)
    

