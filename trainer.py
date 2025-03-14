from base.trainer import GenericVideoTrainer
from base.scheduler import GradualWarmupScheduler, MyWarmupScheduler

from torch import optim
import torch

import time
import copy
import os

import numpy as np


class Trainer(GenericVideoTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.best_epoch_info = {
            'model_weights': copy.deepcopy(self.model.state_dict()),
            'loss': 1e10,
            'ccc': -1e10,
            'acc': -1,
            'train_f1_score': 0,
            'f1_score': 0,
            'kappa': 0,
            'epoch': 0,
            'metrics': {
                'train_loss': -1,
                'val_loss': -1,
                'train_f1': -1,
                'val_f1': -1,
            }
        }

    def init_optimizer_and_scheduler(self, epoch=0):
        params_pretrain, params_train = self.get_parameters2()
        scale = 0.5
        # 更新优化器定义
        self.optimizer = optim.AdamW([
            {'params': params_pretrain, 'lr': self.learning_rate*scale,'initial_lr':self.learning_rate*scale}, 
            {'params': params_train, 'lr': self.learning_rate, 'initial_lr': self.learning_rate}  
        ], lr=self.learning_rate, weight_decay = 0.00001)
        
        # self.optimizer = optim.Adam(self.get_parameters(), lr=self.learning_rate, weight_decay=0.001)

        self.scheduler = MyWarmupScheduler(
            optimizer=self.optimizer, lr = self.learning_rate, min_lr=self.min_learning_rate,
            best=self.best_epoch_info['f1_score'], mode="max", patience=self.patience,
            factor=self.factor, num_warmup_epoch=self.min_epoch, init_epoch=epoch)
        
        

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
                'f1_score': -1e10
            }

        for epoch in np.arange(start_epoch, self.max_epoch):

            if self.fit_finished:
                if self.verbose:
                    print("\nEarly Stop!\n")
                break

            improvement = False

            if epoch in self.milestone:# or (parameter_controller.get_current_lr() < self.min_learning_rate and epoch >= self.min_epoch and self.scheduler.relative_epoch > self.min_epoch):
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
            print(train_loss,train_record_dict)

            validate_kwargs = {"dataloader_dict": dataloader_dict, "epoch": epoch}
            validate_loss, validate_record_dict = self.validate(**validate_kwargs)

            # if epoch % 1 == 0:
            #     test_kwargs = {"dataloader_dict": dataloader_dict, "epoch": None, "train_mode": 0}
            #     validate_loss, test_record_dict = self.test(checkpoint_controller=checkpoint_controller, feature_extraction=0, **test_kwargs)
            #     print(test_record_dict['overall']['ccc'])

            if validate_loss < 0:
                raise ValueError('validate loss negative')

            self.train_losses.append(train_loss)
            self.validate_losses.append(validate_loss)
            
            train_f1_score = train_record_dict['f1_score']
            validate_f1_score = validate_record_dict['f1_score']
            # validate_ccc = validate_record_dict['overall']['ccc']

            self.scheduler.best = self.best_epoch_info['f1_score']

            if validate_f1_score > self.best_epoch_info['f1_score']:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "model_state_dict" + str(validate_f1_score) + ".pth"))

                improvement = True
                self.best_epoch_info = {
                    'model_weights': copy.deepcopy(self.model.state_dict()),
                    'loss': validate_loss,
                    'train_f1_score': train_f1_score,
                    'f1_score': validate_f1_score,
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

                print(train_record_dict['f1_score'])
                print(validate_record_dict['f1_score'])
                print("------")

            checkpoint_controller.save_log_to_csv(
                epoch, train_record_dict, validate_record_dict)

            # Early stopping controller.
            if self.early_stopping and self.scheduler.relative_epoch > self.min_epoch:
                if improvement:
                    self.early_stopping_counter = self.early_stopping
                else:
                    self.early_stopping_counter -= 1

                if self.early_stopping_counter <= 0:
                    self.fit_finished = True

            self.scheduler.step(metrics=validate_f1_score, epoch=epoch)


            self.start_epoch = epoch + 1

            if self.load_best_at_each_epoch:
                self.model.load_state_dict(self.best_epoch_info['model_weights'])

            checkpoint_controller.save_checkpoint(self, parameter_controller, self.save_path)

        self.fit_finished = True
        checkpoint_controller.save_checkpoint(self, parameter_controller, self.save_path)

        self.model.load_state_dict(self.best_epoch_info['model_weights'])
