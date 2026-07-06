# trainer.py
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: trainer
import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import gc
import pickle
import datetime
import logging
import numpy as np
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from metrics.base_metrics_class import Recorder
from torch.optim.swa_utils import AveragedModel, SWALR
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn import metrics
from metrics.utils import get_test_metrics
from dataset.balanced_sampler import DemographicBalancedSampler
from torch.cuda.amp import autocast, GradScaler


FFpp_pool=['FaceForensics++','FF-DF','FF-F2F','FF-FS','FF-NT']#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    def __init__(
        self,
        config,
        model,
        optimizer,
        scheduler,
        logger,
        metric_scoring='auc',
        time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
        swa_model=None
        ):
        # check if all the necessary components are implemented
        if config is None or model is None or optimizer is None or logger is None:
            raise ValueError("config, model, optimizier, logger, and tensorboard writer must be implemented")

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.swa_model = swa_model
        self.writers = {}  # dict to maintain different tensorboard writers for each dataset and metric
        self.logger = logger
        self.metric_scoring = metric_scoring
        # maintain the best metric of all epochs
        self.best_metrics_all_time = defaultdict(
            lambda: defaultdict(lambda: float('-inf')
            if self.metric_scoring != 'eer' else float('inf'))
        )
        self.best_fairness_all_time = defaultdict(
            lambda: defaultdict(lambda: float('inf'))  # EO: makin kecil makin baik
        )
        self.speed_up()  # move model to GPU

        # get current time
        self.timenow = time_now
        # create directory path
        if 'task_target' not in config:
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + '_' + self.timenow
            )
        else:
            task_str = f"_{config['task_target']}" if config['task_target'] is not None else ""
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + task_str + '_' + self.timenow
            )
        os.makedirs(self.log_dir, exist_ok=True)
        self.scaler = GradScaler()

    def get_writer(self, phase, dataset_key, metric_key):
        writer_key = f"{phase}-{dataset_key}-{metric_key}"
        if writer_key not in self.writers:
            # update directory path
            writer_path = os.path.join(
                self.log_dir,
                phase,
                dataset_key,
                metric_key,
                "metric_board"
            )
            os.makedirs(writer_path, exist_ok=True)
            # update writers dictionary
            self.writers[writer_key] = SummaryWriter(writer_path)
        return self.writers[writer_key]


    def speed_up(self):
        self.model.to(device)
        self.model.device = device
        if self.config['ddp'] == True:
            num_gpus = torch.cuda.device_count()
            print(f'avai gpus: {num_gpus}')
            # local_rank=[i for i in range(0,num_gpus)]
            self.model = DDP(self.model, device_ids=[self.config['local_rank']],find_unused_parameters=True, output_device=self.config['local_rank'])
            #self.optimizer =  nn.DataParallel(self.optimizer, device_ids=[int(os.environ['LOCAL_RANK'])])

    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            suffix = model_path.split('.')[-1]
            if suffix == 'p':
                self.model.load_state_dict(saved.state_dict())
            else:
                self.model.load_state_dict(saved)
            self.logger.info('Model found in {}'.format(model_path))
        else:
            raise NotImplementedError(
                "=> no model found at '{}'".format(model_path))

    def save_resume_ckpt(self, epoch):
        """
        Simpan checkpoint lengkap untuk resume training.
        File: {log_dir}/resume_ckpt.pth  (selalu overwrite, hanya butuh 1 file)
        """
        parent_log_dir = self.config['log_dir']
        os.makedirs(parent_log_dir, exist_ok=True)
        save_path = os.path.join(parent_log_dir, 'resume_ckpt.pth')
 
        # Ambil model state — handle DDP wrapper
        if isinstance(self.model, DDP):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()

        best_metrics_serializable = {
            dataset_key: dict(metric_dict)
            for dataset_key, metric_dict in self.best_metrics_all_time.items()
        }

        best_fairness_serializable = {
            dataset_key: dict(metric_dict)
            for dataset_key, metric_dict in self.best_fairness_all_time.items()
        }
 
        checkpoint = {
            'epoch':                 epoch,
            'model_state_dict':      model_state,
            'optimizer_state_dict':  self.optimizer.state_dict(),
            'scaler_state_dict':     self.scaler.state_dict(),
            'best_metrics_all_time': best_metrics_serializable,
            'best_fairness_all_time': best_fairness_serializable,
            'log_dir':               self.log_dir,
        }
 
        # Simpan scheduler state jika ada
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
 
        # Simpan SWA state jika ada
        if self.swa_model is not None:
            checkpoint['swa_model_state_dict'] = self.swa_model.state_dict()
 
        torch.save(checkpoint, save_path)
        self.logger.info(f"[Resume ckpt] Tersimpan di {save_path} (epoch {epoch})")
 
    def load_resume_ckpt(self, resume_path=None):
        """
        Load checkpoint untuk melanjutkan training.
 
        Args:
            resume_path: path ke file .pth. Kalau None, otomatis cari
                         resume_ckpt.pth di log_dir.
 
        Returns:
            start_epoch (int): epoch berikutnya yang harus dijalankan.
                               Kembalikan 0 kalau file tidak ditemukan.
        """
        if resume_path is None:
            resume_path = os.path.join(self.config['log_dir'], 'resume_ckpt.pth')
 
        if not os.path.isfile(resume_path):
            self.logger.info(f"[Resume ckpt] Tidak ditemukan di {resume_path}, mulai dari epoch 0.")
            return 0
 
        self.logger.info(f"[Resume ckpt] Memuat dari {resume_path} ...")
        checkpoint = torch.load(resume_path, map_location='cpu')
 
        # 1. Load model weights
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
 
        # 2. Load optimizer state (momentum Adam tersimpan di sini)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
 
        # Pindahkan optimizer state ke GPU yang benar
        # (torch kadang load optimizer state ke CPU meskipun model di GPU)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    if k == 'step':
                        state[k] = v.cpu()  # pastikan step tetap di CPU
                    else:
                        state[k] = v.to(device)
 
        # 3. Load scheduler state
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.logger.info("[Resume ckpt] Scheduler state dimuat.")
 
        if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
            saved_scaler = checkpoint['scaler_state_dict']
            # Reset scale ke nilai kecil agar tidak overflow saat resume
            self.scaler = GradScaler(init_scale=256.0)
            self.logger.info("[Resume ckpt] GradScaler di-reset ke init_scale=256 untuk keamanan resume.")
            
        else:
            self.logger.info("[Resume ckpt] WARNING: AMP GradScaler state tidak ditemukan. Hati-hati gradient shock!")
            
        # 4. Load SWA state
        if self.swa_model is not None and 'swa_model_state_dict' in checkpoint:
            self.swa_model.load_state_dict(checkpoint['swa_model_state_dict'])
            self.logger.info("[Resume ckpt] SWA model state dimuat.")
 
        # 5. Restore best metrics agar checkpoint "best" tidak di-overwrite oleh epoch awal resume yang mungkin lebih jelek
        if 'best_metrics_all_time' in checkpoint:
            saved_best = checkpoint['best_metrics_all_time']
            for dataset_key, metric_dict in saved_best.items():
                for metric_name, value in metric_dict.items():
                    self.best_metrics_all_time[dataset_key][metric_name] = value
            self.logger.info("[Resume ckpt] best_metrics_all_time dipulihkan.")
 
        if 'best_fairness_all_time' in checkpoint:
            saved_fairness = checkpoint['best_fairness_all_time']
            for dataset_key, metric_dict in saved_fairness.items():
                for metric_name, value in metric_dict.items():
                    self.best_fairness_all_time[dataset_key][metric_name] = value
            self.logger.info("[Resume ckpt] best_fairness_all_time dipulihkan.")

        if 'log_dir' in checkpoint:
            self.log_dir = checkpoint['log_dir']
            self.logger.info(f"[Resume ckpt] log_dir dipulihkan ke {self.log_dir}")

        last_epoch = checkpoint['epoch']
        start_epoch = last_epoch + 1
        self.logger.info(f"[Resume ckpt] Epoch terakhir: {last_epoch}. Lanjut dari epoch {start_epoch}.")
        return start_epoch


    def save_ckpt(self, phase, dataset_key,ckpt_info=None):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"ckpt_best.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        if self.config['ddp'] == True:
            torch.save(self.model.state_dict(), save_path)
        else:
            if 'svdd' in self.config['model_name']:
                torch.save({'R': self.model.R,
                            'c': self.model.c,
                            'state_dict': self.model.state_dict(),}, save_path)
            else:
                torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Checkpoint saved to {save_path}, current ckpt is {ckpt_info}")

    def save_swa_ckpt(self):
        save_dir = self.log_dir
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"swa.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        torch.save(self.swa_model.state_dict(), save_path)
        self.logger.info(f"SWA Checkpoint saved to {save_path}")


    def save_feat(self, phase, fea, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        features = fea
        feat_name = f"feat_best.npy"
        save_path = os.path.join(save_dir, feat_name)
        np.save(save_path, features)
        self.logger.info(f"Feature saved to {save_path}")

    def save_data_dict(self, phase, data_dict, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'data_dict_{phase}.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(data_dict, file)
        self.logger.info(f"data_dict saved to {file_path}")

    def save_metrics(self, phase, metric_one_dataset, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'metric_dict_best.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(metric_one_dataset, file)
        self.logger.info(f"Metrics saved to {file_path}")

    def train_step(self, data_dict, iteration):
        if self.config['optimizer']['type']=='sam':
            for i in range(2):
                predictions = self.model(data_dict)
                losses = self.model.get_losses(data_dict, predictions)
                if i == 0:
                    pred_first = predictions
                    losses_first = losses
                self.optimizer.zero_grad()
                losses['overall'].backward()
                if i == 0:
                    self.optimizer.first_step(zero_grad=True)
                else:
                    self.optimizer.second_step(zero_grad=True)
            return losses_first, pred_first
        else:

            accumulation_steps = 2
 
            with autocast():
                predictions = self.model(data_dict)
                if type(self.model) is DDP:
                    losses = self.model.module.get_losses(data_dict, predictions)
                else:
                    losses = self.model.get_losses(data_dict, predictions)
 
            if torch.isnan(losses['overall']) or torch.isinf(losses['overall']):
                self.logger.warning(
                    f"[NaN Guard] Loss NaN/Inf di iterasi {iteration}, skip backward."
                )
                self.optimizer.zero_grad()
                return losses, predictions

            loss_to_backward = losses['overall'] / accumulation_steps
            self.scaler.scale(loss_to_backward).backward()
 
            if (iteration + 1) % accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
 
            return losses, predictions


    def train_epoch(
        self,
        epoch,
        train_data_loader,
        test_data_loaders=None,
        ):

        self.logger.info("===> Epoch[{}] start!".format(epoch))
        self.optimizer.zero_grad()
        if epoch>=1:
            times_per_epoch = 2
        else:
            times_per_epoch = 1



        test_step = len(train_data_loader) // times_per_epoch
        step_cnt = epoch * len(train_data_loader)

        # save the training data_dict
        data_dict = train_data_loader.dataset.data_dict
        self.save_data_dict('train', data_dict, ','.join(self.config['train_dataset']))
        # define training recorder
        train_recorder_loss = defaultdict(Recorder)
        train_recorder_metric = defaultdict(Recorder)

        for iteration, data_dict in tqdm(enumerate(train_data_loader),total=len(train_data_loader)):
            self.setTrain()
            self.model.epoch = epoch
            # more elegant and more scalable way of moving data to GPU
            for key in data_dict.keys():
                if data_dict[key]!=None and key!='name':
                    data_dict[key]=data_dict[key].cuda()

            losses, predictions = self.train_step(data_dict, iteration)

            # update learning rate

            if 'SWA' in self.config and self.config['SWA'] and epoch>self.config['swa_start']:
                self.swa_model.update_parameters(self.model)

            # compute training metric for each batch data
            if type(self.model) is DDP:
                batch_metrics = self.model.module.get_train_metrics(data_dict, predictions)
            else:
                batch_metrics = self.model.get_train_metrics(data_dict, predictions)

            # store data by recorder
            ## store metric
            for name, value in batch_metrics.items():
                train_recorder_metric[name].update(value)
            ## store loss
            for name, value in losses.items():
                train_recorder_loss[name].update(value)

            # run tensorboard to visualize the training process
            if iteration % 300 == 0 and self.config['local_rank']==0:
                if self.config['SWA'] and (epoch>self.config['swa_start'] or self.config['dry_run']):
                    self.scheduler.step()
                # info for loss
                loss_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_loss.items():
                    v_avg = v.average()
                    if v_avg == None:
                        loss_str += f"training-loss, {k}: not calculated"
                        continue
                    loss_str += f"training-loss, {k}: {v_avg}    "
                    # tensorboard-1. loss
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_loss/{k}', v_avg, global_step=step_cnt)
                self.logger.info(loss_str)
                # info for metric
                metric_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_metric.items():
                    v_avg = v.average()
                    if v_avg == None:
                        metric_str += f"training-metric, {k}: not calculated    "
                        continue
                    metric_str += f"training-metric, {k}: {v_avg}    "
                    # tensorboard-2. metric
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_metric/{k}', v_avg, global_step=step_cnt)
                self.logger.info(metric_str)



                # clear recorder.
                # Note we only consider the current 300 samples for computing batch-level loss/metric
                for name, recorder in train_recorder_loss.items():  # clear loss recorder
                    recorder.clear()
                for name, recorder in train_recorder_metric.items():  # clear metric recorder
                    recorder.clear()

            # run test
            if (step_cnt+1) % test_step == 0:
                if test_data_loaders is not None and (not self.config['ddp'] ):
                    self.logger.info("===> Test start!")
                    test_best_metric = self.test_epoch(
                        epoch,
                        iteration,
                        test_data_loaders,
                        step_cnt,
                    )
                elif test_data_loaders is not None and (self.config['ddp'] and dist.get_rank() == 0):
                    self.logger.info("===> Test start!")
                    test_best_metric = self.test_epoch(
                        epoch,
                        iteration,
                        test_data_loaders,
                        step_cnt,
                    )
                else:
                    test_best_metric = None

                    # total_end_time = time.time()
            # total_elapsed_time = total_end_time - total_start_time
            # print("总花费的时间: {:.2f} 秒".format(total_elapsed_time))
            step_cnt += 1
        
        if hasattr(self.model, 'reset_epoch_buffer'):
            self.model.reset_epoch_buffer()
            self.logger.info(f"[Epoch {epoch}] Group loss buffer di-reset.")
        elif hasattr(self.model, 'module') and hasattr(self.model.module, 'reset_epoch_buffer'):
            # Handle DDP wrapper
            self.model.module.reset_epoch_buffer()
            self.logger.info(f"[Epoch {epoch}] Group loss buffer di-reset (DDP).")

        if self.config.get('save_ckpt', False) and self.config.get('local_rank', 0) == 0:
            self.save_resume_ckpt(epoch)

        # ════════════════════════════════════════════════════════
        # PROTOKOL BERSIH-BERSIH MEMORI (CPU RAM & GPU VRAM)
        # ════════════════════════════════════════════════════════
        
        # 1. Hapus referensi ke variabel besar agar terdeteksi sebagai "sampah"
        try:
            del data_dict, losses, predictions, batch_metrics
        except UnboundLocalError:
            pass
            
        # 2. Paksa Python menyapu RAM Sistem (CPU)
        gc.collect()
        
        # 3. Bersihkan sisa memori yang nyangkut di GPU (VRAM)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.logger.info(f"[Memory Sweep] RAM & VRAM dibersihkan untuk Epoch {epoch+1}.")
        return test_best_metric

    def get_respect_acc(self, prob, label, threshold=0.5):
        pred = np.where(prob >= threshold, 1, 0)
        judge = (pred == label)
        real_idx = np.where(label == 0)[0]
        fake_idx = np.where(label == 1)[0]
        
        # Tambahkan safety check agar tidak dibagi nol
        acc_real = np.count_nonzero(judge[real_idx]) / len(real_idx) if len(real_idx) > 0 else 0.0
        acc_fake = np.count_nonzero(judge[fake_idx]) / len(fake_idx) if len(fake_idx) > 0 else 0.0

        return acc_real, acc_fake

    def test_one_dataset(self, data_loader):
        # define test recorder
        test_recorder_loss = defaultdict(Recorder)
        prediction_lists = []
        feature_lists=[]
        label_lists = []
        for i, data_dict in tqdm(enumerate(data_loader),total=len(data_loader)):
            # get data
            if 'label_spe' in data_dict:
                data_dict.pop('label_spe')  # remove the specific label
            data_dict['label'] = torch.where(data_dict['label']!=0, 1, 0)  # fix the label to 0 and 1 only
            # move data to GPU elegantly
            for key in data_dict.keys():
                if data_dict[key]!=None:
                    data_dict[key]=data_dict[key].cuda()
            # model forward without considering gradient computation
            predictions = self.inference(data_dict)
            label_lists += list(data_dict['label'].cpu().detach().numpy())
            prediction_lists += list(predictions['prob'].cpu().detach().numpy())
            feature_lists += list(predictions['feat'].cpu().detach().numpy())
            if type(self.model) is not AveragedModel:
                # compute all losses for each batch data
                if type(self.model) is DDP:
                    losses = self.model.module.get_losses(data_dict, predictions)
                else:
                    losses = self.model.get_losses(data_dict, predictions)

                # store data by recorder
                for name, value in losses.items():
                    test_recorder_loss[name].update(value)

        return test_recorder_loss, np.array(prediction_lists), np.array(label_lists),np.array(feature_lists)

    def save_best(self,epoch,iteration,step,losses_one_dataset_recorder,key,metric_one_dataset):
        best_metric = self.best_metrics_all_time[key].get(self.metric_scoring,
                                                          float('-inf') if self.metric_scoring != 'eer' else float(
                                                              'inf'))
        # Check if the current score is an improvement
        improved = (metric_one_dataset[self.metric_scoring] > best_metric) if self.metric_scoring != 'eer' else (
                    metric_one_dataset[self.metric_scoring] < best_metric)
        if improved:
            for save_key in ['acc', 'auc', 'eer', 'ap', 'f1', 
                            'video_auc', 'opt_thresh',
                            'gfpr', 'gtpr', 'efpr', 'eo']:
                if save_key in metric_one_dataset:
                    self.best_metrics_all_time[key][save_key] = metric_one_dataset[save_key]

            if key == 'avg':
                self.best_metrics_all_time[key]['dataset_dict'] = metric_one_dataset['dataset_dict']
            # Save checkpoint, feature, and metrics if specified in config
            if self.config['save_ckpt'] and key not in FFpp_pool:
                self.save_ckpt('test', key, f"{epoch}+{iteration}")
            self.save_metrics('test', metric_one_dataset, key)
        
        # Hanya untuk dataset yg punya EO
        is_indo = any(x in key for x in ['Indo-MM', 'Indo-FS', 'Indo-VC', 'IndoDeepfake'])
        if is_indo and 'eo' in metric_one_dataset:
            current_eo = metric_one_dataset['eo']
            best_eo = self.best_fairness_all_time[key].get('eo', float('inf'))
            
            fairness_improved = (current_eo < best_eo)  # EO: makin kecil makin baik
            
            if fairness_improved:
                self.logger.info(
                    f"[Fairness BEST] {key} EO membaik: {best_eo:.4f} → {current_eo:.4f}"
                )
                # Update tracker fairness
                for save_key in ['acc', 'auc', 'eer', 'ap', 'f1',
                                'video_auc', 'opt_thresh',
                                'gfpr', 'gtpr', 'efpr', 'eo']:
                    if save_key in metric_one_dataset:
                        self.best_fairness_all_time[key][save_key] = metric_one_dataset[save_key]
                
                # Simpan checkpoint fairness terpisah
                if self.config['save_ckpt'] and key not in FFpp_pool:
                    self._save_fairness_ckpt(key, epoch, iteration)
                
                # Simpan metrics fairness terpisah
                self._save_fairness_metrics(key, metric_one_dataset)

        if losses_one_dataset_recorder is not None:
            # info for each dataset
            loss_str = f"dataset: {key}    step: {step}    "
            for k, v in losses_one_dataset_recorder.items():
                writer = self.get_writer('test', key, k)
                v_avg = v.average()
                if v_avg == None:
                    print(f'{k} is not calculated')
                    continue
                # tensorboard-1. loss
                writer.add_scalar(f'test_losses/{k}', v_avg, global_step=step)
                loss_str += f"testing-loss, {k}: {v_avg}    "
            self.logger.info(loss_str)
        # tqdm.write(loss_str)
        metric_str = f"dataset: {key}    step: {step}    "
        for k, v in metric_one_dataset.items():
            if k == 'pred' or k == 'label' or k=='dataset_dict':
                continue
            metric_str += f"testing-metric, {k}: {v}    "
            # tensorboard-2. metric
            writer = self.get_writer('test', key, k)
            writer.add_scalar(f'test_metrics/{k}', v, global_step=step)
        if 'pred' in metric_one_dataset:
            thresh = metric_one_dataset.get('opt_thresh', 0.5) 
            acc_real, acc_fake = self.get_respect_acc(metric_one_dataset['pred'], metric_one_dataset['label'], threshold=thresh)
            
            metric_str += f'testing-metric, acc_real:{acc_real}; acc_fake:{acc_fake}'
            writer.add_scalar(f'test_metrics/acc_real', acc_real, global_step=step)
            writer.add_scalar(f'test_metrics/acc_fake', acc_fake, global_step=step)
        self.logger.info(metric_str)

    def _save_fairness_ckpt(self, dataset_key, epoch, iteration):
        """Simpan checkpoint khusus fairness terbaik."""
        save_dir = os.path.join(self.log_dir, 'test', dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'ckpt_best_fairness.pth')
        
        if isinstance(self.model, DDP):
            torch.save(self.model.state_dict(), save_path)
        else:
            torch.save(self.model.state_dict(), save_path)
        
        self.logger.info(
            f"[Fairness ckpt] Tersimpan: {save_path} (epoch {epoch}+iter {iteration})"
        )


    def _save_fairness_metrics(self, dataset_key, metric_one_dataset):
        """Simpan metrics saat fairness terbaik."""
        save_dir = os.path.join(self.log_dir, 'test', dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'metric_dict_best_fairness.pickle')
        
        with open(save_path, 'wb') as f:
            pickle.dump(metric_one_dataset, f)
        self.logger.info(f"[Fairness metrics] Tersimpan: {save_path}")

    def test_epoch(self, epoch, iteration, test_data_loaders, step):
        # set model to eval mode
        self.setEval()

        # define test recorder
        losses_all_datasets = {}
        metrics_all_datasets = {}
        best_metrics_per_dataset = defaultdict(dict)  # best metric for each dataset, for each metric
        avg_metric = {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0,'video_auc': 0,'dataset_dict':{}}
        # testing for all test data
        keys = test_data_loaders.keys()
        for key in keys:
            # save the testing data_dict
            data_dict = test_data_loaders[key].dataset.data_dict
            self.save_data_dict('test', data_dict, key)

            # compute loss for each dataset
            losses_one_dataset_recorder, predictions_nps, label_nps, feature_nps = self.test_one_dataset(test_data_loaders[key])
            # print(f'stack len:{predictions_nps.shape};{label_nps.shape};{len(data_dict["image"])}')
            losses_all_datasets[key] = losses_one_dataset_recorder
            metric_one_dataset=get_test_metrics(y_pred=predictions_nps,y_true=label_nps,img_names=data_dict['image'], dataset_name=key)
            for metric_name, value in metric_one_dataset.items():
                if metric_name in avg_metric:
                    avg_metric[metric_name]+=value
            avg_metric['dataset_dict'][key] = metric_one_dataset[self.metric_scoring]
            if type(self.model) is AveragedModel:
                metric_str = f"Iter Final for SWA:    "
                for k, v in metric_one_dataset.items():
                    metric_str += f"testing-metric, {k}: {v}    "
                self.logger.info(metric_str)
                continue
            self.save_best(epoch,iteration,step,losses_one_dataset_recorder,key,metric_one_dataset)

        if len(keys)>0 and self.config.get('save_avg',False):
            # calculate avg value
            for key in avg_metric:
                if key != 'dataset_dict':
                    avg_metric[key] /= len(keys)
            self.save_best(epoch, iteration, step, None, 'avg', avg_metric)

        indo_keys = [k for k in self.best_fairness_all_time.keys() 
                    if any(x in k for x in ['Indo-MM', 'Indo-FS', 'Indo-VC'])]
        if indo_keys:
            self.logger.info("===> Best Fairness So Far ===")
            for k in indo_keys:
                fd = self.best_fairness_all_time[k]
                self.logger.info(
                    f"  {k}: EO={fd.get('eo', 'N/A'):.4f}  "
                    f"gfpr={fd.get('gfpr', 'N/A'):.4f}  "
                    f"AUC={fd.get('auc', 'N/A'):.4f}"
                )
                
        self.logger.info('===> Test Done!')
        return self.best_metrics_all_time  # return all types of mean metrics for determining the best ckpt

    @torch.no_grad()
    def inference(self, data_dict):
        predictions = self.model(data_dict, inference=True)
        return predictions
