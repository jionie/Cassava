# import os and define graphic card
import os

os.environ["OMP_NUM_THREADS"] = "1"

# import common libraries
import random
import argparse
import numpy as np
import time

# import pytorch related libraries
import torch
from tensorboardX import SummaryWriter
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup, get_linear_schedule_with_warmup

# import dataset class
from dataset.dataset import get_train_val_split, get_train_val_loader

# import utils
from utils.ranger import Ranger
from utils.lrs_scheduler import GradualWarmupScheduler, WarmRestart, CosineAnnealingWarmUpRestarts
from utils.metric import accuracy_metric
from utils.file import Logger
from utils.loss_function import CrossEntropyLossOHEM

# import model
from model.model import CassavaModel

# import config
from config import Config

############################################################################## Define Argument
parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--fold', type=int, default=0, required=False, help="specify the fold for training")
parser.add_argument('--model_type', type=str, default="se_resnext50", required=False, help="specify the model type")
parser.add_argument('--seed', type=int, default=2020, required=False, help="specify the seed")
parser.add_argument('--batch_size', type=int, default=16, required=False, help="specify the batch size")
parser.add_argument('--accumulation_steps', type=int, default=1, required=False, help="specify the accumulation_steps")
parser.add_argument('--height', type=int, default=512, required=False, help="specify the image height")
parser.add_argument('--width', type=int, default=512, required=False, help="specify the image width")


############################################################################## seed All
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
    os.environ['PYHTONHASHseed'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True


############################################################################## Class for Plant
class Cassava():
    def __init__(self, config):
        super(Cassava).__init__()
        self.config = config
        self.setup_logger()
        self.setup_gpu()
        self.load_data()
        self.prepare_train()
        self.setup_model()

    def setup_logger(self):
        self.log = Logger()
        self.log.open((os.path.join(self.config.checkpoint_folder, "train_log.txt")), mode='a+')

    def setup_gpu(self):
        # confirm the device which can be either cpu or gpu
        self.config.use_gpu = torch.cuda.is_available()
        self.num_device = torch.cuda.device_count()
        if self.config.use_gpu:
            self.config.device = 'cuda'
            if self.num_device <= 1:
                self.config.data_parallel = False
            elif self.config.data_parallel:
                torch.multiprocessing.set_start_method('spawn', force=True)
        else:
            self.config.device = 'cpu'
            self.config.data_parallel = False

    def load_data(self):
        self.log.write('\nLoading data...')

        get_train_val_split(data_path=self.config.data_path,
                            save_path=self.config.save_path,
                            n_splits=self.config.n_splits,
                            seed=self.config.seed,
                            split=self.config.split)

        # no test data for now
        # self.test_data_loader = get_test_loader(data_path=self.config.data_path,
        #                                         batch_size=self.config.val_batch_size,
        #                                         num_workers=self.config.num_workers,
        #                                         transforms=self.config.val_transforms)

        self.train_data_loader, self.val_data_loader = get_train_val_loader(data_path=self.config.data_path,
                                                                            seed=self.config.seed,
                                                                            fold=self.config.fold,
                                                                            batch_size=self.config.batch_size,
                                                                            val_batch_size=self.config.val_batch_size,
                                                                            num_workers=self.config.num_workers,
                                                                            transforms=self.config.transforms,
                                                                            val_transforms=self.config.val_transforms)

    def prepare_train(self):
        # preparation for training
        self.step = 0
        self.epoch = 0
        self.finished = False
        self.valid_epoch = 0
        self.train_loss, self.valid_loss, self.valid_metric_optimal = float('inf'), float('inf'), float('-inf')
        self.writer = SummaryWriter()
        ############################################################################### eval setting
        self.eval_step = int(len(self.train_data_loader) * self.config.saving_rate)
        self.log_step = int(len(self.train_data_loader) * self.config.progress_rate)
        self.eval_count = 0
        self.count = 0

    def pick_model(self):
        # for switching model
        self.model = CassavaModel(model_name=self.config.model_type, num_classes=5).to(self.config.device)

    def differential_lr(self):

        param_optimizer = list(self.model.named_parameters())

        prefix = "backbone"

        def is_backbone(n):
            return prefix in n

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        self.optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and is_backbone(n)],
             'lr': self.config.backbone_lr,
             'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and not is_backbone(n)],
             'lr': self.config.lr,
             'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and is_backbone(n)],
             'lr': self.config.backbone_lr,
             'weight_decay': 0.0},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and not is_backbone(n)],
             'lr': self.config.lr,
             'weight_decay': 0.0}
        ]

    def prepare_optimizer(self):

        # differential lr for each sub module first
        self.differential_lr()

        # optimizer
        if self.config.optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(self.optimizer_grouped_parameters, eps=self.config.adam_epsilon)
        elif self.config.optimizer_name == "Ranger":
            self.optimizer = Ranger(self.optimizer_grouped_parameters)
        elif self.config.optimizer_name == "AdamW":
            self.optimizer = torch.optim.AdamW(self.optimizer_grouped_parameters, eps=self.config.adam_epsilon)
        else:
            raise NotImplementedError

        # lr scheduler
        if self.config.lr_scheduler_name == "WarmupCosineAnealing":
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.config.num_epoch - 1)
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=10, total_epoch=1,
                                                    after_scheduler=scheduler_cosine)
            self.lr_scheduler_each_iter = False
        elif self.config.lr_scheduler_name == "WarmCosineAnealingRestart-v2":
            T = len(self.train_data_loader) // self.config.accumulation_steps * 20  # cycle
            self.scheduler = CosineAnnealingWarmUpRestarts(self.optimizer, T_0=T, T_mult=1, eta_max=self.config.lr * 25,
                                                           T_up=T // 20, gamma=0.2)
            self.lr_scheduler_each_iter = True
        elif self.config.lr_scheduler_name == "WarmCosineAnealingRestart":
            num_train_optimization_steps = self.config.num_epoch * len(self.train_data_loader) \
                                           // self.config.accumulation_steps
            self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(self.optimizer,
                                                                                num_warmup_steps=int(
                                                                                    num_train_optimization_steps * self.config.warmup_proportion),
                                                                                num_training_steps=num_train_optimization_steps)
            self.lr_scheduler_each_iter = True
        elif self.config.lr_scheduler_name == "WarmRestart":
            self.scheduler = WarmRestart(self.optimizer, T_max=5, T_mult=1, eta_min=1e-6)
            self.lr_scheduler_each_iter = False
        elif self.config.lr_scheduler_name == "WarmupLinear":
            num_train_optimization_steps = self.config.num_epoch * len(self.train_data_loader) \
                                           // self.config.accumulation_steps
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                             num_warmup_steps=int(
                                                                 num_train_optimization_steps * self.config.warmup_proportion),
                                                             num_training_steps=num_train_optimization_steps)
            self.lr_scheduler_each_iter = True
        elif self.config.lr_scheduler_name == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.4,
                                                                        patience=1, min_lr=1e-6)
            self.lr_scheduler_each_iter = False
        else:
            raise NotImplementedError

        # lr scheduler step for checkpoints
        if self.lr_scheduler_each_iter:
            self.scheduler.step(self.step)
        else:
            self.scheduler.step(self.epoch)

    def prepare_apex(self):
        self.scaler = torch.cuda.amp.GradScaler()

    def load_check_point(self):
        self.log.write('Model loaded as {}.'.format(self.config.load_point))
        checkpoint_to_load = torch.load(self.config.load_point, map_location=self.config.device)
        self.step = checkpoint_to_load['step']
        self.epoch = checkpoint_to_load['epoch']
        self.valid_metric_optimal = checkpoint_to_load['valid_metric_optimal']

        model_state_dict = checkpoint_to_load['model']
        state_dict = self.model.state_dict()

        keys = list(state_dict.keys())

        for key in keys:
            if any(s in key for s in self.config.skip_layers):
                continue
            try:
                state_dict[key] = model_state_dict[key]
            except:
                print("Missing key:", key)

        self.model.load_state_dict(state_dict)

        if self.config.load_optimizer:
            self.optimizer.load_state_dict(checkpoint_to_load['optimizer'])

    def save_check_point(self):
        # save model, optimizer, and everything required to keep
        if self.num_device > 1:
            checkpoint_to_save = {
                'step': self.step,
                'epoch': self.epoch,
                'valid_metric_optimal': self.valid_metric_optimal,
                'model': self.model.module.state_dict()
            }
        else:
            checkpoint_to_save = {
                'step': self.step,
                'epoch': self.epoch,
                'valid_metric_optimal': self.valid_metric_optimal,
                'model': self.model.state_dict()
            }

        save_path = self.config.save_point.format(self.step, self.epoch)
        torch.save(checkpoint_to_save, save_path)
        self.log.write('Model saved as {}.'.format(save_path))

    def setup_model(self):
        self.pick_model()

        if self.config.data_parallel:
            self.prepare_optimizer()

            if self.config.apex:
                self.prepare_apex()

            if self.config.reuse_model:
                self.load_check_point()

            self.model = torch.nn.DataParallel(self.model)

        else:
            if self.config.reuse_model:
                self.load_check_point()

            self.prepare_optimizer()

            if self.config.apex:
                self.prepare_apex()

    def count_parameters(self):
        # get total size of trainable parameters
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def show_info(self):
        # show general information before training
        self.log.write('\n*General Setting*')
        self.log.write('\nseed: {}'.format(self.config.seed))
        self.log.write('\nmodel: {}'.format(self.config.model_type))
        self.log.write('\ntrainable parameters:{:,.0f}'.format(self.count_parameters()))
        self.log.write("\nmodel's state_dict:")
        self.log.write('\ndevice: {}'.format(self.config.device))
        self.log.write('\nuse gpu: {}'.format(self.config.use_gpu))
        self.log.write('\ndevice num: {}'.format(self.num_device))
        self.log.write('\noptimizer: {}'.format(self.optimizer))
        self.log.write('\nreuse model: {}'.format(self.config.reuse_model))
        if self.config.reuse_model:
            self.log.write('\nModel restored from {}.'.format(self.config.load_point))
        self.log.write('\n')

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

    def cutmix(self, image, target, alpha, clip=None):
        if clip is None:
            clip = [0.3, 0.7]
        indices = torch.randperm(image.size(0))
        shuffled_target = target[indices]

        lam = np.clip(np.random.beta(alpha, alpha), clip[0], clip[1])
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(image.size(), lam)
        image[:, :, bbx1:bbx2, bby1:bby2] = image[indices, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))
        targets = (target, shuffled_target, lam)

        return image, targets, indices

    def cutmix_criterion(self, prediction, targets):
        target, shuffled_target, lam = targets
        return self.criterion(prediction, target) * lam + self.criterion(prediction, shuffled_target) * (1-lam)

    def train_op(self):
        self.show_info()
        self.log.write('** start training here! **\n')
        self.log.write('   batch_size=%d,  accumulation_steps=%d\n' % (self.config.batch_size,
                                                                       self.config.accumulation_steps))
        self.log.write('   experiment  = %s\n' % str(__file__.split('/')[-2:]))

        self.timer = time.time()
        self.criterion = CrossEntropyLossOHEM(top_k=1, ignore_index=None)

        while self.epoch <= self.config.num_epoch:

            self.train_metrics = 0

            # update lr and start from start_epoch
            if (self.epoch >= 1) and (not self.lr_scheduler_each_iter) \
                    and (self.config.lr_scheduler_name != "ReduceLROnPlateau"):
                self.scheduler.step(self.epoch - 1)

            self.log.write("Epoch%s\n" % self.epoch)
            self.log.write('\n')

            sum_train_loss = np.zeros_like(self.train_loss)
            sum_train = np.zeros_like(self.train_loss)

            # init optimizer
            torch.cuda.empty_cache()
            self.model.zero_grad()

            for tr_batch_i, (image, label, pseudo_label) in enumerate(self.train_data_loader):

                rate = 0
                for param_group in self.optimizer.param_groups:
                    rate += param_group['lr'] / len(self.optimizer.param_groups)

                # set model training mode
                self.model.train()

                # set input to cuda mode
                image = image.to(self.config.device).float()
                label = label.to(self.config.device)
                pseudo_label = pseudo_label.to(self.config.device)

                image, targets, indices = self.cutmix(image, label, 1)
                pseudo_targets = (targets[0], pseudo_label[indices], targets[2])

                prediction = self.model(image)
                if self.config.apex:
                    with torch.cuda.amp.autocast():
                        loss = 0.7 * self.cutmix_criterion(prediction, targets) + 0.3 * self.cutmix_criterion(prediction, pseudo_targets)
                    self.scaler.scale(loss).backward()
                else:
                    loss = 0.7 * self.cutmix_criterion(prediction, targets) + 0.3 * self.cutmix_criterion(prediction, pseudo_targets)
                    loss.backward()

                if (tr_batch_i + 1) % self.config.accumulation_steps == 0:
                    # use apex
                    if self.config.apex:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm,
                                                       norm_type=2)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm,
                                                       norm_type=2)
                        self.optimizer.step()

                    self.model.zero_grad()

                    # adjust lr
                    if (self.lr_scheduler_each_iter):
                        self.scheduler.step()

                    self.writer.add_scalar('train_loss_' + str(self.config.fold), loss.item(),
                                           (self.epoch - 1) * len(
                                               self.train_data_loader) * self.config.batch_size + tr_batch_i *
                                           self.config.batch_size)
                    self.step += 1

                # translate to predictions
                prediction = np.argmax(prediction.detach().cpu().numpy(), axis=-1)
                label = label.detach().cpu().numpy()

                # running mean
                metrics = accuracy_metric(np.squeeze(label), np.squeeze(prediction))
                if metrics is not None:
                    self.train_metrics = (self.train_metrics * tr_batch_i + metrics) / (tr_batch_i + 1)

                l = np.array([loss.item() * self.config.batch_size])
                n = np.array([self.config.batch_size])
                sum_train_loss = sum_train_loss + l
                sum_train = sum_train + n

                # log for training
                if (tr_batch_i + 1) % self.log_step == 0:
                    train_loss = sum_train_loss / (sum_train + 1e-12)
                    sum_train_loss[...] = 0
                    sum_train[...] = 0

                    curr_time = time.time()
                    self.log.write(
                        'time elapsed: %f lr: %f train loss: %f train_acc: %f \n' % (curr_time - self.timer, rate, train_loss[0], self.train_metrics))
                    self.timer = curr_time

                if (tr_batch_i + 1) % self.eval_step == 0:
                    self.evaluate_op()

            if self.count >= self.config.early_stopping:
                break

            self.epoch += 1

    def evaluate_op(self):

        self.eval_count += 1
        valid_loss = np.zeros(1, np.float32)
        valid_num = np.zeros_like(valid_loss)

        self.eval_metrics = 0
        self.eval_prediction = None
        self.eval_label = None

        with torch.no_grad():

            # init cache
            torch.cuda.empty_cache()
            for val_batch_i, (image, label, pseudo_label) in enumerate(self.val_data_loader):

                # set model to eval mode
                self.model.eval()

                # set input to cuda mode
                image = image.to(self.config.device).float()
                label = label.to(self.config.device)
                pseudo_label = pseudo_label.to(self.config.device)

                prediction = self.model(image)
                loss = 0.7 * self.criterion(prediction, label) + 0.3 * self.criterion(prediction, pseudo_label)

                self.writer.add_scalar('val_loss_' + str(self.config.fold), loss.item(), (self.eval_count - 1) * len(
                    self.val_data_loader) * self.config.val_batch_size + val_batch_i * self.config.val_batch_size)

                # translate to predictions
                prediction = np.argmax(prediction.detach().cpu().numpy(), axis=-1)
                label = label.detach().cpu().numpy()

                if self.eval_prediction is None:
                    self.eval_prediction = prediction
                else:
                    self.eval_prediction = np.concatenate([self.eval_prediction, prediction], axis=0)

                if self.eval_label is None:
                    self.eval_label = label
                else:
                    self.eval_label = np.concatenate([self.eval_label, label], axis=0)

                l = np.array([loss.item() * self.config.val_batch_size])
                n = np.array([self.config.val_batch_size])
                valid_loss = valid_loss + l
                valid_num = valid_num + n

            self.eval_metrics = accuracy_metric(np.squeeze(self.eval_label), np.squeeze(self.eval_prediction))
            valid_loss = valid_loss / valid_num

            self.log.write(
                'val loss: %f ' % (valid_loss[0]) +
                'val_acc: %f \n'
                % self.eval_metrics)

        if self.config.lr_scheduler_name == "ReduceLROnPlateau":
            self.scheduler.step(self.eval_metrics)

        if self.eval_metrics >= self.valid_metric_optimal:

            self.log.write('Validation metric improved ({:.6f} --> {:.6f}).  Saving model ...'.format(
                self.valid_metric_optimal, self.eval_metrics))

            self.valid_metric_optimal = self.eval_metrics
            self.save_check_point()

            self.count = 0

            np.save(
                os.path.join(self.config.checkpoint_folder_all_fold, "predict_fold_{}.npy".format(self.config.fold)),
                np.squeeze(self.eval_prediction))
            np.save(
                os.path.join(self.config.checkpoint_folder_all_fold, "label_fold_{}.npy".format(self.config.fold)),
                np.squeeze(self.eval_label))

        else:
            self.count += 1

        return


if __name__ == "__main__":
    args = parser.parse_args()

    # update fold
    config = Config(args.fold, model_type=args.model_type, seed=args.seed, batch_size=args.batch_size,
                    accumulation_steps=args.accumulation_steps, height=args.height, width=args.width)
    seed_everything(config.seed)
    qa = Cassava(config)
    qa.train_op()
