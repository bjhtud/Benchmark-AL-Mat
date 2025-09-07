#!/usr/bin/env python3
"""
LL4AL (Learning Loss for Active Learning) Strategy

This module implements the LL4AL active learning strategy using neural networks
to predict loss values for uncertainty estimation in regression tasks.

Author: Jinghou Bi
Email: jinghou.bi@tu-dresden.de
License: MIT License
Created: 2025-01-06
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.metrics import r2_score
from .utils_models.LL4AL_model import MainNet, LossNet

import numpy as np
import warnings

warnings.filterwarnings("ignore")


class LearningLoss:
    """
    Learning Loss for Active Learning strategy.

    This class implements an active learning strategy that uses neural networks
    to predict loss values for uncertainty estimation.

    Attributes:
        BATCH (int): Batch size for training.
        LR (float): Learning rate for optimization.
        MARGIN (float): Margin for loss prediction loss function.
        WEIGHT (float): Weight for loss prediction module.
        EPOCH (int): Number of training epochs.
        EPOCHL (int): Number of epochs before stopping gradient propagation.
        WDECAY (float): Weight decay for regularization.
        random_state (int): Random seed for reproducibility.
    """

    def __init__(self, BATCH=32, LR=0.01, MARGIN=0.7, WEIGHT=1.5, EPOCH=200, EPOCHL=30, WDECAY=5e-4, random_state=42):
        """
        Initialize the LearningLoss strategy.

        Args:
            BATCH (int, optional): Batch size. Defaults to 32.
            LR (float, optional): Learning rate. Defaults to 0.01.
            MARGIN (float, optional): Margin for loss function. Defaults to 0.7.
            WEIGHT (float, optional): Weight for loss prediction. Defaults to 1.5.
            EPOCH (int, optional): Training epochs. Defaults to 200.
            EPOCHL (int, optional): Epochs before gradient stop. Defaults to 30.
            WDECAY (float, optional): Weight decay. Defaults to 5e-4.
            random_state (int, optional): Random seed. Defaults to 42.
        """
        self.BATCH = BATCH
        self.LR = LR
        self.MARGIN = MARGIN
        self.WEIGHT = WEIGHT
        self.EPOCH = EPOCH
        self.EPOCHL = EPOCHL
        self.WDECAY = WDECAY
        self.iters = 0
        self.device = torch.device('cpu')
        self.random_state = random_state
        torch.manual_seed(self.random_state)
        
    def fit(self, X_labeled, y_labeled):
        """
        Fit method for compatibility.

        Args:
            X_labeled: Labeled features (unused).
            y_labeled: Labeled targets (unused).
        """
        pass  # do nothing

    # ============================== 定义主动学习相关函数 ==============================
    def LossPredLoss(self, input, target, margin=1.0,
                     reduction='mean'):  # inout 是过lossnet后得到的预测损失，target是训练集中使用mainnet得到的真实损失
        assert len(input) % 2 == 0, 'the batch size is not even.'
        assert input.shape == input.flip(0).shape

        input = (input - input.flip(0))[
                :len(input) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
        target = (target - target.flip(0))[:len(target) // 2]
        target = target.detach()
        # 这个损失函数说明损失预测模型的实际目的是得到对应数据的损失值的大小关系而不是确定的损失值
        one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors

        if reduction == 'mean':
            loss = torch.sum(torch.clamp(margin - one * input, min=0))
            loss = loss / input.size(0)  # Note that the size of input is already halved
        elif reduction == 'none':
            loss = torch.clamp(margin - one * input, min=0)
        else:
            NotImplementedError()

        return loss

    # 计算未标记数据的loss，作为不确定性度量
    def get_uncertainty(self, models, unlabeled_loader, criterion):
        models['backbone'].eval()
        models['module'].eval()
        uncertainty = torch.tensor([])
        # 创建一个变量，用于记录使用lossnet预测未标记数据集“比较大小”的精确度
        accuracy_list = []

        with torch.no_grad():
            for (inputs, labels) in unlabeled_loader:
                # print(f'inputs shape: {inputs.shape}')
                # print(f'labels shape: {labels.shape}')
                # inputs = inputs
                # labels = labels

                scores, features = models['backbone'](inputs)  # 模型返回了预测值和中间层特征组成的元组
                pred_loss = models['module'](features)  # pred_loss = criterion(scores, labels) # ground truth loss
                pred_loss = pred_loss.view(pred_loss.size(0))  # 将预测损失展平
                # 评估环节
                # 计算target_loss
                target_loss = criterion(scores, labels)
                target_loss = target_loss.detach()
                target_loss = target_loss.view(target_loss.size(0))
                # # 比较两个张量的大小关系
                # print(f"pred_loss: {pred_loss}")
                # print(f"pred_loss.flip(0): {pred_loss.flip(0)}")
                # print(f"len(pred_loss) // 2: {len(pred_loss) // 2}")
                # pred_loss_compare = (pred_loss - pred_loss.flip(0))[:len(pred_loss) // 2]
                # print(f'lenth of pred_loss: {len(pred_loss_compare)}')
                # real_loss_compare = (target_loss - target_loss.flip(0))[:len(pred_loss) // 2]
                # # 初始化 accuracy
                # accuracy_batch = 0
                # # 比较两个张量相同索引位置的值
                # for val1, val2 in zip(pred_loss_compare, real_loss_compare):
                #     if (val1 > 0 and val2 > 0) or (val1 < 0 and val2 < 0):
                #         accuracy_batch += 1
                # accuracy_batch = accuracy_batch / len(pred_loss_compare)
                # accuracy_list.append(accuracy_batch)
                # 某一批“比较矩阵”的对比
                # tensor([-0.5520, 0.3524, -0.1835, -0.4242, 0.6650, -0.3441, -0.4357, 0.2084, 0.1111, 0.1558, -0.2874, 0.5138, -0.3408, 0.5229, -0.2080, 0.5975])
                # tensor([-0.0005, 0.0865, -0.0004, -0.0141, -0.0327, -0.0041, -0.0097, 0.0273, 0.0225, 0.0050, -0.0747, 0.0164, 0.0169, -0.0585, -0.0400, 0.0346])

                uncertainty = torch.cat((uncertainty, pred_loss), 0)  # 合并所有批次的预测损失
        # accuracy = sum(accuracy_list) / len(accuracy_list)

        return uncertainty.cpu()

    # Train Utils
    def train_epoch(self, models, criterion, optimizers, dataloaders, epoch, epoch_loss, vis=None, plot_data=None):
        models['backbone'].train()
        models['module'].train()

        for data in dataloaders['train']:
            inputs = data[0]
            labels = data[1]

            optimizers['backbone'].zero_grad()
            optimizers['module'].zero_grad()

            scores, features = models['backbone'](inputs)
            target_loss = criterion(scores, labels)  # 传入一个现有的criterion是为了计算主网络的target_loss，是最终loss的一部分

            if epoch > epoch_loss:
                # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
                features[0] = features[0].detach()
                features[1] = features[1].detach()
                # features[2] = features[2].detach()

            pred_loss = models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))

            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
            m_module_loss = self.LossPredLoss(pred_loss, target_loss, margin=self.MARGIN)
            loss = m_backbone_loss + self.WEIGHT * m_module_loss

            loss.backward()
            optimizers['backbone'].step()
            optimizers['module'].step()  # 更新模型参数

    # 原论文是分类任务，需要修改
    def test(self, models, dataloaders, criterion, mode='val'):
        assert mode == 'val' or mode == 'test'
        models['backbone'].eval()
        models['module'].eval()

        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in dataloaders['test']:
                inputs, targets = inputs.to(self.device), targets.to(self.device)  # 确保在正确设备上
                outputs, features = models['backbone'](inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)

        test_loss /= len(dataloaders['test'].dataset)
        print(f'Test Loss: {test_loss:.4f}')
        # 调用R2评估测试集
        r2 = r2_score(targets, outputs)
        r2 = round(r2, 4)
        print(f"r2_score : {r2:.4f}")

        return test_loss, r2

    def train(self, models, criterion, optimizers, dataloaders, num_epochs, epoch_loss):
        # print('>> Train a Model.')

        for epoch in range(num_epochs):
            # schedulers['backbone'].step()
            # schedulers['module'].step()

            self.train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss)

        # print('>> Finished.')

    # ============================== 主程序 ==============================
    # plt.ion()

    def query(self, X_unlabeled, X_labeled, y_unlabeled, y_labeled, n_act=1):
        """
        Query samples using learning loss strategy.

        Args:
            X_unlabeled (pd.DataFrame): Unlabeled data features.
            X_labeled (pd.DataFrame): Labeled data features.
            y_unlabeled (pd.DataFrame): Unlabeled data targets.
            y_labeled (pd.DataFrame): Labeled data targets.
            n_act (int, optional): Number of samples to select. Defaults to 1.

        Returns:
            list: Indices of selected samples.
        """
        # 生成标记数据集dataloader和未标记数据集dataloader，注意X_labeled等都是dataframe
        X_labeled_tensor = torch.tensor(X_labeled.values).float()
        y_labeled_tensor = torch.tensor(y_labeled.values).float()
        X_unlabeled_tensor = torch.tensor(X_unlabeled.values).float()
        y_unlabeled_tensor = torch.tensor(y_unlabeled.values).float()

        train_loader = DataLoader(TensorDataset(X_labeled_tensor, y_labeled_tensor), batch_size=self.BATCH,
                                  shuffle=True)
        unlabeled_loader = DataLoader(TensorDataset(X_unlabeled_tensor, y_unlabeled_tensor), batch_size=self.BATCH,
                                      shuffle=False)
        dataloaders = {'train': train_loader}
        # Model
        mainnet = MainNet(X_labeled.shape[1]).to(self.device)
        lossnet = LossNet(X_unlabeled.shape[1]).to(self.device)
        models = {'backbone': mainnet, 'module': lossnet}

        criterion_train = nn.MSELoss(reduction='none')  # 逐个计算损失
        optim_backbone = optim.AdamW(models['backbone'].parameters(), lr=self.LR, weight_decay=self.WDECAY)
        optim_module = optim.AdamW(models['module'].parameters(), lr=self.LR, weight_decay=self.WDECAY)

        optimizers = {'backbone': optim_backbone, 'module': optim_module}

        # Training and test
        self.train(models, criterion_train, optimizers, dataloaders, self.EPOCH, self.EPOCHL)

        # Measure uncertainty of each data points in the subset
        uncertainty = self.get_uncertainty(models, unlabeled_loader, criterion_train)

        # 转换uncertainty为numpy数组
        uncertainty_numpy = uncertainty.numpy()
        # 寻找uncertainty前10大的数据点的索引

        incertitude_index = np.argsort(uncertainty_numpy)[-n_act:]
        selected_indices = [int(i) for i in X_unlabeled.index[incertitude_index]]
        return selected_indices
