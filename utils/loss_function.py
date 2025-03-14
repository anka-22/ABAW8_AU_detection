import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score

class CCCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gold, pred, weights=None):

        gold_mean = torch.mean(gold, 1, keepdim=True, out=None)
        pred_mean = torch.mean(pred, 1, keepdim=True, out=None)
        covariance = (gold - gold_mean) * (pred - pred_mean)
        gold_var = torch.var(gold, 1, keepdim=True, unbiased=True, out=None)
        pred_var = torch.var(pred, 1, keepdim=True, unbiased=True, out=None)
        ccc = 2. * covariance / (
                (gold_var + pred_var + torch.mul(gold_mean - pred_mean, gold_mean - pred_mean)) + 1e-50)
        ccc_loss = 1. - ccc

        if weights is not None:
            ccc_loss *= weights

        return torch.mean(ccc_loss)

class F1_Score(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, y_true, y_pred):
        """计算Soft F1 Score作为损失函数。
        Args:
            y_true: 真实标签，维度为(batch_size, num_classes)，One-hot 编码格式。
            y_pred: 预测得分或概率，维度为(batch_size, num_classes)，通过softmax函数处理。
        Returns:
            loss: Soft F1 Score损失。
        """
        tp = torch.sum(y_true * y_pred, axis=0)
        fp = torch.sum((1 - y_true) * y_pred, axis=0)
        fn = torch.sum(y_true * (1 - y_pred), axis=0)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_loss = 1 - f1.mean()

        return f1_loss

class compute_AU_loss_BCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weighs = [5.278187667237907, 12.42675934708439, 3.9182460964352357, 2.3713836486482838, 1.5735074526534674, 1.8198100142634426, 2.586578383618721, 22.66022149366239, 20.121237822349574, 22.104986149584487, 1.0, 8.264810560409762]

    def forward(self, pred, label):
        weights = torch.tensor(self.weighs).cuda()
        bz,seq,_  = pred.shape
        label = label.view(bz*seq,-1)
        pred = pred.view(bz*seq,-1)

        cri_AU = nn.BCEWithLogitsLoss(weights)
        bz,c = pred.shape
        cls_loss = cri_AU(pred, label)
        
        AU_pred = nn.Sigmoid()(pred)

        return cls_loss, AU_pred, label

def compute_AU_F1(pred,label):
    pred = np.array(pred.detach().cpu())
    label = np.array(label.detach().cpu())
    AU_targets = [[] for i in range(12)]
    AU_preds = [[] for i in range(12)]
    F1s = []
    for i in range(pred.shape[0]):
        for j in range(12):
            p = pred[i,j]
            if p>=0.5:
                AU_preds[j].append(1)
            else:
                AU_preds[j].append(0)
            AU_targets[j].append(label[i,j])
    
    for i in range(12):
        F1s.append(f1_score(AU_targets[i], AU_preds[i], average='binary'))

    F1s = np.array(F1s)
    F1_mean = np.mean(F1s)
    return F1s, F1_mean