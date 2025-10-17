import time
import torch
import torch.nn as nn
from metrics.abstract_metrics import CrossEntropyMetric
from torchmetrics import Metric, MeanSquaredError

# 118 个元素的参考表（可保留）
weight_check_full = torch.tensor(
    [4.003, 6.941, 9.012, 10.812, 12.011, 14.007, 15.999, 18.998, 20.18, 22.99, 24.305, 26.982, 28.086, 30.974, 32.067, 35.453, 39.948, 39.098, 40.078, 44.956, 47.867, 50.942, 51.996, 54.938, 55.845, 58.933, 58.693, 63.546, 65.39, 69.723, 72.61, 74.922, 78.96, 79.904, 83.8, 85.468, 87.62, 88.906, 91.224, 92.906, 95.94, 98.0, 101.07, 102.906, 106.42, 107.868, 112.412, 114.818, 118.711, 121.76, 127.6, 126.904, 131.29, 132.905, 137.328, 138.906, 140.116, 140.908, 144.24, 145.0, 150.36, 151.964, 157.25, 158.925, 162.5, 164.93, 167.26, 168.934, 173.04, 174.967, 178.49, 180.948, 183.84, 186.207, 190.23, 192.217, 195.078, 196.967, 200.59, 204.383, 207.2, 208.98, 209.0, 210.0, 222.0, 223.0, 226.0, 227.0, 232.038, 231.036, 238.029, 237.0, 244.0, 243.0, 247.0, 247.0, 251.0, 252.0, 257.0, 258.0, 259.0, 262.0, 267.0, 268.0, 269.0, 270.0, 269.0, 278.0, 281.0, 281.0, 285.0, 284.0, 289.0, 288.0, 293.0, 292.0, 294.0, 294.0],
    dtype=torch.float,
)

class AtomWeightMetric(Metric):
    """
    日志用：|sum(MW_pred) - sum(MW_true)| 的累计 MAE（argmax，不反传）
    这里会根据当前 batch 的 dx 对齐到 weight_vec_dx
    """
    def __init__(self):
        super().__init__()
        self.add_state('total_loss', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.register_buffer('weight_full', weight_check_full.clone())
        self.register_buffer('weight_vec_dx', torch.tensor([], dtype=torch.float))  # 动态缓存对齐后的向量

    def _ensure_weight_vec(self, dx, device, dtype, class_to_Z=None):
        if self.weight_vec_dx.numel() == dx:
            return self.weight_vec_dx.to(device=device, dtype=dtype)
        if class_to_Z is not None:
            # class_to_Z: 长度 dx 的 long 索引（映射到 weight_full）
            w = self.weight_full[class_to_Z.to(self.weight_full.device)]
        elif self.weight_full.numel() == dx:
            w = self.weight_full
        else:
            # 兜底：取前 dx 项（建议你提供正确的映射以替换）
            w = self.weight_full[:dx]
        self.weight_vec_dx = w.detach().clone()
        return self.weight_vec_dx.to(device=device, dtype=dtype)

    @torch.no_grad()
    def update(self, X, Y, node_mask=None, class_to_Z=None):
        # X, Y: (bs, n, dx)
        bs, n, dx = X.shape
        w = self._ensure_weight_vec(dx, X.device, X.dtype, class_to_Z=class_to_Z)

        atom_pred_num = X.argmax(dim=-1)        # (bs, n)
        atom_real_num = Y.argmax(dim=-1)        # (bs, n)

        pred_weight = w[atom_pred_num]          # (bs, n)
        real_weight = w[atom_real_num]          # (bs, n)

        if node_mask is not None:
            m = node_mask.squeeze(-1).to(pred_weight.dtype)
            pred_weight = pred_weight * m
            real_weight = real_weight * m

        lss = torch.abs(pred_weight.sum(dim=-1) - real_weight.sum(dim=-1)).sum()
        self.total_loss += lss
        self.total_samples += bs

    def compute(self):
        return self.total_loss / self.total_samples.clamp_min(1)


class TrainLossDiscrete(nn.Module):
    """ Train with Cross entropy + 可导的原子量正则 """
    def __init__(self, lambda_train, weight_node=None, weight_edge=None, class_to_Z: torch.Tensor | None = None):
        """
        class_to_Z: (dx,) 的 long 张量；将你的类别索引 -> 元素索引（用于从 118 表里抽取）
        若不提供，会自动对齐（weight_full[:dx]），能跑但可能不准确。
        """
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.weight_loss = AtomWeightMetric()      # 日志

        self.y_loss = MeanSquaredError()
        self.lambda_train = lambda_train

        self.register_buffer("weight_full", weight_check_full.clone())
        if class_to_Z is not None:
            assert class_to_Z.dim() == 1, "class_to_Z must be 1-D"
            self.register_buffer("class_to_Z", class_to_Z.clone().long())
        else:
            self.class_to_Z = None  # 兜底模式

        # 训练时缓存对齐后的权重（长度 dx）
        self.register_buffer('weight_vec_dx', torch.tensor([], dtype=torch.float))

    def _get_weight_vec(self, dx, device, dtype):
        if self.weight_vec_dx.numel() == dx:
            return self.weight_vec_dx.to(device=device, dtype=dtype)
        if self.class_to_Z is not None:
            w = self.weight_full[self.class_to_Z.to(self.weight_full.device)]
        elif self.weight_full.numel() == dx:
            w = self.weight_full
        else:
            # 兜底：取前 dx 项（建议之后提供 class_to_Z 以确保语义正确）
            w = self.weight_full[:dx]
        self.weight_vec_dx = w.detach().clone()
        return self.weight_vec_dx.to(device=device, dtype=dtype)

    def _weight_regularizer(self, pred_logits, true_onehot, node_mask=None):
        """
        可导：使用 softmax 概率的期望原子量
        pred_logits, true_onehot: (bs, n, dx)
        """
        p = pred_logits.softmax(dim=-1)  # (bs, n, dx)
        bs, n, dx = p.shape
        w = self._get_weight_vec(dx, p.device, p.dtype)          # (dx,)

        mw_pred = (p * w.view(1, 1, -1)).sum(dim=-1)             # (bs, n)
        mw_true = (true_onehot * w.view(1, 1, -1)).sum(dim=-1)   # (bs, n)

        if node_mask is not None:
            m = node_mask.squeeze(-1).to(mw_pred.dtype)          # (bs, n)
            mw_pred = mw_pred * m
            mw_true = mw_true * m

        per_graph_abs = torch.abs(mw_pred.sum(dim=1) - mw_true.sum(dim=1))  # (bs,)
        return per_graph_abs.mean()

    def forward(self, masked_pred_X, masked_pred_E, pred_y, true_X, true_E, true_y, node_mask, log: bool):
        """
        masked_pred_X : (bs, n, dx)
        masked_pred_E : (bs, n, n, de)
        pred_y        : (bs, )
        true_X        : (bs, n, dx)
        true_E        : (bs, n, n, de)
        true_y        : (bs, )
        """
        # 可导的原子量正则
        loss_weight = self._weight_regularizer(masked_pred_X, true_X, node_mask=node_mask)

        # 日志（与 forward 的 dx 对齐）
        self.weight_loss.update(masked_pred_X.detach(), true_X.detach(), node_mask=node_mask, class_to_Z=self.class_to_Z)

        # ===== 下面保持你原始展平与 CE 计算的逻辑 =====
        mask_X_valid = (true_X != 0.).any(dim=-1)
        if node_mask is not None:
            mask_X_valid = mask_X_valid & node_mask.squeeze(-1).bool()

        flat_true_X = true_X[mask_X_valid, :]
        flat_pred_X = masked_pred_X[mask_X_valid, :]

        true_E_flat = true_E.reshape(-1, true_E.size(-1))
        masked_pred_E_flat = masked_pred_E.reshape(-1, masked_pred_E.size(-1))
        mask_E_valid = (true_E_flat != 0.).any(dim=-1)

        flat_true_E = true_E_flat[mask_E_valid, :]
        flat_pred_E = masked_pred_E_flat[mask_E_valid, :]

        loss_X = self.node_loss(flat_pred_X, flat_true_X) if flat_true_X.numel() > 0 else 0.0
        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if flat_true_E.numel() > 0 else 0.0

        return self.lambda_train[0] * loss_X + self.lambda_train[1] * loss_E + loss_weight

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss, self.weight_loss]:
            metric.reset()

    def log_epoch_metrics(self, current_epoch, start_epoch_time, log=True):
        epoch_node_loss = self.node_loss.compute() if getattr(self.node_loss, "total_samples", 0) > 0 else -1
        epoch_edge_loss = self.edge_loss.compute() if getattr(self.edge_loss, "total_samples", 0) > 0 else -1
        epoch_weight_loss = self.weight_loss.compute() if self.weight_loss.total_samples > 0 else -1

        print(f"Epoch {current_epoch} finished: X_CE: {epoch_node_loss :.4f} -- E_CE: {epoch_edge_loss :.4f} "
              f"Weight: {epoch_weight_loss :.4f} "
              f"-- Time taken {time.time() - start_epoch_time:.1f}s ")
