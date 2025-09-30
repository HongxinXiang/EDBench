from typing import Optional
import matplotlib.pyplot as plt
import os
import torch
import json
from pytorch_lightning import LightningModule
from torch.nn.functional import l1_loss, mse_loss, binary_cross_entropy 
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os
def load_representation(index, rep_dir):
    import json
    file_path = os.path.join(rep_dir, f"{index}.json")
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        return json.load(f)

def process_row(row, rep_dir, topks):
    
    try:
        idx = row['index']
        rep = load_representation(idx, rep_dir)
    except:
        return None

    if rep is None:
        return None

    try:
        neg_indices = list(map(int, str(row['negative_index']).split()))
    except:
        return None

    try:
        neg_reps = [load_representation(nidx, rep_dir) for nidx in neg_indices]
        neg_reps = [r for r in neg_reps if r is not None]
    except:
        return None

    if not neg_reps:
        return None

    g_f = np.array(rep['g_f']).reshape(1, -1)
    p_f = np.array(rep['p_f']).reshape(1, -1)

    result = {f"p_to_g_top{k}": 0 for k in topks}
    result.update({f"g_to_p_top{k}": 0 for k in topks})

    # p -> g
    neg_g_fs = np.array([r['g_f'] for r in neg_reps])
    all_g_fs = np.vstack([rep['g_f'], neg_g_fs])
    sims = cosine_similarity(p_f, all_g_fs).flatten()
    ranks = np.argsort(-sims)
    for k in topks:
        if 0 in ranks[:k]:
            result[f"p_to_g_top{k}"] += 1

    # g -> p
    neg_p_fs = np.array([r['p_f'] for r in neg_reps])
    all_p_fs = np.vstack([rep['p_f'], neg_p_fs])
    sims = cosine_similarity(g_f, all_p_fs).flatten()
    ranks = np.argsort(-sims)
    for k in topks:
        if 0 in ranks[:k]:
            result[f"g_to_p_top{k}"] += 1

    return result

def compute_topk_accuracy_lazy_parallel(csv_file, rep_dir, topks=[1, 3, 5], max_workers=8):
    df = pd.read_csv(csv_file)
    test_df = df[df['scaffold_split'] == 'test']

    result_dict = {f"p_to_g_top{k}": 0 for k in topks}
    result_dict.update({f"g_to_p_top{k}": 0 for k in topks})
    total = 0

    rows = [row for _, row in test_df.iterrows()]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_row, row, rep_dir, topks) for row in rows]

        for future in tqdm(futures):
            res = future.result()
            if res:
                total += 1
                for key in res:
                    result_dict[key] += res[key]

    if total > 0:
        for k in topks:
            result_dict[f"p_to_g_top{k}"] /= total
            result_dict[f"g_to_p_top{k}"] /= total

    print(total)
    return result_dict

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算平均绝对误差 (MAE)
    
    参数:
        y_true: 真实值数组，形状 (n_samples,)
        y_pred: 预测值数组，形状 (n_samples,)
    
    返回:
        MAE 值
    """
    return np.mean(np.abs(y_true - y_pred))

def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算均方误差 (MSE)
    
    参数:
        y_true: 真实值数组，形状 (n_samples,)
        y_pred: 预测值数组，形状 (n_samples,)
    
    返回:
        MSE 值
    """
    return np.mean((y_true - y_pred) ** 2)

from geoformer.model.modeling_contrastive_learning import create_model


class LNNP(LightningModule):
    def __init__(self, config) -> None:
        super(LNNP, self).__init__()

        self.save_hyperparameters(config)
        self.model = create_model(self.hparams)
        self.gpu_memory_per_step = []
        self.epoch_logs = []
        self.test_outputs = {}
        self._reset_losses_dict()
        self.rep_path = os.path.join(self.hparams.get("log_dir", "."), "rep")
        os.makedirs(self.rep_path, exist_ok=True)

    def configure_optimizers(self) -> Optional[AdamW]:
        optimizer = AdamW(
            params=self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.lr_scheduler == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.lr_cosine_length,
                eta_min=self.hparams.lr_min,
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        elif self.hparams.lr_scheduler == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                "min",
                factor=self.hparams.lr_factor,
                patience=self.hparams.lr_patience,
                min_lr=self.hparams.lr_min,
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        else:
            raise NotImplementedError(
                f"Unknown lr_schedule: {self.hparams.lr_scheduler}"
            )

        return [optimizer], [lr_scheduler]

    def forward(self, batch):
        return self.model(data = batch)

    # def training_step(self, batch, batch_idx):
    #     if self.hparams.loss_type == "MSE":
    #         return self.step(batch, mse_loss, "train")
    #     elif self.hparams.loss_type == "MAE":
    #         return self.step(batch, l1_loss, "train")
    #     else:
    #         NotImplementedError(f"Unknown loss type: {self.hparams.loss_type}")

    def training_step(self, batch, batch_idx):
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated() / 1024 ** 2  # 单位 MB
        self.gpu_memory_per_step.append(mem)
        return self.step(batch,'train')


    def validation_step(self, batch, batch_idx):
        return self.step(batch,'valid')


    # def test_step(self, batch, batch_idx):
    #     return self.step(batch, l1_loss, "test")
    def test_step(self, batch, batch_idx):
        pred, g_f, p_f = self(batch)

        results = []
        for i in range(len(batch["name"])):
            idx = batch["name"][i]
            file_path = os.path.join(self.rep_path, f"{idx}.json")
            with open(file_path, 'w') as f:
                json.dump({
                    'g_f': g_f[i].detach().cpu().numpy().tolist(),
                    'p_f': p_f[i].detach().cpu().numpy().tolist()
                }, f)
                
#         return loss


    def step(self, batch, stage):
        with torch.set_grad_enabled(stage == "train"):
            loss,_,_ = self(batch)
        
        self.losses[stage].append(loss.detach())
        return loss

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / float(self.hparams.lr_warmup_steps),
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            result_dict = {
                "epoch": float(self.current_epoch),
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                "train_loss": torch.stack(self.losses["train"]).mean(),
                "val_loss": torch.stack(self.losses["valid"]).mean(),
            }

            # add test loss if available
            if len(self.losses["test"]) > 0:
                result_dict["test_loss"] = torch.stack(
                    self.losses["test"]
                ).mean()

            self.log_dict(result_dict, prog_bar=True, sync_dist=True)

        # self._reset_losses_dict()

    # def on_test_epoch_end(self):
    #     result_dict = {}
    #     if len(self.losses["test"]) > 0:
    #         result_dict["test_loss"] = torch.stack(self.losses["test"]).mean()
    #     self.log_dict(result_dict, sync_dist=True)
    #     self._reset_losses_dict()

    def on_test_epoch_end(self):
        '''
        我有个csv文件，有index,smiles,canonical_smiles,mol_cluster,negative_index,scaffold_split,random_split。计算scaffold为’test‘行的指标，具体的，我们的数据格式
        我们设计了一个对比学习任务，有两种表征，g_f和p_f分别代表分子的图表征和点云表征，我们要做的是计算余弦相似度，分别g到p和从p到g,看能否将index从negative_index中排序到最前面，最后的指标为
        result_dict["p_to_g_top1"]
        result_dict["p_to_g_top3"]
        result_dict["p_to_g_top5"]
        result_dict["g_to_p_top1"]
        result_dict["g_to_p_top3"]
        result_dict["g_to_p_top5"]
        将index排序到topn的比例。
        现在已经计算出表征来了，需要你计算指标，数据格式是字典{index:{'g_f':[1,2],'p_f':[3,4]}}
        '''
            
        result_dict = compute_topk_accuracy_lazy_parallel('/code/retrieve/ed_retrievel_5w/raw/ed_retrievel_5w.csv', self.rep_path)
#         保存测试结果为 json 文件
#         if hasattr(self, "test_outputs"):
#             output_path = os.path.join(self.hparams.get("log_dir", "."), "test_results.json")
#             with open(output_path, "w") as f:
#                 json.dump(self.test_outputs, f, indent=4)
#             del self.test_outputs # 清空
        
        self.log_dict(result_dict, sync_dist=True)
        self._reset_losses_dict()

    def on_train_epoch_end(self):
        # 计算当前 epoch 的平均显存和 loss
        if self.gpu_memory_per_step:
            avg_gpu_mem = sum(self.gpu_memory_per_step) / len(self.gpu_memory_per_step)
        else:
            avg_gpu_mem = 0
        avg_train_loss = torch.stack(self.losses["train"]).mean().item()
        epoch_log = {
            "epoch": int(self.current_epoch),
            "avg_train_loss": avg_train_loss,
            "avg_gpu_memory_MB": avg_gpu_mem,
        }

        self.epoch_logs.append(epoch_log)

        # 写入到 json 文件中
        output_path = os.path.join(self.hparams.get("log_dir", "."), "epoch_stats.json") 
        with open(output_path, "w") as f:
            json.dump(self.epoch_logs, f, indent=4)

        # 清除显存记录，为下一个 epoch 做准备
        self.gpu_memory_per_step.clear()

    def _reset_losses_dict(self):
        self.losses = {
            "train": [],
            "valid": [],
            "test": [],
            "pred":[],
            "true":[]
        }

    def on_fit_end(self):
        if not self.epoch_logs:
            return  # 没有记录就跳过

        epochs = [item["epoch"] for item in self.epoch_logs]
        losses = [item["avg_train_loss"] for item in self.epoch_logs]
        memories = [item["avg_gpu_memory_MB"] for item in self.epoch_logs]

        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Avg Train Loss', color='tab:blue')
        ax1.plot(epochs, losses, label='Avg Train Loss', color='tab:blue', marker='o')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Avg GPU Memory (MB)', color='tab:red')
        ax2.plot(epochs, memories, label='Avg GPU Memory', color='tab:red', marker='x')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        plt.title("Training Loss and GPU Memory Usage per Epoch")
        fig.tight_layout()

        save_path = os.path.join(self.hparams.get("log_dir", "."), "epoch_plot.png")
        plt.savefig(save_path, dpi=300)
        print(f"[on_fit_end] Epoch training summary plot saved to: {save_path}")