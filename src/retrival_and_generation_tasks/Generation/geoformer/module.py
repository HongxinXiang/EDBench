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
from scipy.stats import pearsonr, spearmanr
import pickle
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

from geoformer.model.modeling_geoformer import create_model


class LNNP(LightningModule):
    def __init__(self, config) -> None:
        super(LNNP, self).__init__()

        self.save_hyperparameters(config)
        self.model = create_model(self.hparams)
        self.gpu_memory_per_step = []
        self.epoch_logs = []
        self.out_dir = os.path.join(self.hparams.get("log_dir", "."), "predict")
        os.makedirs(self.out_dir, exist_ok=True)
        self._reset_losses_dict()

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

        if self.hparams.loss_type == "MSE":
            return self.step(batch, mse_loss, "train")
        elif self.hparams.loss_type == "MAE":
            return  self.step(batch, l1_loss, "train")
        elif self.hparams.loss_type == "BCE":
            return  self.step(batch, binary_cross_entropy, "train")
        else:
            raise NotImplementedError(f"Unknown loss type: {self.hparams.loss_type}")


    def validation_step(self, batch, batch_idx):
        if self.hparams.loss_type == "MSE":
            return self.step(batch, mse_loss, "val")
        elif self.hparams.loss_type == "MAE":
            return self.step(batch, l1_loss, "val")
        elif self.hparams.loss_type == "BCE":
            return self.step(batch, binary_cross_entropy, "val")
        else:
            raise NotImplementedError(f"Unknown loss type: {self.hparams.loss_type}")


    # def test_step(self, batch, batch_idx):
    #     return self.step(batch, l1_loss, "test")
    def test_step(self, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start = time.perf_counter()
        
        pred, mask = self(batch)
        
        if torch.cuda.is_available():
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end) / 1000  # 转换为秒
        else:
            elapsed = time.perf_counter() - start

        if batch["labels"].ndim == 1:
            batch["labels"][mask] = batch["labels"][mask]
        if self.hparams.loss_type == "MSE":
            loss = mse_loss(pred.squeeze(1), batch["labels"])
        elif self.hparams.loss_type == "MAE":
            loss = l1_loss(pred, batch["labels"])
        elif self.hparams.loss_type == "BCE":
            loss = binary_cross_entropy(torch.sigmoid(pred), batch["labels"].float())
        else:
            raise NotImplementedError(f"Unknown loss type: {self.hparams.loss_type}")

        self.losses["test"].append(loss.detach())

        # 收集结果
        if self.hparams.encoder_name != 'EGNN':
            results = []
            for i in range(len(batch["name"])):
                if self.hparams.loss_type == "BCE":
                    results.append({
                        "name": batch["name"][i],
                        "pred": torch.sigmoid(pred[i]).detach().cpu().numpy().tolist(),
                        "label": batch["labels"][i].detach().cpu().numpy().tolist()
                        })
                else:
                    results.append({
                        "name": batch["name"][i],
                        "pred": pred[i].detach().cpu().numpy().tolist()[mask],
                        "label": batch["labels"][i].detach().cpu().numpy().tolist()[mask]
                        })

        
            # 保存为属性，后续统一写入文件
            if not hasattr(self, "test_outputs"):
                self.test_outputs = []
            self.test_outputs.extend(results)
        else:
            self.losses["pred"].extend(list(np.array(pred[mask].squeeze().detach().cpu().numpy())))
            self.losses["true"].extend(list(np.array(batch["labels"][mask].detach().cpu().numpy())))
            self.losses["name"].extend([batch['graph'].name[i] for i in batch['graph'].batch])
            self.losses["pos"].extend(list(np.array(batch["graph"].pos[mask].squeeze().detach().cpu().numpy())))
            self.losses["time"].append(elapsed)
            step_data = {
                "pred": np.array(pred[mask].squeeze().detach().cpu().numpy()),
                "name": [batch['graph'].name[i] for i in batch['graph'].batch],
                "pos": np.array(batch["graph"].pos[mask].squeeze().detach().cpu().numpy()),
            }

            # 保存到文件
            save_path = os.path.join(self.out_dir, f"step_{batch_idx}.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(step_data, f)
            
            
        return loss


    def step(self, batch, loss_fn, stage):
        with torch.set_grad_enabled(stage == "train"):
            pred,mask = self(batch)

        loss = 0

        if "labels" in batch:
            if batch["labels"].ndim == 1:
                batch["labels"] = batch["labels"].unsqueeze(1)
            if self.hparams.loss_type == "MSE":
                loss = loss_fn(pred[mask], batch["labels"][mask])
            elif self.hparams.loss_type == "MAE":
                loss = loss_fn(pred, batch["labels"])
            elif self.hparams.loss_type == "BCE":
                loss = loss_fn(torch.sigmoid(pred),batch["labels"].float())
            else:
                raise NotImplementedError(f"Unknown loss type: {self.hparams.loss_type}")
        
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
                "val_loss": torch.stack(self.losses["val"]).mean(),
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
        result_dict = {}
        if len(self.losses["test"]) > 0:
            result_dict["test_loss"] = torch.stack(self.losses["test"]).mean()

        # 保存测试结果为 json 文件
        if self.hparams.encoder_name != 'EGNN':
            if hasattr(self, "test_outputs"):
                output_path = os.path.join(self.hparams.get("log_dir", "."), "test_results.json")
                with open(output_path, "w") as f:
                    json.dump(self.test_outputs, f, indent=4)
                del self.test_outputs # 清空
        else:
            self.losses["pred"] = np.array(self.losses["pred"])
            self.losses["true"] = np.array(self.losses["true"])
            result_dict["mse"] = calculate_mse(self.losses["pred"],self.losses["true"])
            result_dict["mae"] = calculate_mae(self.losses["pred"],self.losses["true"])
            result_dict["rmse"] = np.sqrt(result_dict["mse"])
            result_dict["pearson"], _ = pearsonr(self.losses["pred"], self.losses["true"])
            result_dict["spearman"], _ = spearmanr(self.losses["pred"], self.losses["true"])
            result_dict["time"] = sum(self.losses["time"]) / len(self.losses["time"])
            save_path = os.path.join(self.hparams.get("log_dir", "."), "prediction.pkl")
            predict = {}
            predict['name'] = list(self.losses["name"])
            predict['pred'] = list(self.losses["pred"])
            predict['true'] = list(self.losses["true"])
#             predict['pos'] = list(self.losses["pos"])
            with open(save_path, "wb") as f:
                pickle.dump(predict, f)
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
            "val": [],
            "test": [],
            "pred":[],
            "true":[],
            "name":[],
            "time":[],
            "pos": []
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