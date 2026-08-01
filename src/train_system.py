import lightning as L
import numpy as np
import torch
import torch.nn as nn
from torchmetrics.regression import R2Score, MeanSquaredError
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, SequentialLR, LinearLR
from typing import Any

class LightningFTTSystem(L.LightningModule):
    def __init__(
        self, 
        net: nn.Module, 
        tokenizer: nn.Module,  # 💡 하이드라 설정 파일에서 인스턴스화하여 외부 주입받도록 변경
        criterion: nn.Module, 
        base_lr: float = 1e-3,
        eta_min: float = 1e-6,
        weight_decay: float = 0.01,
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        base_epoch: int = 0,
        use_restart: bool = False,
        restart_period: int = 20,
        t_mult: int = 1,
    ):
        super().__init__()
        # 외부 인스턴스 객체들을 무시하고 하이퍼파라미터 일괄 저장
        self.save_hyperparameters(ignore=["net", "tokenizer", "criterion"])

        # 하이드라 혹은 모델 가독성 매핑 유지
        self.hparams.update({
            "num_self_attn_layers": getattr(net, "num_self_attn_layers", None),
            "d_model": getattr(net, "d_model", None),
            "dropout_p": getattr(net, "dropout_p", None),
        })

        self.net = net
        self.tokenizer = tokenizer  # 💡 자식 모듈로 등록됨으로써 디바이스 이동 자동 처리
        self.criterion = criterion

        # 메트릭 객체 선언 (의도를 직관적으로 보이게 네이밍 정리)
        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()
        
        self.val_rmse = MeanSquaredError(squared=False)
        self.test_rmse = MeanSquaredError(squared=False)

        self.trainer : Any
        self.logger : Any
        self.hparams : Any

    def forward(self, x_num, x_cat):
        return self.net(x_num, x_cat, self.tokenizer)

    def _inverse_transform_y(self, y_tensor: torch.Tensor) -> torch.Tensor:
        """💡 데이터모듈의 스케일러와 로그를 감지하여 원래 단위로 복원하는 헬퍼 함수"""
        dm = self.trainer.datamodule 
        
        # 1. 넘파이 변환 (사이킷런 스케일러 대응)
        y_np = y_tensor.detach().cpu().numpy()
        
        # 2. 데이터 모듈에 y_scaler가 존재하면 역변환 수행
        if hasattr(dm, "y_scaler") and dm.y_scaler is not None:
            y_np = dm.y_scaler.inverse_transform(y_np)
            
        y_original = y_np
        
        # 4. 다시 현재 디바이스의 텐서로 복원
        return torch.from_numpy(y_original).to(self.device, dtype=torch.float32)

    def on_fit_start(self):
        """Wandb 또는 TensorBoard 계열 로거 메트릭 타겟 설정"""
        if hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "define_metric"):
            self.logger.experiment.define_metric("train/r2", summary="max") 
            self.logger.experiment.define_metric("val/loss", summary="min")            
            self.logger.experiment.define_metric("val/r2", summary="max")
            self.logger.experiment.define_metric("val/RMSE", summary="min")
   
    def training_step(self, batch, batch_idx):
        x_num, x_cat, y = batch
        outputs = self(x_num, x_cat)
        loss = self.criterion(outputs, y)
        
        self.train_r2(outputs.detach(), y.detach())

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/r2", self.train_r2, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_num, x_cat, y = batch
        outputs = self(x_num, x_cat)
        loss = self.criterion(outputs, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_r2.update(outputs, y)
        self.val_rmse.update(outputs, y)
        return loss

    def test_step(self, batch, batch_idx):
        x_num, x_cat, y = batch
        outputs = self(x_num, x_cat)
        loss = self.criterion(outputs, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.test_r2.update(outputs, y)
        self.test_rmse.update(outputs, y)
        return loss

    def on_validation_epoch_end(self):
        epoch_val_r2 = self.val_r2.compute()
        epoch_val_rmse = self.val_rmse.compute()
        val_loss_val = self.trainer.callback_metrics.get("val/loss", torch.tensor(0.0)).item()

        self.log("val/RMSE", epoch_val_rmse, on_step=False, on_epoch=True)
        self.log("val/r2", epoch_val_r2, on_step=False, on_epoch=True, prog_bar=True)

        self.print(f"🌟 [Epoch {self.current_epoch+1}] Val Loss: {val_loss_val:.4f} | Real RMSE: {epoch_val_rmse:.4f} | R² Score: {epoch_val_r2:.4f}")

        self.val_r2.reset()
        self.val_rmse.reset()

        global_epoch = self.hparams.base_epoch + self.current_epoch
        self.log("global_epoch", float(global_epoch), sync_dist=True)

    def on_test_epoch_end(self):
        epoch_test_r2 = self.test_r2.compute()
        epoch_test_rmse = self.test_rmse.compute()
        test_loss_val = self.trainer.callback_metrics.get("test/loss", torch.tensor(0.0)).item()

        self.log("test/RMSE_original", epoch_test_rmse)
        self.log("test/R2_Score", epoch_test_r2)

        self.print(f"\n🚀 [FINAL TEST RESULT] Loss: {test_loss_val:.4f} | Real RMSE: {epoch_test_rmse:.4f} | R² Score: {epoch_test_r2:.4f}\n")

        self.test_r2.reset()
        self.test_rmse.reset()
            
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.base_lr, weight_decay=self.hparams.weight_decay)
        
        if self.hparams.use_restart:
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=self.hparams.restart_period, T_mult=self.hparams.t_mult, eta_min=self.hparams.eta_min
            )
        else:
            warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=self.hparams.warmup_epochs)
            main_scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.total_epochs - self.hparams.warmup_epochs, eta_min=self.hparams.eta_min)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[self.hparams.warmup_epochs])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }
