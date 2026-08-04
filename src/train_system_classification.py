import lightning as L
import numpy as np
import torch
import torch.nn as nn
# 💡 분류용 추가 메트릭 임포트
from torchmetrics.classification import (
    BinaryAccuracy, 
    BinaryAUROC, 
    BinaryF1Score, 
    BinaryRecall, 
    BinaryPrecision
)
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, SequentialLR, LinearLR
from typing import Any

class LightningFTTClassifier(L.LightningModule):
    def __init__(
        self, 
        net: nn.Module, 
        tokenizer: nn.Module,  
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
        self.save_hyperparameters(ignore=["net", "tokenizer", "criterion"])

        self.hparams.update({
            "num_self_attn_layers": getattr(net, "num_self_attn_layers", None),
            "d_model": getattr(net, "d_model", None),
            "dropout_p": getattr(net, "dropout_p", None),
        })
        self.save_hyperparameters(self.hparams)

        self.net = net
        self.tokenizer = tokenizer  
        self.criterion = criterion

        # 💡 [Train] 메트릭 객체 선언
        self.train_acc = BinaryAccuracy()
        
        # 💡 [Validation] 메트릭 객체 선언
        self.val_acc = BinaryAccuracy()
        self.val_auroc = BinaryAUROC()
        self.val_f1 = BinaryF1Score()
        self.val_recall = BinaryRecall()
        self.val_precision = BinaryPrecision()
        
        # 💡 [Test] 메트릭 객체 선언
        self.test_acc = BinaryAccuracy()
        self.test_auroc = BinaryAUROC()
        self.test_f1 = BinaryF1Score()
        self.test_recall = BinaryRecall()
        self.test_precision = BinaryPrecision()

        self.trainer : Any
        self.logger : Any
        self.hparams : Any

    def forward(self, x_num, x_cat):
        return self.net(x_num, x_cat, self.tokenizer)

    def on_fit_start(self):
        """Wandb 또는 TensorBoard 계열 로거 메트릭 타겟 설정"""
        if hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "define_metric"):
            self.logger.experiment.define_metric("train/acc", summary="max") 
            self.logger.experiment.define_metric("val/loss", summary="min")            
            self.logger.experiment.define_metric("val/acc", summary="max")
            self.logger.experiment.define_metric("val/auroc", summary="max")
            self.logger.experiment.define_metric("val/f1", summary="max")
   
    def training_step(self, batch, batch_idx):
        x_num, x_cat, y = batch
        outputs = self(x_num, x_cat)
        
        if outputs.shape != y.shape:
            y = y.view_as(outputs)
            
        loss = self.criterion(outputs, y.float()) 
        
        self.train_acc(outputs.detach(), y.detach())

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_num, x_cat, y = batch
        outputs = self(x_num, x_cat)
        
        if outputs.shape != y.shape:
            y = y.view_as(outputs)
            
        loss = self.criterion(outputs, y.float())

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # 💡 밸리데이션 메트릭 업데이트
        self.val_acc.update(outputs, y)
        self.val_auroc.update(outputs, y)
        self.val_f1.update(outputs, y)
        self.val_recall.update(outputs, y)
        self.val_precision.update(outputs, y)
        return loss

    def test_step(self, batch, batch_idx):
        x_num, x_cat, y = batch
        outputs = self(x_num, x_cat)
        
        if outputs.shape != y.shape:
            y = y.view_as(outputs)
            
        loss = self.criterion(outputs, y.float())

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        
        # 💡 테스트 메트릭 업데이트
        self.test_acc.update(outputs, y)
        self.test_auroc.update(outputs, y)
        self.test_f1.update(outputs, y)
        self.test_recall.update(outputs, y)
        self.test_precision.update(outputs, y)
        return loss

    def on_validation_epoch_end(self):
        epoch_val_acc = self.val_acc.compute()
        epoch_val_auroc = self.val_auroc.compute()
        epoch_val_f1 = self.val_f1.compute()
        epoch_val_recall = self.val_recall.compute()
        epoch_val_precision = self.val_precision.compute()
        
        val_loss_val = self.trainer.callback_metrics.get("val/loss", torch.tensor(0.0)).item()

        # 💡 신규 메트릭 로깅 추가
        self.log("val/acc", epoch_val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auroc", epoch_val_auroc, on_step=False, on_epoch=True)
        self.log("val/f1", epoch_val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", epoch_val_recall, on_step=False, on_epoch=True)
        self.log("val/precision", epoch_val_precision, on_step=False, on_epoch=True)

        # 💡 가독성을 위해 터미널 출력문 확장
        self.print(
            f"🌟 [Epoch {self.current_epoch+1}] Val Loss: {val_loss_val:.4f} | "
            f"Acc: {epoch_val_acc:.4f} | AUROC: {epoch_val_auroc:.4f} | "
            f"F1: {epoch_val_f1:.4f} | Rec: {epoch_val_recall:.4f} | Prec: {epoch_val_precision:.4f}"
        )

        # 💡 메트릭 초기화
        self.val_acc.reset()
        self.val_auroc.reset()
        self.val_f1.reset()
        self.val_recall.reset()
        self.val_precision.reset()

        global_epoch = self.hparams.base_epoch + self.current_epoch
        self.log("global_epoch", float(global_epoch), sync_dist=True)

    def on_test_epoch_end(self):
        epoch_test_acc = self.test_acc.compute()
        epoch_test_auroc = self.test_auroc.compute()
        epoch_test_f1 = self.test_f1.compute()
        epoch_test_recall = self.test_recall.compute()
        epoch_test_precision = self.test_precision.compute()
        
        test_loss_val = self.trainer.callback_metrics.get("test/loss", torch.tensor(0.0)).item()

        # 💡 테스트 최종결과 로깅 추가
        self.log("test/Accuracy", epoch_test_acc)
        self.log("test/AUROC", epoch_test_auroc)
        self.log("test/F1_Score", epoch_test_f1)
        self.log("test/Recall", epoch_test_recall)
        self.log("test/Precision", epoch_test_precision)

        self.print(
            f"\n🚀 [FINAL TEST RESULT] Loss: {test_loss_val:.4f}\n"
            f"👉 Accuracy:  {epoch_test_acc:.4f} | AUROC: {epoch_test_auroc:.4f}\n"
            f"👉 F1-Score:  {epoch_test_f1:.4f} | Recall: {epoch_test_recall:.4f} | Precision: {epoch_test_precision:.4f}\n"
        )

        self.test_acc.reset()
        self.test_auroc.reset()
        self.test_f1.reset()
        self.test_recall.reset()
        self.test_precision.reset()
            
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.base_lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.use_restart:
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.hparams.restart_period, T_mult=self.hparams.t_mult, eta_min=self.hparams.eta_min)
        else:
            warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=self.hparams.warmup_epochs)
            main_scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.total_epochs - self.hparams.warmup_epochs, eta_min=self.hparams.eta_min)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[self.hparams.warmup_epochs])

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
