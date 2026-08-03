import hydra
from omegaconf import DictConfig
import lightning as L
import lightning.pytorch as pl
import torch
import wandb
from src import utils

utils.ignore_warnings()

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    
    if "seed_everything" in cfg:
        pl.seed_everything(cfg.seed_everything)
        
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="fit")

    tokenizer = hydra.utils.instantiate(
        cfg.tokenizer,
        num_numerical=datamodule.num_numerical,
        cat_cardinalities=datamodule.cat_cardinalities,
        bin_edges_dict = datamodule.bin_edges_dict
    )

    model = hydra.utils.instantiate(cfg.model, tokenizer = tokenizer)
    logger = hydra.utils.instantiate(cfg.logger)

    callbacks = [
        hydra.utils.instantiate(cb_conf)
        for cb_conf in cfg.callbacks.values()
    ]

    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    ckpt_path = cfg.get("ckpt_path")

    # 💡 [Warmup Restart 모드]: 가중치(Weight)만 이식 후 0 에폭부터 출발
    if cfg.model.use_restart:
        if not ckpt_path:
            raise ValueError("❌ 'use_restart'가 True일 때는 'ckpt_path'에 불러올 Baseline 체크포인트를 지정해야 합니다!")
        
        print(f"🔄 [Warmup Restart] Loading weights only from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")        
        auto_base_epoch = ckpt.get("epoch", -1) + 1

        model.base_epoch = auto_base_epoch
        model.load_state_dict(ckpt["state_dict"])

        
        
        trainer.fit(model, datamodule=datamodule)

    else:
        if ckpt_path:
            print(f"⏩ [Resume Training] Continuing training from: {ckpt_path}")
        else:
            print("🚀 [Fresh Start] Starting brand new training...")
            
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()