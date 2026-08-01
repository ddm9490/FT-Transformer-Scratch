#!/bin/sh
python train.py \
  callbacks=restart \
  ckpt_path="checkpoints/forwarmup-v1.ckpt" \
  model.use_restart=True \
  trainer.max_epochs=150