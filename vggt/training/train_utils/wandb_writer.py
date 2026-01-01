# training/train_utils/wandb_writer.py

import os
import wandb
import torch
import numpy as np

class WandbLogger:
    def __init__(self, exp_name="vggt_experiment", project_name="vggt", entity=None, path=None):
        """
        Wandb Logger 초기화
        Args:
            exp_name: 실험 이름 (wandb run name)
            project_name: wandb 프로젝트 이름
            entity: wandb 사용자/팀 이름 (선택 사항)
            path: 로그 저장 경로 (tensorboard 호환성을 위해 받지만 wandb는 자동 관리됨)
        """
        self.run = wandb.init(
            project=project_name,
            name=exp_name,
            entity=entity,
            dir=path if path else "./logs",
            resume="allow"
        )
        print(f"[WandbLogger] Experiment '{exp_name}' initialized in project '{project_name}'")

    def log(self, key, value, step):
        """스칼라 값 로깅 (Loss, LR 등)"""
        # Tensor인 경우 스칼라로 변환
        if torch.is_tensor(value):
            value = value.item()
        
        # Wandb에 로그 전송
        wandb.log({key: value}, step=step)

    def log_visuals(self, key, visuals, step, fps=4):
        """
        이미지 또는 비디오 로깅
        Args:
            key: 로그 이름 (예: "Visuals/val")
            visuals: (B, C, H, W) 또는 (B, T, C, H, W) 형태의 numpy 배열 (값 범위 -1~1 또는 0~1)
        """
        # 값 범위를 0~255로 변환 (이미지 출력을 위해)
        if visuals.min() < 0:
            visuals = (visuals + 1.0) / 2.0  # -1~1 -> 0~1
        
        visuals = (visuals * 255).clip(0, 255).astype(np.uint8)

        # 비디오인 경우 (5차원: Batch, Time, Channel, Height, Width)
        if visuals.ndim == 5:
            # Wandb는 (Time, Channel, Height, Width) 순서를 선호하므로 배치 첫 번째만 로깅하거나 처리
            # 여기서는 배치의 첫 번째 샘플을 비디오로 저장한다고 가정
            video_sample = visuals[0] # (T, C, H, W)
            wandb.log({key: wandb.Video(video_sample, fps=fps, format="mp4")}, step=step)
        
        # 이미지인 경우 (Grid로 합쳐진 이미지 가정: Channel, Height, Width)
        elif visuals.ndim == 3:
            # (C, H, W) -> (H, W, C)로 변환
            img_sample = np.transpose(visuals, (1, 2, 0))
            wandb.log({key: [wandb.Image(img_sample)]}, step=step)
            
        else:
            print(f"[WandbLogger] Unsupported visual format: {visuals.shape}")

    def close(self):
        wandb.finish()