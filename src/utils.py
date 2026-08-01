import random,os,torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box
import warnings

def seed_everything(seed=42):
    # 1. 파이썬 기본 라이브러리 시드 고정
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 2. 넘파이 시드 고정
    np.random.seed(seed)

    # 3. 파이토치 및 CUDA(GPU) 시드 고정
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 멀티 GPU를 쓸 때 사용

def ignore_warnings():
    warnings.filterwarnings('ignore')

def print_hparams_pretty(hparams_dict, title="Hyperparameters"):
    console = Console()
    
    # 표 생성 (Lightning/Ray Tune 스타일의 두꺼운/라운드 테두리)
    table = Table(title=f"=== {title} ===", box=box.HEAVY_HEAD, header_style="bold magenta")
    
    # 컬럼 정의
    table.add_column("Hyperparameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    # 딕셔너리 순회하며 행 추가 (재귀적으로 nested dict도 깔끔하게 풀어냄)
    def add_rows(d, prefix=""):
        for k, v in d.items():
            key_str = f"{prefix}{k}"
            if isinstance(v, dict):
                add_rows(v, prefix=f"{key_str}.")
            else:
                table.add_row(key_str, str(v))

    add_rows(hparams_dict)
    
    # 출력
    console.print(table)

seed_everything(42)