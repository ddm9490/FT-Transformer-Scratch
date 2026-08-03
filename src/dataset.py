import pandas as pd
import numpy as np
import json
import lightning as L

from torch.utils.data import TensorDataset, DataLoader
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, OrdinalEncoder, PowerTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from typing import Any

class ScalableTabularDataModule(L.LightningDataModule):
    def __init__(self, dataset_dir, target_names, out_features, batch_size=1024, num_workers=3, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_dir = dataset_dir
        self.target_names = target_names
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.out_features = out_features

        # These will be dynamically populated during setup()
        self.num_numerical = 0
        self.cat_cardinalities = []

        self.bin_edges_dict = None
        self.n_bins = kwargs.get("n_bins", None)

    def setup(self, stage=None):
        # 1. 원본 데이터 로드 및 Feature/Target 분리
        df : Any = pd.read_csv(self.dataset_dir)

        
        feature_df = df.drop(columns=self.target_names)

        self.num_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
        self.num_numerical = len(self.num_cols)
        self.has_numerical = len(self.num_cols) > 0
        self.has_categorical = len(self.cat_cols) > 0

        # 2. 안전한 데이터 분할 (Data Leakage 방지)
        df_temp, df_test = train_test_split(df, test_size=0.2, random_state=42)
        df_train, df_val = train_test_split(df_temp, test_size=0.2, random_state=42)

        y_train= df_train[self.target_names].to_numpy().reshape(-1, 1)
        y_val= df_val[self.target_names].to_numpy().reshape(-1, 1)
        y_test= df_test[self.target_names].to_numpy().reshape(-1, 1)
        
        y_scaler = PowerTransformer()
        self.y_scaler = y_scaler

        y_train_scaled = y_scaler.fit_transform(y_train)
        y_val_scaled = y_scaler.transform(y_val)
        y_test_scaled = y_scaler.transform(y_test)
            
        
        # 4. 수치형 데이터 전처리 (결측치 채우기 및 분위수 변환)
        if self.has_numerical:
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')), # 결측치 중앙값 자동 채우기
                ('scaler', QuantileTransformer(n_quantiles=100, output_distribution="normal", random_state=42))                    # 스케일링 자동 적용
            ]) 
            
            X_train_num = num_pipeline.fit_transform(df_train[self.num_cols])
            X_val_num = num_pipeline.transform(df_val[self.num_cols])
            X_test_num = num_pipeline.transform(df_test[self.num_cols])
            
            if self.n_bins is not None:
                X_train_num_imp = pd.DataFrame(
                    X_train_num,
                    columns=self.num_cols
                )
                
                self.bin_edges_dict = compute_bin_edges(X_train_num_imp, self.num_cols, n_bins=self.n_bins)

            X_train_num_tensor = torch.tensor(X_train_num, dtype = torch.float32)
            X_val_num_tensor = torch.tensor(X_val_num, dtype = torch.float32)
            X_test_num_tensor = torch.tensor(X_test_num, dtype = torch.float32)
        else:
            X_train_num_tensor = torch.empty((len(y_train), 0), dtype=torch.long)
            X_val_num_tensor   = torch.empty((len(y_test), 0), dtype=torch.long)
            X_test_num_tensor  = torch.empty((len(y_val), 0), dtype=torch.long)

        if self.has_categorical:
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', OrdinalEncoder(
                    dtype=np.int64, # type:ignore
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                ))
            ])
             # 5. 범주형 데이터 전처리 (미지 카테고리 에러 방지 옵션 추가)
    
            X_train_cat = cat_pipeline.fit_transform(df_train[self.cat_cols])
            X_val_cat = cat_pipeline.transform(df_val[self.cat_cols])
            X_test_cat = cat_pipeline.transform(df_test[self.cat_cols])

            X_train_cat = np.where(X_train_cat == -1, 0, X_train_cat + 1)
            X_val_cat = np.where(X_val_cat == -1, 0, X_val_cat + 1)
            X_test_cat = np.where(X_test_cat == -1, 0, X_test_cat + 1)

            encoder = cat_pipeline.named_steps['encoder']
            # 0번 인덱스가 새로 확보되었으므로, 모델의 임베딩 레이어 크기도 이에 맞게 1씩 더 늘려 잡습니다.
            self.cat_cardinalities = [len(cats) + 1 for cats in encoder.categories_]
            
            X_train_cat_tensor = torch.tensor(X_train_cat, dtype = torch.long)
            X_val_cat_tensor = torch.tensor(X_val_cat, dtype = torch.long)
            X_test_cat_tensor = torch.tensor(X_test_cat, dtype = torch.long)
        
        else:
            self.cat_cardinalities = []

            X_train_cat_tensor = torch.empty((len(y_train), 0), dtype=torch.long)
            X_val_cat_tensor   = torch.empty((len(y_val), 0), dtype=torch.long)
            X_test_cat_tensor  = torch.empty((len(y_test), 0), dtype=torch.long)

            

        y_train_scaled_tensor = torch.tensor(y_train_scaled, dtype = torch.float32)
        y_val_scaled_tensor = torch.tensor(y_val_scaled, dtype = torch.float32)
        y_test_scaled_tensor = torch.tensor(y_test_scaled, dtype = torch.float32)

        self.train_dataset = TensorDataset(
            X_train_num_tensor ,
            X_train_cat_tensor,
            y_train_scaled_tensor
        )
        self.val_dataset = TensorDataset(
            X_val_num_tensor,
            X_val_cat_tensor,
            y_val_scaled_tensor
        )
        self.test_dataset = TensorDataset(
            X_test_num_tensor,
            X_test_cat_tensor,
            y_test_scaled_tensor
        )
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers= self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers= self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers= self.num_workers)

def compute_bin_edges(df: pd.DataFrame, num_cols: list, n_bins: int = 16) -> dict:
    """각 수치형 컬럼별로 분위수(Quantile) 기반 경계값을 계산하는 함수"""
    bin_edges_dict = {}
    for col in num_cols:
        values = df[col].dropna().values # 혹시 모를 NaN 제거
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(values, quantiles) # type:ignore
        
        edges = np.unique(edges)
        if len(edges) < 2:
            edges = np.array([values.min(), values.max() + 1e-5]) # type:ignore
            
        bin_edges_dict[col] = torch.tensor(edges, dtype=torch.float32)
        
    return bin_edges_dict