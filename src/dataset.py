import pandas as pd
import numpy as np

import lightning as L
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, OrdinalEncoder, PowerTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
import torch

from typing import Any

class ScalableTabularDataModule(L.LightningDataModule):
    def __init__(self, dataset_dir, target_names, out_features, batch_size=1024, num_workers=3):
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

    def setup(self, stage=None):
        # 1. 원본 데이터 로드 및 Feature/Target 분리
        df : Any = pd.read_csv(self.dataset_dir)
        
        feature_df = df.drop(columns=self.target_names)

        self.num_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
        self.num_numerical = len(self.num_cols)

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
        num_imputer = SimpleImputer(strategy='mean')
        x_scaler = QuantileTransformer(n_quantiles=100, output_distribution="normal", random_state=42)

        X_train_num_imp = num_imputer.fit_transform(df_train[self.num_cols])
        X_train_num = x_scaler.fit_transform(X_train_num_imp)

        X_val_num = x_scaler.transform(num_imputer.transform(df_val[self.num_cols]))
        X_test_num = x_scaler.transform(num_imputer.transform(df_test[self.num_cols]))
        
        # 5. 범주형 데이터 전처리 (미지 카테고리 에러 방지 옵션 추가)
        cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')
        
        encoder = OrdinalEncoder(
            dtype=np.int64, # type:ignore
            handle_unknown='use_encoded_value',  # Val/Test에서 처음 보는 문자가 나와도 에러를 내지 않음
            unknown_value=-1                     # 처음 보는 문자는 우선 -1로 인코딩 함
        )
        
        X_train_cat_imp = cat_imputer.fit_transform(df_train[self.cat_cols])
        X_val_cat_imp = cat_imputer.transform(df_val[self.cat_cols])
        X_test_cat_imp = cat_imputer.transform(df_test[self.cat_cols])

        X_train_cat = encoder.fit_transform(X_train_cat_imp)
        X_val_cat = encoder.transform(X_val_cat_imp)
        X_test_cat = encoder.transform(X_test_cat_imp)

        # 6. PyTorch nn.Embedding 호환을 위한 인덱스 조정 (핵심 🚀)
        # 정상적인 카테고리(0, 1, 2...)는 1씩 밀어서 1, 2, 3...으로 만들고
        # 결측치나 처음 본 미지 카테고리(-1)는 정확히 '0'으로 수렴시킵니다.
        X_train_cat = np.where(X_train_cat == -1, 0, X_train_cat + 1)
        X_val_cat = np.where(X_val_cat == -1, 0, X_val_cat + 1)
        X_test_cat = np.where(X_test_cat == -1, 0, X_test_cat + 1)

        # 0번 인덱스가 새로 확보되었으므로, 모델의 임베딩 레이어 크기도 이에 맞게 1씩 더 늘려 잡습니다.
        self.cat_cardinalities = [len(cats) + 1 for cats in encoder.categories_]

        # 7. 텐서 변환 및 Dataset 생성
        self.train_dataset = TensorDataset(
            torch.FloatTensor(X_train_num), 
            torch.LongTensor(X_train_cat), 
            torch.FloatTensor(y_train_scaled)
        )
        self.val_dataset = TensorDataset(
            torch.FloatTensor(X_val_num), 
            torch.LongTensor(X_val_cat), 
            torch.FloatTensor(y_val_scaled)
        )
        self.test_dataset = TensorDataset(
            torch.FloatTensor(X_test_num), 
            torch.LongTensor(X_test_cat), 
            torch.FloatTensor(y_test_scaled)
        )
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers= self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers= self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers= self.num_workers)
