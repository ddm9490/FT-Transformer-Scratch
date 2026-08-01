from src.dataset import HousingDataModule,HousingData2Module
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

dataset = HousingDataModule("data/fetch_california_housing.csv")
dataset = HousingData2Module()
dataset.setup()

lgb_model = lgb.LGBMRegressor(
    n_estimators=2000,      # 트리의 개수 (넉넉하게 설정)
    learning_rate=0.05,     # 학습률
    random_state=42
)

data = dataset.numpy_data
y_scaler = dataset.y_scaler

lgb_model.fit(
    data["X_train"], data["y_train"],
    eval_set=[(data["X_val"], data["y_val"])],
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
)

print(data["y_train"].shape)

# 3. Validation 데이터 예측 및 평가
y_pred = lgb_model.predict(data["X_val"])

y_pred_original = y_scaler.inverse_transform(y_pred.reshape(-1,1)).flatten()
y_val_original = y_scaler.inverse_transform(data["y_val"].reshape(-1,1)).flatten()

r2 = r2_score(data["y_val"], y_pred)
rmse = np.sqrt(mean_squared_error(y_val_original, y_pred_original))

print(f" LightGBM Result -> R2: {r2:.4f} | RMSE: {rmse:.4f}")