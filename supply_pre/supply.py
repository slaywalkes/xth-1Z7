import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
import json
from datetime import timedelta, datetime
import subprocess

# 解决Matplotlib中文显示问题 + 全局中文配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']

# ===================== 全局配置【中文适配】=====================
CONFIG = {
    "seq_len": 30,
    "test_ratio": 0.2,
    "multi_step_pred_days": 7,
    "confidence_level": 0.95,
    "lead_time": 7,
    "region_list": ["华北", "华南", "华东", "华西"]  # 英文改中文
}


# ===================== 1. 数据准备【核心修改：全中文数据，产品/品类/区域/促销全中文】=====================
def generate_sample_data():
    sales_df = pd.DataFrame()
    inventory_df = pd.DataFrame()
    economic_df = pd.DataFrame()
    promotion_calendar = pd.DataFrame()

    try:
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        product_ids = ["产品001", "产品002", "产品003", "产品004", "产品005"]  # 英文改中文
        product_meta = {
            "产品001": {"category": "电子产品", "region": "华北", "promotion": "无促销"},  # 全中文
            "产品002": {"category": "电子产品", "region": "华南", "promotion": "折扣促销"},
            "产品003": {"category": "服饰类", "region": "华东", "promotion": "无促销"},
            "产品004": {"category": "食品类", "region": "华西", "promotion": "折扣促销"},
            "产品005": {"category": "食品类", "region": "华北", "promotion": "无促销"}
        }
        comp_prices = {pid: np.random.uniform(80, 150) for pid in product_ids}

        sales_data = []
        for pid in product_ids:
            cat = product_meta[pid]["category"]
            promo = product_meta[pid]["promotion"]
            region = product_meta[pid]["region"]
            comp_price = comp_prices[pid]
            base_sales = 100 if cat == "电子产品" else 80 if cat == "服饰类" else 50
            promo_factor = 1.5 if promo == "折扣促销" else 1.0
            for date in dates:
                season_factor = 1.2 if date.month in [6,7,8] else 0.9 if date.month in [1,2,12] else 1.0
                comp_factor = 0.8 if comp_price < 100 else 1.0
                sales = int(np.random.normal(base_sales * promo_factor * season_factor * comp_factor, 10))
                sales = max(0, sales)
                sales_data.append({
                    "产品编号": pid,          # 所有列名全改中文
                    "销售日期": date,
                    "销量": sales,
                    "产品品类": cat,
                    "促销类型": promo,
                    "销售区域": region,
                    "竞品价格": comp_price
                })

        inventory_data = []
        for pid in product_ids:
            inventory = 500 if pid.startswith("产品001") else 300 if pid.startswith("产品002") else 200
            for date in dates:
                inventory_data.append({
                    "产品编号": pid,
                    "记录日期": date,
                    "库存数量": inventory
                })
                daily_sales = next(s["销量"] for s in sales_data if s["产品编号"] == pid and s["销售日期"] == date)
                inventory = max(0, inventory - daily_sales)
                if date.day == 1:
                    inventory = 500 if pid.startswith("产品001") else 300 if pid.startswith("产品002") else 200

        economic_data = {
            "日期": dates,
            "经济指标": np.random.uniform(90, 110, len(dates))
        }
        economic_df = pd.DataFrame(economic_data)

        promotion_calendar = pd.DataFrame({
            "促销活动": ["春季促销", "618大促", "双十一促销", "年终促销"],  # 中文促销名
            "开始日期": ["2023-03-01", "2023-06-01", "2023-11-01", "2023-12-01"],
            "结束日期": ["2023-03-15", "2023-06-20", "2023-11-15", "2023-12-31"]
        })

        sales_df = pd.DataFrame(sales_data).loc[:, ~pd.DataFrame(sales_data).columns.duplicated()]
        inventory_df = pd.DataFrame(inventory_data).loc[:, ~pd.DataFrame(inventory_data).columns.duplicated()]
        economic_df = economic_df.loc[:, ~economic_df.columns.duplicated()]

    except Exception as e:
        print(f"数据准备阶段出现异常：{str(e)}")
        return sales_df, inventory_df, economic_df, promotion_calendar

    return sales_df, inventory_df, economic_df, promotion_calendar


# ===================== 2. 数据预处理【全中文列名适配，无功能修改】=====================
def preprocess_data(sales_df, inventory_df, economic_df, promotion_calendar):
    required_sales_cols = ['产品编号', '销售区域', '销售日期', '销量']
    missing_sales_cols = [col for col in required_sales_cols if col not in sales_df.columns]
    if missing_sales_cols:
        raise ValueError(f"销售数据 缺失核心列：{missing_sales_cols}，无法进行预处理")

    required_inv_cols = ['产品编号', '记录日期', '库存数量']
    missing_inv_cols = [col for col in required_inv_cols if col not in inventory_df.columns]
    if missing_inv_cols:
        raise ValueError(f"库存数据 缺失核心列：{missing_inv_cols}，无法进行预处理")

    if economic_df.empty:
        raise ValueError("经济数据为空，无法进行数据预处理，请检查数据准备阶段")

    merged_df = pd.merge(
        sales_df,
        inventory_df[['产品编号', '记录日期', '库存数量']],
        left_on=['产品编号', '销售日期'],
        right_on=['产品编号', '记录日期'],
        how='left',
        suffixes=('', '_重复列')
    )
    merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_重复列')]
    merged_df = merged_df.drop('记录日期', axis=1, errors='ignore')

    merged_df = pd.merge(merged_df,economic_df,left_on='销售日期',right_on='日期',how='left')
    merged_df = merged_df.drop('日期', axis=1, errors='ignore')

    promotion_calendar['开始日期'] = pd.to_datetime(promotion_calendar['开始日期'])
    promotion_calendar['结束日期'] = pd.to_datetime(promotion_calendar['结束日期'])
    merged_df['是否促销期'] = 0
    for _, promo in promotion_calendar.iterrows():
        mask = (merged_df['销售日期'] >= promo['开始日期']) & (merged_df['销售日期'] <= promo['结束日期'])
        merged_df.loc[mask, '是否促销期'] = 1

    numeric_cols = ['销量', '库存数量', '经济指标', '竞品价格']
    merged_df[numeric_cols] = merged_df[numeric_cols].fillna(merged_df[numeric_cols].median())
    merged_df['促销类型'] = merged_df['促销类型'].fillna('无促销')

    for col in numeric_cols:
        mean = merged_df[col].mean()
        std = merged_df[col].std()
        merged_df = merged_df[(merged_df[col] >= mean - 3 * std) & (merged_df[col] <= mean + 3 * std)]

    encoded_df = pd.get_dummies(merged_df,columns=['促销类型'],prefix=['促销'],drop_first=False)
    target_col = '销量'
    keep_cols = ['产品编号', '销售日期', '产品品类', '销售区域', target_col]
    keep_cols = list(dict.fromkeys(keep_cols))
    features = encoded_df.drop(keep_cols, axis=1, errors='ignore')
    target = encoded_df[target_col]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns, index=encoded_df.index)

    result_df = pd.concat([encoded_df[keep_cols],features_scaled_df], axis=1)
    result_df = result_df.loc[:, ~result_df.columns.duplicated()]

    missing_core_cols = [col for col in ['产品编号', '销售区域'] if col not in result_df.columns]
    if missing_core_cols:
        raise ValueError(f"预处理后缺失核心列：{missing_core_cols}")

    return result_df, scaler, features.columns.tolist()


# ===================== 3. 特征工程【中文列名适配】=====================
def extract_features(df):
    df = df.copy()
    df['销售日期'] = pd.to_datetime(df['销售日期'])

    missing_core_cols = [col for col in ['产品编号', '销售区域'] if col not in df.columns]
    if missing_core_cols:
        raise ValueError(f"输入数据缺失核心列：{missing_core_cols}，无法进行特征工程")

    df = df.loc[:, ~df.columns.duplicated()]

    holidays = pd.to_datetime(["2023-01-01", "2023-01-22", "2023-04-05", "2023-05-01", "2023-10-01"])
    df['是否节假日'] = df['销售日期'].isin(holidays).astype(int)
    df['年份'] = df['销售日期'].dt.year
    df['月份'] = df['销售日期'].dt.month
    df['星期'] = df['销售日期'].dt.weekday
    df['是否周末'] = (df['星期'] >= 5).astype(int)
    df['季度'] = df['销售日期'].dt.quarter
    df['当月日期'] = df['销售日期'].dt.day

    grouped_pid = df.groupby('产品编号')
    grouped_cat = df.groupby('产品品类')
    grouped_region = df.groupby('销售区域')

    base_features = {
        '7日滚动销量均值': grouped_pid['销量'].rolling(window=7, min_periods=1).mean().reset_index(0,drop=True),
        '30日滚动销量方差': grouped_pid['销量'].rolling(window=30, min_periods=1).var().reset_index(0,drop=True),
        '库存周转率': df['销量'] / (df['库存数量'] + 1e-5),
        '区域7日滚动销量': grouped_region['销量'].rolling(window=7, min_periods=1).mean().reset_index(0,drop=True)
    }

    for col_name, col_data in base_features.items():
        if col_name in df.columns:
            df = df.drop(columns=[col_name])
        df[col_name] = col_data
        df[col_name] = df[col_name].fillna(df[col_name].median())

    cat_rolling_col = '品类7日滚动销量'
    if cat_rolling_col in df.columns:
        df = df.drop(columns=[cat_rolling_col])
    df[cat_rolling_col] = grouped_cat['销量'].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
    df[cat_rolling_col] = df[cat_rolling_col].fillna(df[cat_rolling_col].median())

    cat_ratio_col = '品类销量占比'
    if cat_ratio_col in df.columns:
        df = df.drop(columns=[cat_ratio_col])
    df[cat_ratio_col] = df['销量'] / (df[cat_rolling_col] + 1e-8)
    df[cat_ratio_col] = df[cat_ratio_col].fillna(df[cat_ratio_col].median())

    lag_days = [7,14,30]
    for lag in lag_days:
        col_name = f'{lag}日滞后销量'
        if col_name in df.columns:
            df = df.drop(columns=[col_name])
        df[col_name] = grouped_pid['销量'].shift(lag).reset_index(0, drop=True)
        df[col_name] = df[col_name].fillna(df[col_name].median())

    for lag in lag_days:
        col_name = f'{lag}日滞后销量'
        df[col_name] = df.groupby('产品编号')[col_name].ffill()
        df[col_name] = df[col_name].fillna(df['销量'].mean())

    df = df.dropna()
    df = df.loc[:, ~df.columns.duplicated()]

    required_cols = ['产品编号', '销售区域', '品类7日滚动销量', '品类销量占比']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"特征工程后缺失关键列：{missing_cols}")

    return df


# ===================== 4. 数据划分【中文列名适配】=====================
def split_time_series(df, split_dim="产品编号", test_ratio=0.2):
    missing_core_cols = [col for col in ['产品编号', '销售区域'] if col not in df.columns]
    if missing_core_cols:
        raise ValueError(f"输入数据缺失核心列：{missing_core_cols}，无法进行数据划分")

    df = df.loc[:, ~df.columns.duplicated()]
    df_sorted = df.sort_values([split_dim, '销售日期']).reset_index(drop=True)
    test_start_date = df_sorted.groupby(split_dim)['销售日期'].apply(lambda x: x.iloc[int(len(x) * (1 - test_ratio))]).reset_index()
    test_start_date.columns = [split_dim, '测试起始日期']

    df_sorted = pd.merge(df_sorted,test_start_date,on=split_dim,how='left',suffixes=('', '_重复列'))
    df_sorted = df_sorted.drop(columns=[col for col in df_sorted.columns if col.endswith('_重复列')])

    train_df = df_sorted[df_sorted['销售日期'] < df_sorted['测试起始日期']].drop('测试起始日期', axis=1)
    test_df = df_sorted[df_sorted['销售日期'] >= df_sorted['测试起始日期']].drop('测试起始日期', axis=1)

    train_df = train_df.loc[:, ~train_df.columns.duplicated()]
    test_df = test_df.loc[:, ~test_df.columns.duplicated()]

    return train_df, test_df


# ===================== 5. LSTM输入构建【中文列名适配】=====================
def create_lstm_input(df, target_col='销量', seq_len=30, split_dim="产品编号"):
    missing_core_cols = [col for col in ['产品编号', '销售区域'] if col not in df.columns]
    if missing_core_cols:
        raise ValueError(f"输入数据缺失核心列：{missing_core_cols}，无法构建LSTM输入")

    df = df.loc[:, ~df.columns.duplicated()]
    feature_cols = df.columns.difference(['产品编号', '销售日期', '产品品类', '销售区域', target_col])
    features = df[feature_cols].values
    target = df[target_col].values
    split_ids = df[split_dim].values
    dates = df['销售日期'].values
    product_ids = df['产品编号'].values
    product_cats = df['产品品类'].values
    product_regions = df['销售区域'].values

    X, y, ids, pred_dates, products, cats, regions = [], [], [], [], [], [], []
    unique_ids = np.unique(split_ids)
    for uid in unique_ids:
        mask = split_ids == uid
        feat = features[mask]
        tgt = target[mask]
        dt = dates[mask]
        pids = product_ids[mask]
        pcats = product_cats[mask]
        pregions = product_regions[mask]

        for i in range(seq_len, len(feat)):
            X.append(feat[i - seq_len:i])
            y.append(tgt[i])
            ids.append(uid)
            pred_dates.append(dt[i])
            products.append(pids[i])
            cats.append(pcats[i])
            regions.append(pregions[i])

    return np.array(X), np.array(y), np.array(ids), np.array(pred_dates), np.array(products), np.array(cats), np.array(regions)


# ===================== 6. 模型定义与训练【无修改，核心算法不变】=====================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,batch_first=True, dropout=dropout if num_layers >1 else 0)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :]).squeeze()

def train_models(X_train, y_train, X_test, y_test):
    input_dim = X_train.shape[-1]
    lstm_model = LSTMModel(input_dim=input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    lstm_model.train()
    for epoch in range(30):
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = lstm_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        if (epoch +1) %10 ==0:
            lstm_model.eval()
            with torch.no_grad():
                test_loss = criterion(lstm_model(X_test_tensor), y_test_tensor).item()
            print(f"LSTM 训练轮次 [{epoch +1}/30], 训练损失: {epoch_loss:.4f}, 测试损失: {test_loss:.4f}")
            lstm_model.train()

    X_train_flat = X_train[:, -1, :]
    X_test_flat = X_test[:, -1, :]
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,objective='reg:squarederror', random_state=42)
    xgb_model.fit(X_train_flat, y_train, eval_set=[(X_test_flat, y_test)], verbose=10)

    return lstm_model, xgb_model

def ensemble_predict(lstm_model, xgb_model, X):
    lstm_model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        lstm_pred = lstm_model(X_tensor).numpy()

    X_flat = X[:, -1, :]
    xgb_pred = xgb_model.predict(X_flat)
    return 0.6 * lstm_pred + 0.4 * xgb_pred


# ===================== 7. 多步预测模块【核心中文修改，所有输出中文】=====================
def multi_step_prediction(lstm_model, xgb_model, df, split_dim="产品编号", pred_days=7):
    missing_core_cols = [col for col in ['产品编号', '销售区域'] if col not in df.columns]
    if missing_core_cols:
        raise ValueError(f"输入数据缺失核心列：{missing_core_cols}，无法进行多步预测")

    df = df.loc[:, ~df.columns.duplicated()]
    df_sorted = df.sort_values([split_dim, '销售日期']).reset_index(drop=True)
    final_preds = []
    feature_cols = df_sorted.columns.difference(['产品编号', '销售日期', '产品品类', '销售区域', '销量'])

    for uid in np.unique(df_sorted[split_dim]):
        uid_df = df_sorted[df_sorted[split_dim] == uid].tail(CONFIG["seq_len"])
        if len(uid_df) < CONFIG["seq_len"]:
            continue

        features = uid_df[feature_cols].values
        current_seq = features.reshape(1, CONFIG["seq_len"], -1)
        last_date = uid_df['销售日期'].max()
        product_id = uid_df['产品编号'].iloc[0]
        product_category = uid_df['产品品类'].iloc[0]
        region = uid_df['销售区域'].iloc[0]

        col_mapping = {col: idx for idx, col in enumerate(feature_cols)}
        for i in range(pred_days):
            pred_sales = ensemble_predict(lstm_model, xgb_model, current_seq)[0]
            pred_sales = max(0, round(pred_sales))
            next_date = last_date + timedelta(days=i + 1)
            next_features = features[-1:].copy()

            if '月份' in col_mapping: next_features[0, col_mapping['月份']] = next_date.month
            if '星期' in col_mapping: next_features[0, col_mapping['星期']] = next_date.weekday()
            if '是否周末' in col_mapping: next_features[0, col_mapping['是否周末']] = 1 if next_date.weekday() >=5 else 0
            if '季度' in col_mapping: next_features[0, col_mapping['季度']] = (next_date.month -1)//3 +1
            if '当月日期' in col_mapping: next_features[0, col_mapping['当月日期']] = next_date.day
            if '是否节假日' in col_mapping: next_features[0, col_mapping['是否节假日']] = 1 if next_date in pd.to_datetime(["2023-01-01", "2023-01-22"]) else 0

            for lag in [7,14,30]:
                lag_col = f'{lag}日滞后销量'
                if lag_col in col_mapping:
                    next_features[0, col_mapping[lag_col]] = uid_df['销量'].mean()

            current_seq = np.roll(current_seq, -1, axis=1)
            current_seq[0, -1] = next_features

            final_preds.append({
                '产品编号': product_id,
                '产品品类': product_category,
                '销售区域': region,
                split_dim: uid,
                '预测日期': next_date,
                '预测销量': pred_sales
            })

    pred_df = pd.DataFrame(final_preds).loc[:, ~pd.DataFrame(final_preds).columns.duplicated()]
    return pred_df


# ===================== 8. 库存优化与补货计划【全中文列名+中文内容】=====================
def generate_replenishment_plan(pred_df, inventory_df, split_dim="产品编号"):
    required_pred_cols = ['产品编号', '销售区域', '预测日期', '预测销量', '实际销量']
    missing_pred_cols = [col for col in required_pred_cols if col not in pred_df.columns]
    if missing_pred_cols:
        raise KeyError(f"预测数据 缺失核心列：{missing_pred_cols}，无法生成补货计划")

    required_inv_cols = ['产品编号', '记录日期', '库存数量']
    missing_inv_cols = [col for col in required_inv_cols if col not in inventory_df.columns]
    if missing_inv_cols:
        raise KeyError(f"库存数据 缺失核心列：{missing_inv_cols}，无法生成补货计划")

    pred_df = pred_df.loc[:, ~pred_df.columns.duplicated()]
    inventory_df = inventory_df.loc[:, ~inventory_df.columns.duplicated()]

    pred_df_filtered = pred_df[['产品编号', '预测日期', '预测销量', '实际销量', '产品品类', '销售区域']].copy()
    inventory_df_filtered = inventory_df[['产品编号', '记录日期', '库存数量']].copy()

    merged_df = pd.merge(pred_df_filtered,inventory_df_filtered,left_on=['产品编号', '预测日期'],right_on=['产品编号', '记录日期'],how='left',suffixes=('', '_重复列'))
    merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_重复列')]
    merged_df = merged_df.drop('记录日期', axis=1, errors='ignore')
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated(keep='first')]

    if split_dim not in merged_df.columns:
        merged_df[split_dim] = merged_df['产品编号']
    safety_stock = merged_df.groupby(split_dim)['实际销量'].agg(['mean', 'std']).reset_index()
    safety_stock['安全库存'] = 1.645 * safety_stock['std'] * np.sqrt(CONFIG["lead_time"])
    safety_stock['安全库存'] = safety_stock['安全库存'].round().astype(int)

    merged_df = pd.merge(merged_df,safety_stock[[split_dim, '安全库存']],on=split_dim,how='left',suffixes=('', '_重复列'))
    merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_重复列')]
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated(keep='first')]

    merged_df['库存数量'] = merged_df.groupby(split_dim)['库存数量'].ffill().fillna(0)

    if split_dim in ["产品品类", "销售区域"]:
        weekly_demand = merged_df.groupby([split_dim, pd.Grouper(key='预测日期', freq='W')])['预测销量'].sum().reset_index()
        weekly_demand.columns = [split_dim, '预测周', '周预测销量']
        merged_df = pd.merge(merged_df, weekly_demand, on=split_dim, how='left')
        merged_df['补货数量'] = merged_df['周预测销量'] + merged_df['安全库存'] - merged_df['库存数量']
    else:
        merged_df['补货数量'] = merged_df['预测销量'] + merged_df['安全库存'] - merged_df['库存数量']

    merged_df['补货数量'] = merged_df['补货数量'].clip(lower=0).round().astype(int)
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated(keep='first')]

    return_cols = [split_dim, '产品编号', '产品品类', '销售区域', '预测日期', '实际销量', '预测销量', '库存数量', '安全库存', '补货数量']
    return_cols = [col for col in return_cols if col in merged_df.columns]
    return merged_df[merged_df['补货数量'] > 0][return_cols]


# ===================== 9. 异常检测模块【中文列名+中文异常类型】=====================
def detect_anomalies(df, split_dim="产品编号"):
    missing_core_cols = [col for col in ['产品编号', '销售区域'] if col not in df.columns]
    if missing_core_cols:
        raise ValueError(f"输入数据缺失核心列：{missing_core_cols}，无法进行异常检测")

    df = df.loc[:, ~df.columns.duplicated()]
    anomalies = []
    grouped = df.groupby(split_dim)

    for uid, group in grouped:
        group_copy = group.copy(deep=True)
        group_copy['7日滚动均值'] = group_copy['销量'].rolling(window=7, min_periods=3).mean()
        group_copy['7日滚动标准差'] = group_copy['销量'].rolling(window=7, min_periods=3).std()
        group_copy['上限值'] = group_copy['7日滚动均值'] + 2 * group_copy['7日滚动标准差']
        group_copy['下限值'] = group_copy['7日滚动均值'] - 2 * group_copy['7日滚动标准差']
        group_copy['是否异常'] = (group_copy['销量'] > group_copy['上限值']) | (group_copy['销量'] < group_copy['下限值'])

        selected_cols = [split_dim, '产品编号', '产品品类', '销售区域', '销售日期', '销量', '7日滚动均值', '上限值', '下限值']
        available_cols = [col for col in selected_cols if col in group_copy.columns]
        anomaly_records = group_copy[group_copy['是否异常']][available_cols]
        anomaly_records = anomaly_records.loc[:, ~anomaly_records.columns.duplicated()]
        anomalies.append(anomaly_records)

    anomaly_df = pd.concat(anomalies, ignore_index=True, axis=0)
    anomaly_df = anomaly_df.loc[:, ~anomaly_df.columns.duplicated()]

    if all(col in anomaly_df.columns for col in ['销量', '上限值', '下限值']):
        anomaly_df['异常类型'] = np.where(anomaly_df['销量'] > anomaly_df['上限值'], '销量突增', '销量突降') # 中文异常类型

    return anomaly_df


# ===================== 10. 协同预测接口【中文JSON内容，无序列化报错】=====================
def generate_collaborative_plan(pred_df, replenishment_plan):
    missing_core_cols = [col for col in ['产品编号', '销售区域'] if col not in pred_df.columns]
    if missing_core_cols:
        raise ValueError(f"预测数据 缺失核心列：{missing_core_cols}，无法生成协同预测计划")

    pred_df = pred_df.loc[:, ~pred_df.columns.duplicated()]
    replenishment_plan = replenishment_plan.loc[:, ~replenishment_plan.columns.duplicated()]

    if '预测日期' in pred_df.columns:
        pred_df = pred_df.copy()
        pred_df['预测日期'] = pred_df['预测日期'].dt.strftime('%Y-%m-%d %H:%M:%S')

    replenishment_plan = replenishment_plan.copy()
    for col in replenishment_plan.columns:
        if pd.api.types.is_datetime64_any_dtype(replenishment_plan[col]):
            replenishment_plan[col] = replenishment_plan[col].dt.strftime('%Y-%m-%d %H:%M:%S')

    supplier_plan = pred_df.groupby('产品品类')['预测销量'].sum().reset_index()
    supplier_plan.columns = ['产品品类', '总需求']
    supplier_plan['建议供货量'] = supplier_plan['总需求'] * 1.1

    collaborative_plan = pd.merge(replenishment_plan,supplier_plan,on='产品品类',how='left',suffixes=('', '_重复列'))
    collaborative_plan = collaborative_plan.loc[:, ~collaborative_plan.columns.str.endswith('_重复列')]

    min_date = pred_df['预测日期'].min() if '预测日期' in pred_df.columns else '未知'
    max_date = pred_df['预测日期'].max() if '预测日期' in pred_df.columns else '未知'

    report = {
        "预测周期": f"{min_date} 至 {max_date}",
        "总预测需求量": int(pred_df['预测销量'].sum()),
        "总补货数量": int(replenishment_plan['补货数量'].sum()),
        "品类需求分布": supplier_plan.to_dict('records'),
        "补货详情": collaborative_plan.to_dict('records')
    }

    with open("results/协同预测报告.json", 'w', encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2,default=lambda x: str(x))

    return report


# ===================== 11. 结果可视化【核心修改：图表标题/标签全中文，文件名中文】=====================
def visualize_results(pred_df, multi_step_pred_df, replenishment_plan, anomaly_df):
    pred_df = pred_df.copy(deep=True)
    replenishment_plan = replenishment_plan.copy(deep=True)
    multi_step_pred_df = multi_step_pred_df.copy(deep=True)

    missing_core_cols = [col for col in ['产品编号', '销售区域'] if col not in pred_df.columns]
    if missing_core_cols:
        raise ValueError(f"预测数据 缺失核心列：{missing_core_cols}，无法进行可视化")

    pred_df = pred_df.loc[:, ~pred_df.columns.duplicated()]
    replenishment_plan = replenishment_plan.loc[:, ~replenishment_plan.columns.duplicated()]
    multi_step_pred_df = multi_step_pred_df.loc[:, ~multi_step_pred_df.columns.duplicated()]

    if '产品编号' in pred_df.columns:
        pred_df.loc[:, '产品编号'] = pred_df['产品编号'].squeeze()
        pred_df.loc[:, '产品编号'] = pred_df['产品编号'].astype(str)
        pred_df = pred_df.reset_index(drop=True)

    if '产品编号' in replenishment_plan.columns:
        replenishment_plan.loc[:, '产品编号'] = replenishment_plan['产品编号'].squeeze()
        replenishment_plan.loc[:, '产品编号'] = replenishment_plan['产品编号'].astype(str)
        replenishment_plan = replenishment_plan.reset_index(drop=True)

    Path("results").mkdir(exist_ok=True)

    pred_df_viz = pred_df.copy(deep=True)
    if '预测日期' in pred_df_viz.columns:
        try:
            pred_df_viz.loc[:, '预测日期'] = pd.to_datetime(pred_df_viz['预测日期'])
        except:
            pass

    multi_step_pred_df_viz = multi_step_pred_df.copy(deep=True)
    if '预测日期' in multi_step_pred_df_viz.columns:
        try:
            multi_step_pred_df_viz.loc[:, '预测日期'] = pd.to_datetime(multi_step_pred_df_viz['预测日期'])
        except:
            pass

    top_product = pred_df_viz.groupby('产品编号')['实际销量'].sum().nlargest(1).index[0]
    product_data = pred_df_viz[pred_df_viz['产品编号'] == top_product]
    anomaly_data = anomaly_df[anomaly_df['产品编号'] == top_product] if '产品编号' in anomaly_df.columns else pd.DataFrame()
    multi_step_product = multi_step_pred_df_viz[multi_step_pred_df_viz['产品编号'] == top_product]

    plt.figure(figsize=(14, 8))
    plt.plot(pd.to_datetime(product_data['预测日期']), product_data['实际销量'], label='实际销量', linewidth=1.5)
    plt.plot(pd.to_datetime(product_data['预测日期']), product_data['预测销量'], '--', label='预测销量', linewidth=1.5)
    if not anomaly_data.empty:
        plt.scatter(pd.to_datetime(anomaly_data['销售日期']), anomaly_data['销量'], color='red',label='异常值', s=50)
    plt.plot(pd.to_datetime(multi_step_product['预测日期']), multi_step_product['预测销量'], color='orange',label='未来7天预测', linewidth=2)

    plt.xlabel('日期')
    plt.ylabel('销量')
    plt.title(f'{top_product}销量趋势+异常检测+多步预测')  # 中文标题
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/销量趋势异常预测图.png")  # 中文文件名
    plt.close()

    cat_pred = pred_df_viz.groupby('产品品类')['预测销量'].sum().reset_index()
    plt.figure(figsize=(10, 6))
    plt.bar(cat_pred['产品品类'], cat_pred['预测销量'], color=['blue', 'green', 'red'])
    plt.xlabel('产品品类')
    plt.ylabel('预测总销量')
    plt.title('品类级预测销量分布')
    plt.tight_layout()
    plt.savefig("results/品类销量分布图.png")
    plt.close()

    replenish_summary = replenishment_plan.groupby('产品编号')['补货数量'].sum().reset_index()
    plt.figure(figsize=(12, 6))
    plt.bar(replenish_summary['产品编号'], replenish_summary['补货数量'], color='purple')
    plt.xlabel('产品编号')
    plt.ylabel('补货总量')
    plt.title('各产品补货总量汇总')
    plt.tight_layout()
    plt.savefig("results/产品补货汇总图.png")
    plt.close()

    print("可视化图表已保存至 results/ 目录")


# ===================== 12. 主流程【全中文输出、中文文件名、无功能修改】=====================
def main():
    print("=== 智能供应链需求预测系统启动【纯中文数据版】===")

    print("\nStep 1/10: 准备多源中文数据...")
    sales_df, inventory_df, economic_df, promotion_calendar = generate_sample_data()

    if economic_df is None or economic_df.empty:
        raise Exception("经济数据未被正确赋值或为空，程序终止")

    missing_sales_core = [col for col in ['产品编号', '销售区域'] if col not in sales_df.columns]
    if missing_sales_core:
        raise Exception(f"销售数据 缺失核心列：{missing_sales_core}，程序终止")

    print("Step 2/10: 数据预处理+外部因素融合...")
    preprocessed_df, scaler, feature_cols = preprocess_data(sales_df, inventory_df, economic_df, promotion_calendar)

    print("Step 3/10: 提取多维度特征...")
    featured_df = extract_features(preprocessed_df)

    print("Step 4/10: 多维度数据划分...")
    train_df, test_df = split_time_series(featured_df, split_dim="产品编号", test_ratio=CONFIG["test_ratio"])
    X_train, y_train, train_ids, _, _, _, _ = create_lstm_input(train_df, seq_len=CONFIG["seq_len"])
    X_test, y_test, test_split_ids, test_dates, test_products, test_cats, test_regions = create_lstm_input(test_df,seq_len=CONFIG["seq_len"])

    print("Step 5/10: 训练混合模型...")
    lstm_model, xgb_model = train_models(X_train, y_train, X_test, y_test)

    print("Step 6/10: 单品级需求预测...")
    predicted_sales = ensemble_predict(lstm_model, xgb_model, X_test)
    pred_df = pd.DataFrame({
        '产品编号': test_products,
        '预测日期': test_dates,
        '实际销量': y_test,
        '预测销量': predicted_sales.round(0).astype(int),
        '产品品类': test_cats,
        '销售区域': test_regions
    })
    pred_df = pred_df.loc[:, ~pred_df.columns.duplicated(keep='first')]

    print("Step 7/10: 多步需求预测...")
    multi_step_pred_df = multi_step_prediction(lstm_model, xgb_model, featured_df, split_dim="产品编号",pred_days=CONFIG["multi_step_pred_days"])

    print("Step 8/10: 需求异常检测...")
    anomaly_df = detect_anomalies(featured_df, split_dim="产品编号")
    print(f"\n检测到 {len(anomaly_df)} 条需求异常记录")
    anomaly_df.to_csv("results/异常检测结果.csv", index=False, encoding='utf-8-sig')  # 中文编码

    print("Step 9/10: 生成多维度补货计划...")
    replenishment_plan = generate_replenishment_plan(pred_df, inventory_df, split_dim="产品编号")
    cat_pred_df = pred_df.groupby(['产品品类', '预测日期'])['预测销量'].sum().reset_index()
    cat_pred_df['实际销量'] = pred_df.groupby(['产品品类', '预测日期'])['实际销量'].sum().values
    cat_pred_df['产品编号'] = cat_pred_df['产品品类']
    cat_pred_df['销售区域'] = '全部'
    cat_replenishment_plan = generate_replenishment_plan(cat_pred_df, inventory_df, split_dim="产品品类")

    print("Step 10/10: 生成供应链协同预测计划...")
    collaborative_report = generate_collaborative_plan(pred_df, replenishment_plan)

    print("\n=== 模型评估结果 ===")
    mae = mean_absolute_error(y_test, predicted_sales)
    rmse = np.sqrt(mean_squared_error(y_test, predicted_sales))
    print(f"平均绝对误差（MAE）：{mae:.2f}")
    print(f"均方根误差（RMSE）：{rmse:.2f}")

    visualize_results(pred_df, multi_step_pred_df, replenishment_plan, anomaly_df)

    # 保存所有中文文件，指定UTF8编码防止中文乱码
    pred_df.to_csv("results/单品预测结果.csv", index=False, encoding='utf-8-sig')
    multi_step_pred_df.to_csv("results/7天多步预测结果.csv", index=False, encoding='utf-8-sig')
    replenishment_plan.to_csv("results/单品补货计划.csv", index=False, encoding='utf-8-sig')
    cat_replenishment_plan.to_csv("results/品类补货计划.csv", index=False, encoding='utf-8-sig')

    print("\n=== 系统运行完成！以下中文文件已生成 ===")
    print("1. 单品级预测结果：results/单品预测结果.csv")
    print("2. 7天多步预测结果：results/7天多步预测结果.csv")
    print("3. 异常检测结果：results/异常检测结果.csv")
    print("4. 单品级补货计划：results/单品补货计划.csv")
    print("5. 品类级补货计划：results/品类补货计划.csv")
    print("6. 协同预测报告：results/协同预测报告.json")
    print("7. 可视化图表：results/ 目录下3张中文标题图表")


if __name__ == "__main__":
    try:
        import torch, xgboost, pandas, numpy, sklearn, matplotlib
    except ImportError:
        subprocess.check_call(["pip", "install", "torch", "xgboost", "pandas", "numpy", "scikit-learn", "matplotlib"])
    main()