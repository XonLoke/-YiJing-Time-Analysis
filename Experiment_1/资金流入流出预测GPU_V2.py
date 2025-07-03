import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from holidays import country_holidays

# GPU相关导入
import torch

# 易经预测模块
import sys
sys.path.append('D:/sep_venv/AI_venv/AI_Lesson21_codes')
from financial_iching_predictor import IChingFinancialOracle

print(f"PyTorch 版本: {torch.__version__}")

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch 检测到的设备: {device}")
    print(f"PyTorch CUDA 可用状态: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        tensor = torch.randn(1024, 1024).to(device)
        result = torch.matmul(tensor, tensor)
        result = result.to("cpu")
        print("GPU张量乘法结果均值:", result.mean().item())
    else:
        print("未检测到GPU, 仅使用CPU.")
except Exception as e:
    print(f"CUDA 初始化过程中发生错误: {e}")

pd.set_option('display.max_columns', None)
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("环境初始化完成 (GPU版)")
print("=" * 50)

# 创建易经预测器
iching_oracle = IChingFinancialOracle()
adjustment_factor = 0.05  # 5%的调整幅度

try:
    print("正在加载数据文件...")
    user_profile = pd.read_csv('D:/sep_venv/AI_venv/AI_Lesson21_codes/CASE-资金流入流出预测/user_profile_table.csv')
    user_balance = pd.read_csv('D:/sep_venv/AI_venv/AI_Lesson21_codes/CASE-资金流入流出预测/user_balance_table.csv')
    mfd_interest = pd.read_csv('D:/sep_venv/AI_venv/AI_Lesson21_codes/CASE-资金流入流出预测/mfd_day_share_interest.csv')
    shibor = pd.read_csv('D:/sep_venv/AI_venv/AI_Lesson21_codes/CASE-资金流入流出预测/mfd_bank_shibor.csv')
    print("数据加载成功！")
    print("用户信息表形状:", user_profile.shape)
    print("资金流水表形状:", user_balance.shape)
    print("收益率表形状:", mfd_interest.shape)
    print("Shibor表形状:", shibor.shape)
    print("\n资金流水日期范围:", user_balance['report_date'].min(), "至", user_balance['report_date'].max())
    print("\n用户信息表列名:", list(user_profile.columns))
    print("资金流水表列名:", list(user_balance.columns))
    print("收益率表列名:", list(mfd_interest.columns))
    print("Shibor表列名:", list(shibor.columns))
except Exception as e:
    print(f"数据加载失败: {e}")
    exit(1)

print("=" * 50)
print("开始数据预处理...")
def convert_date(df, date_col):
    df[date_col] = pd.to_datetime(df[date_col], format='%Y%m%d')
    return df
user_balance = convert_date(user_balance, 'report_date')
mfd_interest = convert_date(mfd_interest, 'mfd_date')
shibor = convert_date(shibor, 'mfd_date')
daily_balance = user_balance.groupby('report_date').agg({
    'total_purchase_amt': 'sum',
    'total_redeem_amt': 'sum'
}).reset_index()
print(f"聚合后数据形状: {daily_balance.shape}")
print(f"日期范围: {daily_balance['report_date'].min()} 至 {daily_balance['report_date'].max()}")
daily_data = pd.merge(daily_balance, mfd_interest, left_on='report_date', right_on='mfd_date', how='left')
daily_data = pd.merge(daily_data, shibor, left_on='report_date', right_on='mfd_date', how='left')
daily_data.drop(['mfd_date_x', 'mfd_date_y'], axis=1, inplace=True)
daily_data['day_of_week'] = daily_data['report_date'].dt.dayofweek
daily_data['is_weekend'] = daily_data['day_of_week'].isin([5,6]).astype(int)
daily_data['month'] = daily_data['report_date'].dt.month
daily_data['day_of_month'] = daily_data['report_date'].dt.day
cn_holidays = country_holidays('CN')
is_chinese_holiday = lambda date: date in cn_holidays
daily_data['is_holiday'] = daily_data['report_date'].apply(is_chinese_holiday).astype(int)
daily_data['is_workday'] = ~(daily_data['is_holiday'].astype(bool) | daily_data['is_weekend'].astype(bool)).astype(int)
print(f"预处理后数据形状: {daily_data.shape}")
print("数据预处理完成")
print("=" * 50)

print("开始探索性数据分析...")
print("数据基本统计信息:")
print(daily_data[['total_purchase_amt', 'total_redeem_amt', 'mfd_daily_yield', 'mfd_7daily_yield']].describe())
print("\n缺失值情况:")
print(daily_data.isnull().sum())
plt.figure(figsize=(15,6))
plt.plot(daily_data['report_date'], daily_data['total_purchase_amt'], label='申购金额', alpha=0.7)
plt.plot(daily_data['report_date'], daily_data['total_redeem_amt'], label='赎回金额', alpha=0.7)
plt.title('每日申购和赎回金额趋势')
plt.xlabel('日期')
plt.ylabel('金额 (分)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('D:/sep_venv/AI_venv/AI_Lesson21_codes/资金流动趋势图.png', dpi=300, bbox_inches='tight')
plt.show()
fig, ax1 = plt.subplots(figsize=(15,6))
ax1.plot(daily_data['report_date'], daily_data['mfd_daily_yield'], 'g-', label='万份收益', alpha=0.7)
ax1.set_xlabel('日期')
ax1.set_ylabel('万份收益', color='g')
ax1.tick_params(axis='y', labelcolor='g')
ax2 = ax1.twinx()
ax2.plot(daily_data['report_date'], daily_data['total_purchase_amt'], 'b-', label='申购', alpha=0.5)
ax2.plot(daily_data['report_date'], daily_data['total_redeem_amt'], 'r-', label='赎回', alpha=0.5)
ax2.set_ylabel('金额 (分)', color='b')
ax2.tick_params(axis='y', labelcolor='b')
plt.title('收益率与资金流动关系')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('D:/sep_venv/AI_venv/AI_Lesson21_codes/收益率与资金流动关系.png', dpi=300, bbox_inches='tight')
plt.show()
print("EDA完成")
print("=" * 50)

print("开始特征工程与相关性分析...")
shibor_cols = ['Interest_O_N', 'Interest_1_W', 'Interest_2_W', 'Interest_1_M', 'Interest_3_M', 'Interest_6_M', 'Interest_9_M', 'Interest_1_Y']
daily_data[shibor_cols] = daily_data[shibor_cols].fillna(method='ffill')
daily_data[shibor_cols] = daily_data[shibor_cols].fillna(method='bfill')
for col in shibor_cols:
    if daily_data[col].isnull().any():
        daily_data[col].fillna(daily_data[col].mean(), inplace=True)
for lag in [1, 2, 3, 7, 14, 30]:
    daily_data[f'purchase_lag_{lag}'] = daily_data['total_purchase_amt'].shift(lag)
    daily_data[f'redeem_lag_{lag}'] = daily_data['total_redeem_amt'].shift(lag)
for window in [7, 14, 30]:
    daily_data[f'purchase_rolling_mean_{window}'] = daily_data['total_purchase_amt'].rolling(window=window).mean()
    daily_data[f'purchase_rolling_std_{window}'] = daily_data['total_purchase_amt'].rolling(window=window).std()
    daily_data[f'redeem_rolling_mean_{window}'] = daily_data['total_redeem_amt'].rolling(window=window).mean()
    daily_data[f'redeem_rolling_std_{window}'] = daily_data['total_redeem_amt'].rolling(window=window).std()
daily_data = daily_data.dropna().reset_index(drop=True)
corr_matrix = daily_data[['total_purchase_amt', 'total_redeem_amt', 'mfd_daily_yield', 'mfd_7daily_yield'] + shibor_cols].corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('相关性矩阵')
plt.tight_layout()
plt.savefig('D:/sep_venv/AI_venv/AI_Lesson21_codes/相关性矩阵.png', dpi=300, bbox_inches='tight')
plt.show()
print("特征工程与相关性分析完成")
print("=" * 50)

print("开始模型训练与验证... (XGBoost GPU支持)")
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

xgb_params = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 5,
    'random_state': 42,
    'n_jobs': -1
}
print("XGBoost 尝试使用 GPU.")
if torch.cuda.is_available():
    xgb_params['tree_method'] = 'gpu_hist'
    print("XGBoost将使用GPU加速.")
else:
    print("XGBoost将使用CPU.")

feature_cols = [col for col in daily_data.columns if col not in ['report_date', 'total_purchase_amt', 'total_redeem_amt']]
feature_cols = list(set(feature_cols + ['is_holiday', 'is_workday']))
features = daily_data[feature_cols]
purchase_target = daily_data['total_purchase_amt']
redeem_target = daily_data['total_redeem_amt']
tscv = TimeSeriesSplit(n_splits=5)
def evaluate_model(model, X, y):
    scores = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        relative_error = np.mean(np.abs(y_test - preds) / y_test)
        scores.append(relative_error)
    return np.mean(scores)

xgb_purchase = XGBRegressor(**xgb_params)
purchase_score = evaluate_model(xgb_purchase, features, purchase_target)
print(f"XGBoost申购相对误差: {purchase_score:.4f}")
xgb_redeem = XGBRegressor(**xgb_params)
redeem_score = evaluate_model(xgb_redeem, features, redeem_target)
print(f"XGBoost赎回相对误差: {redeem_score:.4f}")
rf_purchase = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_purchase_score = evaluate_model(rf_purchase, features, purchase_target)
print(f"RF申购相对误差: {rf_purchase_score:.4f}")
rf_redeem = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_redeem_score = evaluate_model(rf_redeem, features, redeem_target)
print(f"RF赎回相对误差: {rf_redeem_score:.4f}")
print("模型训练与验证完成 (XGBoost GPU支持)")
print("=" * 50)

print("开始生成动态时间特征预测结果...")
xgb_purchase.fit(features, purchase_target)
xgb_redeem.fit(features, redeem_target)
def update_time_features(df, current_date):
    df['report_date'] = current_date
    df['day_of_week'] = pd.to_datetime(current_date).dayofweek
    df['is_weekend'] = int(pd.to_datetime(current_date).weekday() >= 5)
    df['month'] = pd.to_datetime(current_date).month
    df['day_of_month'] = pd.to_datetime(current_date).day
    is_chinese_holiday = lambda date: date in cn_holidays
    df['is_holiday'] = int(is_chinese_holiday(current_date))
    df['is_workday'] = int(not (is_chinese_holiday(current_date) or pd.to_datetime(current_date).weekday() >= 5))
    return df
last_day = daily_data.iloc[-1:].copy()
preds = []
# 回测历史数据
historical_trend = []
print("开始历史趋势回测...")
for i in range(1, len(daily_data)):
    # 获取当前日期和金额
    current_date = daily_data.iloc[i-1]['report_date']
    purchase = daily_data.iloc[i-1]['total_purchase_amt']
    redeem = daily_data.iloc[i-1]['total_redeem_amt']
    
    # 预测趋势
    purchase_trend, redeem_trend = iching_oracle.predict_trend(
        current_date.strftime('%Y%m%d'), purchase, redeem
    )
    
    # 获取实际值变化
    actual_purchase_change = daily_data.iloc[i]['total_purchase_amt'] - daily_data.iloc[i-1]['total_purchase_amt']
    actual_redeem_change = daily_data.iloc[i]['total_redeem_amt'] - daily_data.iloc[i-1]['total_redeem_amt']
    
    # 判断预测是否正确
    predicted_purchase_trend = 1 if purchase_trend == '升' else (-1 if purchase_trend == '降' else 0)
    predicted_redeem_trend = 1 if redeem_trend == '升' else (-1 if redeem_trend == '降' else 0)
    
    actual_purchase_trend = 1 if actual_purchase_change > 0 else (-1 if actual_purchase_change < 0 else 0)
    actual_redeem_trend = 1 if actual_redeem_change > 0 else (-1 if actual_redeem_change < 0 else 0)
    
    purchase_correct = predicted_purchase_trend == actual_purchase_trend
    redeem_correct = predicted_redeem_trend == actual_redeem_trend
    
    historical_trend.append({
        'date': current_date,
        'purchase_trend': purchase_trend,
        'predicted_purchase_trend': predicted_purchase_trend,
        'actual_purchase_trend': actual_purchase_trend,
        'purchase_correct': purchase_correct,
        'redeem_trend': redeem_trend,
        'predicted_redeem_trend': predicted_redeem_trend,
        'actual_redeem_trend': actual_redeem_trend,
        'redeem_correct': redeem_correct
    })

# 计算准确率
trend_results = pd.DataFrame(historical_trend)
purchase_accuracy = trend_results['purchase_correct'].mean()
redeem_accuracy = trend_results['redeem_correct'].mean()
print(f"\n易经趋势预测准确率:")
print(f"申购趋势准确率: {purchase_accuracy:.2%}")
print(f"赎回趋势准确率: {redeem_accuracy:.2%}")

# 使用易经调整的预测
for i in range(30):
    current_date = last_day['report_date'].values[0] + pd.Timedelta(days=i+1)
    updated_day = update_time_features(last_day.copy(), current_date)
    pred_purchase = xgb_purchase.predict(updated_day[feature_cols])[0]
    pred_redeem = xgb_redeem.predict(updated_day[feature_cols])[0]
    
    # 应用易经调整
    date_str = current_date.strftime('%Y%m%d')
    purchase_trend, redeem_trend = iching_oracle.predict_trend(
        date_str, pred_purchase, pred_redeem
    )
    
    # 调整预测值
    if purchase_trend == '升':
        final_purchase = pred_purchase * (1 + adjustment_factor)
    elif purchase_trend == '降':
        final_purchase = pred_purchase * (1 - adjustment_factor)
    else:
        final_purchase = pred_purchase
    
    if redeem_trend == '升':
        final_redeem = pred_redeem * (1 + adjustment_factor)
    elif redeem_trend == '降':
        final_redeem = pred_redeem * (1 - adjustment_factor)
    else:
        final_redeem = pred_redeem
    
    preds.append({
        'report_date': current_date,
        'ml_purchase': pred_purchase,
        'ml_redeem': pred_redeem,
        'iching_purchase_trend': purchase_trend,
        'iching_redeem_trend': redeem_trend,
        'final_purchase': final_purchase,
        'final_redeem': final_redeem
    })
    
    # 更新滞后特征(简单实现)
    last_day['purchase_lag_1'] = final_purchase
    last_day['redeem_lag_1'] = final_redeem

future_data = pd.DataFrame(preds)
submission = pd.DataFrame({
    'report_date': future_data['report_date'].dt.strftime('%Y%m%d'),
    'purchase': future_data['final_purchase'].round().astype(int).clip(lower=0),
    'redeem': future_data['final_redeem'].round().astype(int).clip(lower=0)
})
submission['purchase'] = submission['purchase'].clip(lower=0)
submission['redeem'] = submission['redeem'].clip(lower=0)
import os
def write_csv(path, df):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        abs_path = os.path.abspath(path)
        df.to_csv(abs_path, index=False, header=False)
        size = os.path.getsize(abs_path)
        print(f"✓ 文件已创建: {abs_path} (大小: {size} bytes)")
        return True
    except Exception as e:
        print(f"✗ 文件写入失败 {path}: {str(e)}")
        return False
output_dir = 'D:/sep_venv/AI_venv/AI_Lesson21_codes/CASE-资金流入流出预测'
output_dir2 = 'D:/sep_venv/AI_venv/AI_Lesson21_codes/Submit'
try:
    submission.to_csv(f"{output_dir2}/tc_comp_predict_table.csv", index=False, header=False)
    print(f"✓ 原始文件已保存: {output_dir2}/tc_comp_predict_table.csv")
    submission.to_csv(f"{output_dir}/tc_comp_predict_table_GPU_V2.csv", index=False, header=False)
    print(f"✓ 原始文件已保存: {output_dir}/tc_comp_predict_table_GPU_V2.csv")
except Exception as e:
    print(f"✗ 文件保存失败: {str(e)}")
plt.figure(figsize=(15,6))
plt.plot(daily_data['report_date'], daily_data['total_purchase_amt'], label='历史申购')
plt.plot(future_data['report_date'], submission['purchase'], label='预测申购')
plt.plot(daily_data['report_date'], daily_data['total_redeem_amt'], label='历史赎回')
plt.plot(future_data['report_date'], submission['redeem'], label='预测赎回')
plt.title('历史与预测资金流动 (易经调整版)')
plt.xlabel('日期')
plt.ylabel('金额 (分)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('D:/sep_venv/AI_venv/AI_Lesson21_codes/历史与预测资金流动_V2.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n预测统计信息:")
print("申购 - 均值:", submission['purchase'].mean(), "标准差:", submission['purchase'].std())
print("赎回 - 均值:", submission['redeem'].mean(), "标准差:", submission['redeem'].std())

improvement_suggestions = """
1. 更精细的特征工程：
   - 添加节假日和特殊事件标记
   - 考虑用户增长趋势
   - 添加宏观经济指标
2. 尝试更复杂的模型：
   - LSTM/GRU等深度学习模型
   - Prophet时间序列模型
   - 模型堆叠集成
3. 更准确的递归预测：
   - 实现完整的滚动统计更新
   - 考虑多步预测策略
4. 异常值处理：
   - 检测并处理特殊日期的异常波动
   - 添加稳健性评估
"""
print("模型改进建议:", improvement_suggestions)
print("全部流程已完成。")
print("=" * 50)