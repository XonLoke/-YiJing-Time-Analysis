import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import random

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 构建完整的文件路径
csv_path = os.path.join(script_dir, 'user_balance_table.csv')

print("正在读取数据...")
# 读取数据
df = pd.read_csv(csv_path, encoding='utf-8')

# 转换 report_date 为日期格式
df['report_date'] = pd.to_datetime(df['report_date'], format='%Y%m%d')

# 截取 2014-03-01 到 2014-08-31 的训练数据
mask = (df['report_date'] >= '2014-03-01') & (df['report_date'] <= '2014-08-31')
df_train = df.loc[mask]

print(f"训练数据期间：{df_train['report_date'].min()} 到 {df_train['report_date'].max()}")
print(f"训练数据天数：{df_train['report_date'].nunique()} 天")

# 按日期聚合申购和赎回金额
daily_data = df_train.groupby('report_date')[['total_purchase_amt', 'total_redeem_amt']].sum().reset_index()

# 添加基础周期因子
print("\n添加基础周期因子...")
daily_data['weekday'] = daily_data['report_date'].dt.dayofweek  # 0=周一, 6=周日
daily_data['day'] = daily_data['report_date'].dt.day  # 1-31
daily_data['month'] = daily_data['report_date'].dt.month  # 月份
daily_data['week_of_month'] = ((daily_data['day'] - 1) // 7) + 1  # 第几周

# === 易卦工具函数 ===
def get_trigram(value):
    """
    八卦映射函数
    根据变量数除以8的余数确定卦象
    余数: 0=坤, 1=乾, 2=兑, 3=离, 4=震, 5=巽, 6=坎, 7=艮
    """
    trigrams = {
        0: ("坤", 0, [0,0,0], "阴"),    # 坤卦平和，全虚，老母
        1: ("乾", 0, [1,1,1], "阳"),    # 乾卦平和，全实，老父
        2: ("兑", -1, [0,1,1], "阴"),   # 兑卦下降，上虚，小女
        3: ("离", 1, [1,0,1], "阴"),    # 离卦上升，中虚，中女
        4: ("震", 1, [0,0,1], "阳"),    # 震卦上升，下实，长子
        5: ("巽", -1, [1,1,0], "阴"),   # 巽卦下降，下虚，大女
        6: ("坎", -1, [0,1,0], "阳"),   # 坎卦下降，中实，中男
        7: ("艮", 1, [1,0,0], "阳")     # 艮卦上升，上实，小儿
    }
    return trigrams[value % 8]

def calculate_changing_line(z_value):
    """
    计算变爻位置
    z除以6的余数决定变爻位置：
    余数1=1号变爻(下卦最下), 余数2=2号变爻(下卦中), 余数3=3号变爻(下卦最上)
    余数4=4号变爻(上卦最下), 余数5=5号变爻(上卦中), 余数0=6号变爻(上卦最上)
    """
    remainder = z_value % 6
    return 6 if remainder == 0 else remainder

def get_changed_trigram(original_trigram, changing_line):
    """
    根据变爻计算变卦
    changing_line: 1-6，分别对应六个爻位
    1-3影响下卦，4-6影响上卦
    """
    trigram_lines = original_trigram[2].copy()  # 复制卦象

    if changing_line <= 3:
        # 影响下卦，变爻位置为 changing_line-1 (因为数组从0开始)
        line_pos = changing_line - 1
        trigram_lines[line_pos] = 1 - trigram_lines[line_pos]  # 实变虚，虚变实

    # 根据变化后的卦象找到对应的卦
    for value, (name, trend, lines, yinyang) in {
        0: ("坤", 0, [0,0,0], "阴"),
        1: ("乾", 0, [1,1,1], "阳"),
        2: ("兑", -1, [0,1,1], "阴"),
        3: ("离", 1, [1,0,1], "阴"),
        4: ("震", 1, [0,0,1], "阳"),
        5: ("巽", -1, [1,1,0], "阴"),
        6: ("坎", -1, [0,1,0], "阳"),
        7: ("艮", 1, [1,0,0], "阳")
    }.items():
        if lines == trigram_lines:
            return (name, trend, lines, yinyang)

    return original_trigram  # 如果没找到匹配，返回原卦

def generate_iching_features(purchase_series, redeem_series, date_series):
    """
    生成易卦特征
    :param purchase_series: 申购金额序列
    :param redeem_series: 赎回金额序列
    :param date_series: 对应的日期序列
    :return: DataFrame包含易卦特征
    """
    features = []

    for i, (purchase_val, redeem_val, date) in enumerate(zip(purchase_series, redeem_series, date_series)):
        # x变量数：使用申购金额（下卦为本为内）
        x = int(purchase_val / 1e6)  # 缩放避免数值过大

        # y变量数：使用时间因子（上卦为用为外）
        y = date.day + date.month * 10 + date.year

        # z变量数：使用赎回金额和时间的组合（用于计算变爻）
        z = int(redeem_val / 1e5) + date.day * date.month

        # 获取原卦
        x_trigram = get_trigram(x)  # 下卦
        y_trigram = get_trigram(y)  # 上卦

        # 计算变爻
        changing_line = calculate_changing_line(z)

        # 计算变卦
        if changing_line <= 3:
            # 变爻影响下卦
            changed_x = get_changed_trigram(x_trigram, changing_line)
            changed_y = y_trigram  # 上卦不变
        else:
            # 变爻影响上卦
            changed_x = x_trigram  # 下卦不变
            changed_y = get_changed_trigram(y_trigram, changing_line)

        # 构建特征向量
        features.append([
            x_trigram[1],      # 原卦下卦趋势
            y_trigram[1],      # 原卦上卦趋势
            changed_x[1],      # 变卦下卦趋势
            changed_y[1],      # 变卦上卦趋势
            changing_line,     # 变爻位置
            1 if x_trigram[3] == "阳" else 0,  # 下卦阴阳
            1 if y_trigram[3] == "阳" else 0,  # 上卦阴阳
            (x_trigram[1] + y_trigram[1]) / 2,  # 原卦综合趋势
            (changed_x[1] + changed_y[1]) / 2   # 变卦综合趋势
        ])

    return pd.DataFrame(features, columns=[
        'original_lower_trend',    # 原卦下卦趋势
        'original_upper_trend',    # 原卦上卦趋势
        'changed_lower_trend',     # 变卦下卦趋势
        'changed_upper_trend',     # 变卦上卦趋势
        'changing_line',           # 变爻位置
        'lower_yang',              # 下卦阴阳
        'upper_yang',              # 上卦阴阳
        'original_combined_trend', # 原卦综合趋势
        'changed_combined_trend'   # 变卦综合趋势
    ])

# 创建基础特征矩阵（与原代码保持一致）
print("\n创建基础特征矩阵...")
# 使用one-hot编码处理weekday
weekday_dummies = pd.get_dummies(daily_data['weekday'], prefix='weekday')
day_normalized = (daily_data['day'] - daily_data['day'].mean()) / daily_data['day'].std()

# 组合基础特征
X = pd.concat([weekday_dummies, day_normalized.rename('day_norm')], axis=1)
y_purchase = daily_data['total_purchase_amt']
y_redeem = daily_data['total_redeem_amt']

print(f"基础特征矩阵形状：{X.shape}")
print(f"基础特征列：{list(X.columns)}")

# === 添加易卦特征 ===
print("\n添加易卦特征...")
iching_features = generate_iching_features(
    daily_data['total_purchase_amt'],
    daily_data['total_redeem_amt'],
    daily_data['report_date']
)

print(f"易卦特征形状：{iching_features.shape}")
print(f"易卦特征列：{list(iching_features.columns)}")

# 合并基础特征和易卦特征
X_enhanced = pd.concat([X, iching_features], axis=1)
print(f"增强特征矩阵形状：{X_enhanced.shape}")

# === 建立融合模型 ===
print("\n=== 建立融合模型 ===")

# 基础模型：使用基础特征
base_purchase_model = LinearRegression()
base_redeem_model = LinearRegression()

# 增强模型：使用基础特征+易卦特征
enhanced_purchase_model = LinearRegression()
enhanced_redeem_model = LinearRegression()

# 残差修正模型：使用随机森林
residual_purchase_model = RandomForestRegressor(n_estimators=50, random_state=42)
residual_redeem_model = RandomForestRegressor(n_estimators=50, random_state=42)

# 训练基础模型
print("训练基础模型...")
base_purchase_model.fit(X, y_purchase)
base_redeem_model.fit(X, y_redeem)

# 训练增强模型
print("训练易卦增强模型...")
enhanced_purchase_model.fit(X_enhanced, y_purchase)
enhanced_redeem_model.fit(X_enhanced, y_redeem)

# 计算增强模型的残差并训练修正模型
print("训练残差修正模型...")
enhanced_purchase_pred = enhanced_purchase_model.predict(X_enhanced)
enhanced_redeem_pred = enhanced_redeem_model.predict(X_enhanced)

purchase_residuals = y_purchase - enhanced_purchase_pred
redeem_residuals = y_redeem - enhanced_redeem_pred

residual_purchase_model.fit(X_enhanced, purchase_residuals)
residual_redeem_model.fit(X_enhanced, redeem_residuals)

# === 模型评估 ===
print("\n=== 模型评估 ===")

# 基础模型评估
base_purchase_pred_train = base_purchase_model.predict(X)
base_redeem_pred_train = base_redeem_model.predict(X)

base_purchase_mae = mean_absolute_error(y_purchase, base_purchase_pred_train)
base_redeem_mae = mean_absolute_error(y_redeem, base_redeem_pred_train)

print(f"基础模型结果：")
print(f"申购金额 MAE: {base_purchase_mae:,.0f}")
print(f"赎回金额 MAE: {base_redeem_mae:,.0f}")
print(f"申购金额 R²: {base_purchase_model.score(X, y_purchase):.4f}")
print(f"赎回金额 R²: {base_redeem_model.score(X, y_redeem):.4f}")

# 增强模型评估
enhanced_purchase_mae = mean_absolute_error(y_purchase, enhanced_purchase_pred)
enhanced_redeem_mae = mean_absolute_error(y_redeem, enhanced_redeem_pred)

print(f"\n易卦增强模型结果：")
print(f"申购金额 MAE: {enhanced_purchase_mae:,.0f}")
print(f"赎回金额 MAE: {enhanced_redeem_mae:,.0f}")
print(f"申购金额 R²: {enhanced_purchase_model.score(X_enhanced, y_purchase):.4f}")
print(f"赎回金额 R²: {enhanced_redeem_model.score(X_enhanced, y_redeem):.4f}")

# 最终融合模型评估
final_purchase_pred = enhanced_purchase_pred + residual_purchase_model.predict(X_enhanced)
final_redeem_pred = enhanced_redeem_pred + residual_redeem_model.predict(X_enhanced)

final_purchase_mae = mean_absolute_error(y_purchase, final_purchase_pred)
final_redeem_mae = mean_absolute_error(y_redeem, final_redeem_pred)

print(f"\n最终融合模型结果：")
print(f"申购金额 MAE: {final_purchase_mae:,.0f}")
print(f"赎回金额 MAE: {final_redeem_mae:,.0f}")

# === 生成2014年9月预测 ===
print("\n=== 生成2014年9月预测 ===")

# 创建9月的日期范围
sept_dates = pd.date_range('2014-09-01', '2014-09-30', freq='D')
sept_df = pd.DataFrame({'report_date': sept_dates})

# 添加基础周期因子
sept_df['weekday'] = sept_df['report_date'].dt.dayofweek
sept_df['day'] = sept_df['report_date'].dt.day

# 创建基础预测特征矩阵
sept_weekday_dummies = pd.get_dummies(sept_df['weekday'], prefix='weekday')
# 确保所有weekday列都存在
for col in weekday_dummies.columns:
    if col not in sept_weekday_dummies.columns:
        sept_weekday_dummies[col] = 0

# 重新排序列以匹配训练数据
sept_weekday_dummies = sept_weekday_dummies[weekday_dummies.columns]

sept_day_normalized = (sept_df['day'] - daily_data['day'].mean()) / daily_data['day'].std()

X_sept = pd.concat([sept_weekday_dummies, sept_day_normalized.rename('day_norm')], axis=1)

# 使用基础模型进行初步预测
initial_purchase_pred = base_purchase_model.predict(X_sept)
initial_redeem_pred = base_redeem_model.predict(X_sept)

# 为9月数据生成易卦特征（使用初步预测值）
sept_iching_features = generate_iching_features(
    initial_purchase_pred,
    initial_redeem_pred,
    sept_dates
)

# 创建增强特征矩阵
X_sept_enhanced = pd.concat([X_sept, sept_iching_features], axis=1)

# 使用增强模型进行预测
enhanced_purchase_pred_sept = enhanced_purchase_model.predict(X_sept_enhanced)
enhanced_redeem_pred_sept = enhanced_redeem_model.predict(X_sept_enhanced)

# 使用残差修正模型进行最终调整
residual_purchase_pred = residual_purchase_model.predict(X_sept_enhanced)
residual_redeem_pred = residual_redeem_model.predict(X_sept_enhanced)

# 最终预测结果
final_purchase_pred_sept = enhanced_purchase_pred_sept + residual_purchase_pred
final_redeem_pred_sept = enhanced_redeem_pred_sept + residual_redeem_pred

# 创建预测结果DataFrame
predictions = pd.DataFrame({
    'report_date': sept_dates.strftime('%Y%m%d'),
    'purchase': final_purchase_pred_sept.astype(int),
    'redeem': final_redeem_pred_sept.astype(int),
    'weekday': sept_df['weekday'],
    'day': sept_df['day']
})

print(f"\n9月预测结果预览：")
print(predictions[['report_date', 'purchase', 'redeem']].head(10))

print(f"\n预测统计：")
print(f"申购金额预测范围：{final_purchase_pred_sept.min():,.0f} - {final_purchase_pred_sept.max():,.0f}")
print(f"申购金额预测平均：{final_purchase_pred_sept.mean():,.0f}")
print(f"赎回金额预测范围：{final_redeem_pred_sept.min():,.0f} - {final_redeem_pred_sept.max():,.0f}")
print(f"赎回金额预测平均：{final_redeem_pred_sept.mean():,.0f}")

# === 易卦解释功能 ===
def interpret_iching(date, purchase_val, redeem_val):
    """生成卦象解释"""
    # 计算x, y, z变量数
    x = int(purchase_val / 1e6)  # 申购金额（下卦）
    y = date.day + date.month * 10 + date.year  # 时间因子（上卦）
    z = int(redeem_val / 1e5) + date.day * date.month  # 赎回+时间（变爻）

    # 获取原卦
    x_trigram = get_trigram(x)
    y_trigram = get_trigram(y)

    # 计算变爻
    changing_line = calculate_changing_line(z)

    # 计算变卦
    if changing_line <= 3:
        changed_x = get_changed_trigram(x_trigram, changing_line)
        changed_y = y_trigram
    else:
        changed_x = x_trigram
        changed_y = get_changed_trigram(y_trigram, changing_line)

    interpretation = {
        "日期": date.strftime("%Y-%m-%d"),
        "原卦": f"{x_trigram[0]}(下) + {y_trigram[0]}(上)",
        "变卦": f"{changed_x[0]}(下) + {changed_y[0]}(上)",
        "变爻": f"{changing_line}爻",
        "原卦趋势": "上升" if (x_trigram[1] + y_trigram[1]) > 0 else "平稳" if (x_trigram[1] + y_trigram[1]) == 0 else "下降",
        "变卦趋势": "上升" if (changed_x[1] + changed_y[1]) > 0 else "平稳" if (changed_x[1] + changed_y[1]) == 0 else "下降",
        "建议": []
    }

    # 根据卦象添加建议
    if x_trigram[0] == "乾" or y_trigram[0] == "乾":
        interpretation["建议"].append("强势卦象，资金流动活跃")
    if "坤" in (x_trigram[0], y_trigram[0]):
        interpretation["建议"].append("平稳卦象，适合保守策略")
    if changing_line in (1, 6):
        interpretation["建议"].append("关键变爻，注意趋势转折")
    if x_trigram[1] != changed_x[1] or y_trigram[1] != changed_y[1]:
        interpretation["建议"].append("卦气发生变化，关注市场动向")

    return interpretation

# 为9月关键日期添加易卦解释
key_dates = ['2014-09-08', '2014-09-15', '2014-09-22', '2014-09-30']
print("\n=== 关键日期易卦分析 ===")
for date_str in key_dates:
    date = pd.to_datetime(date_str)
    idx = predictions[predictions['report_date'] == date.strftime('%Y%m%d')].index[0]
    purchase_val = predictions.loc[idx, 'purchase']
    redeem_val = predictions.loc[idx, 'redeem']

    interpretation = interpret_iching(date, purchase_val, redeem_val)
    print(f"\n{interpretation['日期']}:")
    print(f"  原卦: {interpretation['原卦']}")
    print(f"  变卦: {interpretation['变卦']}")
    print(f"  变爻: {interpretation['变爻']}")
    print(f"  原卦趋势: {interpretation['原卦趋势']}")
    print(f"  变卦趋势: {interpretation['变卦趋势']}")
    print(f"  建议: {', '.join(interpretation['建议'])}")

# 保存预测结果
output_file = os.path.join(script_dir, 'iching_enhanced_predictions.csv')
predictions[['report_date', 'purchase', 'redeem']].to_csv(
    output_file, index=False, header=False, encoding='utf-8-sig'
)
print(f"\n易卦增强预测结果已保存到：{output_file}")

# 保存详细分析结果
detailed_output_file = os.path.join(script_dir, 'iching_detailed_predictions.csv')
predictions.to_csv(detailed_output_file, index=False, encoding='utf-8-sig')
print(f"详细预测结果已保存到：{detailed_output_file}")

# 创建预测可视化
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

# 申购金额对比图
ax1.plot(daily_data['report_date'], daily_data['total_purchase_amt'],
         'o-', label='申购-真实值', color='blue', alpha=0.7, markersize=3)
ax1.plot(daily_data['report_date'], final_purchase_pred,
         's-', label='申购-易卦模型拟合', color='lightblue', alpha=0.7, markersize=3)
ax1.plot(sept_dates, final_purchase_pred_sept,
         '^-', label='申购-9月易卦预测', color='red', markersize=4)

ax1.set_title('申购金额：易卦增强模型训练期拟合 vs 9月预测', fontsize=14, fontweight='bold')
ax1.set_ylabel('申购金额（元）')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 赎回金额对比图
ax2.plot(daily_data['report_date'], daily_data['total_redeem_amt'],
         'o-', label='赎回-真实值', color='orange', alpha=0.7, markersize=3)
ax2.plot(daily_data['report_date'], final_redeem_pred,
         's-', label='赎回-易卦模型拟合', color='gold', alpha=0.7, markersize=3)
ax2.plot(sept_dates, final_redeem_pred_sept,
         '^-', label='赎回-9月易卦预测', color='red', markersize=4)

ax2.set_title('赎回金额：易卦增强模型训练期拟合 vs 9月预测', fontsize=14, fontweight='bold')
ax2.set_xlabel('日期')
ax2.set_ylabel('赎回金额（元）')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'iching_enhanced_predictions_plot.png'), dpi=300, bbox_inches='tight')
print(f"易卦增强预测可视化图已保存到：{os.path.join(script_dir, 'iching_enhanced_predictions_plot.png')}")
plt.show()

print("\n=== 易卦增强资金流预测分析完成 ===")
print("生成的文件：")
print("1. 易卦增强预测结果（标准格式）：iching_enhanced_predictions.csv")
print("2. 详细预测结果：iching_detailed_predictions.csv")
print("3. 易卦增强预测可视化图：iching_enhanced_predictions_plot.png")
print("\n易卦预测原理说明：")
print("- x变量数（下卦）：申购金额，代表内在资金需求")
print("- y变量数（上卦）：时间因子，代表外在市场环境")
print("- z变量数（变爻）：赎回金额+时间组合，代表变化动力")
print("- 通过原卦和变卦的卦气趋势，预测资金流向变化")