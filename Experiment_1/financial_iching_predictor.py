import pandas as pd
import numpy as np
from datetime import datetime
import torch
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

class IChingFinancialOracle:
    """易经金融预测系统"""
    def __init__(self):
        self.trigrams = {
            0: {'name': '坤', 'nature': '阴', 'trend': '平', 'value': 8},
            1: {'name': '乾', 'nature': '阳', 'trend': '平', 'value': 1},
            2: {'name': '兑', 'nature': '阴', 'trend': '降', 'value': 2},
            3: {'name': '离', 'nature': '阴', 'trend': '升', 'value': 3},
            4: {'name': '震', 'nature': '阳', 'trend': '升', 'value': 4},
            5: {'name': '巽', 'nature': '阴', 'trend': '降', 'value': 5},
            6: {'name': '坎', 'nature': '阳', 'trend': '降', 'value': 6},
            7: {'name': '艮', 'nature': '阳', 'trend': '升', 'value': 7}
        }
    
    def _sum_date(self, date_str):
        """计算日期的数字和"""
        date_str = str(date_str)
        return sum(int(c) if c != '0' else 8 for c in date_str)
    
    def get_trigram(self, num):
        """获取卦象"""
        remainder = num % 8
        return self.trigrams[remainder]
    
    def get_changing_line(self, num):
        """获取变爻位置(1-6)"""
        return num % 6 or 6
    
    def predict_trend(self, date_str, purchase, redeem):
        """预测资金流向趋势"""
        # 计算原卦
        date_sum = self._sum_date(date_str)
        x = self.get_trigram(date_sum)  # 下卦(日期)
        y_purchase = self.get_trigram(purchase)  # 上卦(申购)
        y_redeem = self.get_trigram(redeem)  # 上卦(赎回)
        
        # 计算变爻
        diff = abs(purchase - redeem)
        z_purchase = self.get_changing_line(diff)
        z_redeem = self.get_changing_line(diff)
        
        # 申购预测
        purchase_trend = self._get_final_trend(x, y_purchase, z_purchase)
        
        # 赎回预测
        redeem_trend = self._get_final_trend(x, y_redeem, z_redeem)
        
        return purchase_trend, redeem_trend
    
    def _get_final_trend(self, x, y, z):
        """获取最终趋势"""
        original_trend = (x['trend'], y['trend'])
        
        # 确定变爻影响上卦还是下卦
        if z <= 3:  # 变下卦
            changing_trigram = x
        else:  # 变上卦
            changing_trigram = y
        
        # 找到变化后的卦(简单实现：按value+1循环)
        new_value = (changing_trigram['value'] % 8) + 1
        changed_trigram = self.get_trigram(new_value)
        
        # 确定最终趋势
        if z <= 3:  # 下卦变化
            final_trend = (changed_trigram['trend'], y['trend'])
        else:  # 上卦变化
            final_trend = (x['trend'], changed_trigram['trend'])
        
        # 趋势判断规则
        if final_trend[0] == final_trend[1]:
            return final_trend[0]  # 上下同趋势
        elif final_trend[0] == '平':
            return final_trend[1]  # 下平看上
        elif final_trend[1] == '平':
            return final_trend[0]  # 上平看下
        else:
            return '平'  # 上下趋势相反

class FinancialPredictor:
    """资金预测系统(整合易经和机器学习)"""
    def __init__(self):
        self.oracle = IChingFinancialOracle()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # XGBoost参数
        self.xgb_params = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 5,
            'random_state': 42,
            'n_jobs': -1
        }
        if torch.cuda.is_available():
            self.xgb_params['tree_method'] = 'gpu_hist'
    
    def load_data(self, filepath):
        """加载数据"""
        df = pd.read_csv(filepath)
        df['report_date'] = pd.to_datetime(df['report_date'], format='%Y%m%d')
        return df
    
    def preprocess_data(self, df):
        """数据预处理"""
        daily_data = df.groupby('report_date').agg({
            'total_purchase_amt': 'sum',
            'total_redeem_amt': 'sum'
        }).reset_index()
        
        # 添加时间特征
        daily_data['day_of_week'] = daily_data['report_date'].dt.dayofweek
        daily_data['month'] = daily_data['report_date'].dt.month
        
        # 添加滞后特征
        for lag in [1, 2, 3, 7]:
            daily_data[f'purchase_lag_{lag}'] = daily_data['total_purchase_amt'].shift(lag)
            daily_data[f'redeem_lag_{lag}'] = daily_data['total_redeem_amt'].shift(lag)
        
        return daily_data.dropna()
    
    def train_models(self, daily_data):
        """训练模型"""
        feature_cols = [col for col in daily_data.columns 
                       if col not in ['report_date', 'total_purchase_amt', 'total_redeem_amt']]
        
        # 申购模型
        self.purchase_model = XGBRegressor(**self.xgb_params)
        self.purchase_model.fit(daily_data[feature_cols], daily_data['total_purchase_amt'])
        
        # 赎回模型
        self.redeem_model = XGBRegressor(**self.xgb_params)
        self.redeem_model.fit(daily_data[feature_cols], daily_data['total_redeem_amt'])
    
    def hybrid_predict(self, daily_data, n_days=30):
        """混合预测(机器学习+易经)"""
        last_day = daily_data.iloc[-1].copy()
        predictions = []
        
        for i in range(1, n_days+1):
            # 准备特征
            current_date = last_day['report_date'] + pd.Timedelta(days=i)
            features = last_day.copy()
            
            # 更新日期相关特征
            features['report_date'] = current_date
            features['day_of_week'] = current_date.dayofweek
            features['month'] = current_date.month
            
            # 机器学习预测
            ml_purchase = self.purchase_model.predict([features.drop(['report_date', 'total_purchase_amt', 'total_redeem_amt'])])[0]
            ml_redeem = self.redeem_model.predict([features.drop(['report_date', 'total_purchase_amt', 'total_redeem_amt'])])[0]
            
            # 易经预测
            date_str = current_date.strftime('%Y%m%d')
            iching_purchase_trend, iching_redeem_trend = self.oracle.predict_trend(
                date_str, ml_purchase, ml_redeem)
            
            # 应用易经调整
            adjustment_factor = 0.05  # 5%的调整幅度
            if iching_purchase_trend == '升':
                final_purchase = ml_purchase * (1 + adjustment_factor)
            elif iching_purchase_trend == '降':
                final_purchase = ml_purchase * (1 - adjustment_factor)
            else:
                final_purchase = ml_purchase
            
            if iching_redeem_trend == '升':
                final_redeem = ml_redeem * (1 + adjustment_factor)
            elif iching_redeem_trend == '降':
                final_redeem = ml_redeem * (1 - adjustment_factor)
            else:
                final_redeem = ml_redeem
            
            predictions.append({
                'date': current_date,
                'ml_purchase': ml_purchase,
                'ml_redeem': ml_redeem,
                'iching_purchase_trend': iching_purchase_trend,
                'iching_redeem_trend': iching_redeem_trend,
                'final_purchase': final_purchase,
                'final_redeem': final_redeem
            })
            
            # 更新滞后特征(简单实现)
            last_day['purchase_lag_1'] = final_purchase
            last_day['redeem_lag_1'] = final_redeem
        
        return pd.DataFrame(predictions)

# 使用示例
if __name__ == "__main__":
    # 初始化预测器
    predictor = FinancialPredictor()
    
    # 加载和处理数据
    print("加载和处理数据...")
    df = predictor.load_data(r'D:\sep_venv\AI_venv\AI_Lesson21_codes\CASE-资金流入流出预测\user_balance_table.csv')
    daily_data = predictor.preprocess_data(df)
    
    # 训练模型
    print("训练模型...")
    predictor.train_models(daily_data)
    
    # 进行混合预测
    print("进行混合预测...")
    predictions = predictor.hybrid_predict(daily_data, n_days=30)
    
    # 保存结果
    predictions.to_csv('hybrid_predictions.csv', index=False)
    print("预测结果已保存到 hybrid_predictions.csv")
    
    # 评估准确率
    actual = daily_data[['report_date', 'total_purchase_amt', 'total_redeem_amt']]
    actual.columns = ['date', 'actual_purchase', 'actual_redeem']
    evaluation = pd.merge(predictions, actual, on='date', how='left')
    
    if not evaluation.empty:
        purchase_mape = np.mean(np.abs(evaluation['actual_purchase'] - evaluation['final_purchase']) / evaluation['actual_purchase'])
        redeem_mape = np.mean(np.abs(evaluation['actual_redeem'] - evaluation['final_redeem']) / evaluation['actual_redeem'])
        print(f"\n预测准确率评估:")
        print(f"申购平均绝对百分比误差(MAPE): {purchase_mape:.2%}")
        print(f"赎回平均绝对百分比误差(MAPE): {redeem_mape:.2%}")
        