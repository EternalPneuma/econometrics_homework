import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns # 用于更美观的混淆矩阵

def run_xgboost_analysis():
    print(">>> 启动 XGBoost 回测流程...")
    
    # 1. 读取基础数据
    try:
        df = pd.read_csv('data.csv')
        if 'bond_code' in df.columns:
            df['bond_code'] = df['bond_code'].astype(str).str.strip().str.upper()
    except FileNotFoundError:
        print("错误：未找到 data.csv")
        return

    if 'event_type' in df.columns:
        df['Y'] = df['event_type'].map({'NO': 0, 'PROPOSAL': 1, 'IGNORE': 0})
    else:
        print("错误：数据缺少 event_type 列")
        return

    # 2. 读取行情数据
    try:
        print("正在读取行情数据...")
        market_df = pd.read_csv('market.csv')
        market_df['date'] = pd.to_datetime(market_df['date'])
        if 'bond_code' in market_df.columns:
            market_df['bond_code'] = market_df['bond_code'].astype(str).str.strip().str.upper()
        market_df = market_df.sort_values(['bond_code', 'date'])
    except Exception as e:
        print(f"读取 market.csv 失败: {e}")
        return

    # 3. 数据清洗 (W1 去极值)
    if 'W1' in df.columns:
        df['W1'] = pd.to_numeric(df['W1'], errors='coerce')
        lower = df['W1'].quantile(0.05)
        upper = df['W1'].quantile(0.95)
        df = df[(df['W1'] >= lower) & (df['W1'] <= upper)]
    
    features = ['W1', 'W2', 'W3', 'W4a', 'W5', 'W6', 'W7a', 'W8', 'W9']
    
    df_model = df[['event_date', 'Y', 'bond_code'] + features].dropna()
    df_model['event_date'] = pd.to_datetime(df_model['event_date'])
    df_model = df_model.sort_values('event_date').reset_index(drop=True)
    
    # 4. 划分训练/测试集 (2024-08-01)
    split_date = pd.to_datetime('2024-08-01')
    train_df = df_model[df_model['event_date'] < split_date]
    test_df = df_model[df_model['event_date'] >= split_date]
    
    if len(train_df) == 0 or len(test_df) == 0:
        print("错误：数据集划分为空")
        return

    X_train = train_df[features]
    y_train = train_df['Y']
    X_test = test_df[features]
    y_test = test_df['Y']
    
    # 类别权重
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1
    
    print(f"时间切分: {split_date.date()} | 训练集: {len(train_df)} | 测试集: {len(test_df)}")

    # 5. 模型训练
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=200,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='auc'
    )
    model.fit(X_train, y_train)
    
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_prob)
    print(f"测试集 AUC: {auc_score:.4f}")
    
    # 6. 回测 (T-5 买入, T+3 卖出)
    print(">>> 执行策略回测...")
    
    threshold = 0.5 
    results = test_df.copy().reset_index(drop=True)
    results['prob_proposal'] = y_pred_prob
    results['signal_buy'] = (y_pred_prob > threshold).astype(int)
    
    market_groups = market_df.groupby('bond_code')
    
    returns = []
    notes = [] 
    
    for idx, row in results.iterrows():
        if row['signal_buy'] == 0:
            returns.append(0.0)
            notes.append("无信号")
            continue
            
        code = row['bond_code']
        event_dt = row['event_date']
        
        if code not in market_groups.groups:
            returns.append(0.0)
            notes.append("缺行情")
            continue
            
        bond_market = market_groups.get_group(code)
        dates_list = bond_market['date'].values
        price_values = bond_market['close_price'].values
        
        try:
            target_dt = np.datetime64(event_dt)
            curr_idx = np.searchsorted(dates_list, target_dt)
            
            # T-5 买入, T+3 卖出
            entry_idx = curr_idx - 5
            exit_idx = curr_idx + 3
            
            if entry_idx < 0:
                returns.append(0.0)
                notes.append("上市不足")
                continue
                
            if exit_idx >= len(dates_list):
                exit_idx = len(dates_list) - 1 
            
            p_buy = price_values[entry_idx]
            p_sell = price_values[exit_idx]
            ret = (p_sell - p_buy) / p_buy
            
            returns.append(ret)
            notes.append("成交")
            
        except Exception as e:
            returns.append(0.0)
            notes.append(f"错误:{e}")

    results['return'] = returns
    results['note'] = notes
    # 这里保留单笔交易的累加，仅用于导出 CSV 明细
    results['cumulative_return'] = results['return'].cumsum()
    
    # 统计
    trades = results[results['signal_buy'] == 1]
    valid_trades = trades[trades['note'] == "成交"]
    
    if len(valid_trades) > 0:
        avg_ret = valid_trades['return'].mean()
        win_rate = (valid_trades['return'] > 0).mean()
        total_ret = valid_trades['return'].sum()
    else:
        avg_ret, win_rate, total_ret = 0, 0, 0
        
    print(f"回测阈值: {threshold}")
    print(f"触发信号: {len(trades)}")
    print(f"有效成交: {len(valid_trades)}")
    print(f"平均单笔收益: {avg_ret:.4f}")
    print(f"胜率: {win_rate:.2%}")
    print(f"策略累计收益 (单利): {total_ret:.4f}")
    
    results.to_csv('real_market_backtest.csv', index=False)
    print("结果已保存至 real_market_backtest.csv")

    # --- 7. 全面的论文图表生成 ---
    print("\n>>> 正在生成论文专用图表...")
    
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 7.1 特征重要性图 (Feature Importance)
    # -------------------------------------
    try:
        plt.figure(figsize=(10, 6))
        # 获取特征重要性
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1] # 降序排列
        
        plt.title('Feature Importances (XGBoost)', fontsize=14)
        plt.bar(range(len(importances)), importances[indices], align='center', color='#1f77b4')
        plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45)
        plt.ylabel('Importance Score')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300)
        print("[图表1] 特征重要性图已保存至 feature_importance.png")
    except Exception as e:
        print(f"特征重要性绘图失败: {e}")

    # 7.2 ROC 曲线图 (ROC Curve)
    # -------------------------------------
    try:
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig('roc_curve.png', dpi=300)
        print("[图表2] ROC曲线图已保存至 roc_curve.png")
    except Exception as e:
        print(f"ROC绘图失败: {e}")

    # 7.3 混淆矩阵图 (Confusion Matrix)
    # -------------------------------------
    try:
        # 使用当前阈值生成二分类预测
        y_pred_bin = (y_pred_prob > threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred_bin)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Pred: No', 'Pred: Yes'],
                    yticklabels=['Actual: No', 'Actual: Yes'])
        plt.title(f'Confusion Matrix (Threshold={threshold})', fontsize=14)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300)
        print("[图表3] 混淆矩阵图已保存至 confusion_matrix.png")
    except Exception as e:
        # 如果没有安装seaborn，回退到简单绘图
        print(f"混淆矩阵绘图失败 (可能缺少seaborn): {e}")

    # 7.4 累计收益图 (优化版) & 最大回撤计算
    # -------------------------------------
    try:
        plt.figure(figsize=(10, 6))
        
        # 按日期聚合收益
        plot_data = results.groupby('event_date')['return'].sum().reset_index()
        plot_data = plot_data.sort_values('event_date')
        
        # 计算累计收益
        plot_data['cumulative_return'] = plot_data['return'].cumsum()
        
        # 计算最大回撤 (Max Drawdown)
        # 简单单利回撤：当前累计收益 - 之前的最大累计收益
        cum_ret = plot_data['cumulative_return']
        running_max = np.maximum.accumulate(cum_ret)
        drawdown = cum_ret - running_max
        max_drawdown = drawdown.min()
        
        print(f"\n[风险指标] 策略最大回撤: {max_drawdown:.4f}")
        
        # 绘制曲线
        plt.plot(plot_data['event_date'], plot_data['cumulative_return'], 
                 label='Strategy Return', color='#d62728', linewidth=2)
        
        # 标记最大回撤区间 (可选)
        plt.fill_between(plot_data['event_date'], plot_data['cumulative_return'], running_max, 
                         color='gray', alpha=0.1, label='Drawdown Area')

        plt.title('Strategy Cumulative Return & Drawdown', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Return (Simple Interest)', fontsize=12)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.savefig('cumulative_return.png', dpi=300, bbox_inches='tight')
        print("[图表4] 累计收益与回撤图已保存至 cumulative_return.png")
        
    except Exception as e:
        print(f"收益绘图失败: {e}")

if __name__ == "__main__":
    run_xgboost_analysis()