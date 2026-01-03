import pandas as pd
import statsmodels.api as sm
import numpy as np

def run_logit_analysis():
    print("正在读取数据...")
    try:
        df = pd.read_csv('data.csv')
    except FileNotFoundError:
        print("错误：未找到 data.csv 文件。")
        return

    # 1. 数据预处理
    # -------------------------------------------------------
    # 映射目标变量 Y: PROPOSAL -> 1, 其他(NO/IGNORE) -> 0
    if 'event_type' in df.columns:
        df['Y'] = df['event_type'].map({'NO': 0, 'PROPOSAL': 1, 'IGNORE': 0})
        # 再次确认映射是否有误（防止只有NaN）
        if df['Y'].isnull().all():
             print("错误：event_type 列无法正确映射，请检查数据中的类别名称是否为 NO/PROPOSAL")
             return
    else:
        print("错误：缺少 event_type 列")
        return

    # -------------------------------------------------------
    # 新增步骤：去除 W1 (财务状况) 的极端值 (Top/Bottom 5%)
    # -------------------------------------------------------
    if 'W1' in df.columns:
        # 确保 W1 是数值型
        df['W1'] = pd.to_numeric(df['W1'], errors='coerce')
        
        # 计算分位数
        lower_bound = df['W1'].quantile(0.05)
        upper_bound = df['W1'].quantile(0.95)
        
        print("\n" + "-"*40)
        print("正在进行 W1 极端值处理 (去头去尾 5%)...")
        print(f"保留范围: {lower_bound:.4f} ~ {upper_bound:.4f}")
        
        original_len = len(df)
        # 仅保留在 5% - 95% 之间的数据
        df = df[(df['W1'] >= lower_bound) & (df['W1'] <= upper_bound)]
        new_len = len(df)
        
        print(f"原始样本: {original_len} -> 处理后样本: {new_len}")
        print(f"已剔除 {original_len - new_len} 个极端观测值。")
        print("-"*40)
    else:
        print("警告：未找到 W1 列，跳过去极值步骤。")

    # 定义两组特征
    # 全模型：包含所有变量
    features_full = ['W1', 'W2', 'W3', 'W4a', 'W5', 'W6', 'W7a', 'W8', 'W9']
    # 精简模型：加入已清洗的 W1
    features_simple = ['W1', 'W4a', 'W6', 'W7a', 'W8', 'W9']

    # 2. 模型 A: 全模型 (Full Model)
    # -------------------------------------------------------
    print("\n" + "="*60)
    print("模型 A: 全变量 Logit 模型 (已去极值)")
    print(f"包含变量: {features_full}")
    print("="*60)

    # 准备数据
    cols_to_use = features_full + ['Y']
    # 剔除空值
    df_full = df[cols_to_use].dropna()
    
    # 确保所有列都是数值型
    df_full = df_full.apply(pd.to_numeric, errors='coerce').dropna()

    if df_full.empty:
        print("错误：清洗数据后样本为空，请检查数据格式。")
        return

    X_full = df_full[features_full]
    y_full = df_full['Y']
    
    # 添加截距
    X_full = sm.add_constant(X_full)

    try:
        model_full = sm.Logit(y_full, X_full)
        result_full = model_full.fit(disp=0, method='bfgs', maxiter=1000)
        print(result_full.summary())
        
        # 记录 AIC/BIC 以便比较
        aic_full = result_full.aic
        print(f"全模型 AIC: {aic_full:.4f}")

    except Exception as e:
        print(f"全模型回归失败: {e}")

    # 3. 模型 B: 精简模型 (Simplified Model)
    # -------------------------------------------------------
    print("\n" + "="*60)
    print("模型 B: 精简 Logit 模型 (已去极值，加入 W1)")
    print(f"包含变量: {features_simple}")
    print("="*60)

    cols_to_use_simple = features_simple + ['Y']
    df_simple = df[cols_to_use_simple].dropna()
    df_simple = df_simple.apply(pd.to_numeric, errors='coerce').dropna()

    X_simple = df_simple[features_simple]
    y_simple = df_simple['Y']
    X_simple = sm.add_constant(X_simple)

    try:
        model_simple = sm.Logit(y_simple, X_simple)
        result_simple = model_simple.fit(disp=0, method='bfgs', maxiter=1000)
        print(result_simple.summary())
        
        aic_simple = result_simple.aic
        print(f"精简模型 AIC: {aic_simple:.4f}")
        
        # 比较 AIC (越小越好)
        if 'aic_full' in locals():
            print("\n---------------- 模型对比 ----------------")
            print(f"全模型 AIC: {aic_full:.2f} vs 精简模型 AIC: {aic_simple:.2f}")
            if aic_simple < aic_full:
                print("结论: 精简模型 AIC 更低，说明去除冗余变量后模型更优。")
            else:
                print("结论: 全模型 AIC 更低，说明可能有些被剔除的变量仍有贡献，或者 W1 与其他变量存在协同效应。")

        # 4. 输出优势比 (Odds Ratio) - 仅针对精简模型
        # -------------------------------------------------------
        print("\n" + "="*60)
        print("精简模型 - 经济意义解读 (Odds Ratios)")
        print("="*60)
        
        params = result_simple.params
        conf = result_simple.conf_int()
        conf['Odds Ratio'] = params
        conf.columns = ['2.5%', '97.5%', 'OR']
        
        # 计算 OR = exp(coef)
        or_df = np.exp(conf)
        print(or_df[['OR', '2.5%', '97.5%']])
        print("\n解读提示：")
        print("- OR > 1 (如 W6): 说明变量增加，提议下修概率增加。")
        print("- OR < 1 (如 W8): 说明变量增加，提议下修概率降低。")

    except Exception as e:
        print(f"精简模型回归失败: {e}")

if __name__ == "__main__":
    run_logit_analysis()