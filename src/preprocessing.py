# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def standardize_features(df, feature_columns):
    """
    對指定的特徵列進行 Z-score 標準化。

    Args:
        df (pandas.DataFrame): 包含特徵的 DataFrame。
        feature_columns (list): 需要標準化的特徵列名稱列表。

    Returns:
        pandas.DataFrame: 包含標準化後特徵的 DataFrame。
        sklearn.preprocessing.StandardScaler: 訓練好的 Scaler 物件。
    """
    if df is None or df.empty:
        print("錯誤：輸入的 DataFrame 為空或 None。")
        return None, None
        
    data_to_scale = df[feature_columns].copy() # 使用.copy() 避免 SettingWithCopyWarning
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_scale)
    
    scaled_df = pd.DataFrame(scaled_data, columns=feature_columns, index=df.index)
    
    # 將非特徵列（如 'id'）與標準化後的特徵合併
    df_processed = df.drop(columns=feature_columns).join(scaled_df)
    
    print("特徵已成功進行 Z-score 標準化。")
    return df_processed, scaler

def apply_pca(df_scaled_features, n_components=None, explained_variance_ratio=0.95):
    """
    對標準化後的特徵應用 PCA 降維。

    Args:
        df_scaled_features (pandas.DataFrame): 包含標準化後特徵的 DataFrame。
        n_components (int, optional): 要保留的主成分數量。如果為 None，則根據 explained_variance_ratio 決定。
        explained_variance_ratio (float, optional): 當 n_components 為 None 時，選擇能解釋至少這麼多方差的主成分數量。

    Returns:
        pandas.DataFrame: 降維後的特徵 DataFrame。
        sklearn.decomposition.PCA: 訓練好的 PCA 物件。
    """
    if df_scaled_features is None or df_scaled_features.empty:
        print("錯誤：輸入的 DataFrame 為空或 None。")
        return None, None

    if n_components is None:
        pca_temp = PCA(n_components=explained_variance_ratio, random_state=42)
        pca_temp.fit(df_scaled_features)
        n_components_selected = pca_temp.n_components_
        print(f"PCA：根據 {explained_variance_ratio*100}% 的解釋方差比例，選擇了 {n_components_selected} 個主成分。")
    else:
        n_components_selected = n_components
        print(f"PCA：選擇了 {n_components_selected} 個主成分。")

    pca = PCA(n_components=n_components_selected, random_state=42)
    principal_components = pca.fit_transform(df_scaled_features)
    
    pca_columns = [f'PC{i+1}' for i in range(principal_components.shape[1])]
    df_pca = pd.DataFrame(data=principal_components, columns=pca_columns, index=df_scaled_features.index)
    
    print(f"PCA 降維完成。解釋的總方差比例: {pca.explained_variance_ratio_.sum():.4f}")
    return df_pca, pca

if __name__ == '__main__':
    # 示例數據
    data = {'id': [1, 2, 3, 4, 5],
            '1': [6, 7, 8, 9, 10],
            '2': [1, 2, 1, 3, 2],
            '3': [1, 2, 1, 3, 2],
            '4': [1, 2, 1, 3, 2]}
    sample_df = pd.DataFrame(data)
    feature_cols = ['1', '2', '3', '4']

    print("原始數據：")
    print(sample_df)

    # 標準化
    df_standardized, _ = standardize_features(sample_df.copy(), feature_cols) #傳遞副本以避免修改原始 df
    if df_standardized is not None:
        print("\n標準化後的數據：")
        print(df_standardized)

        # PCA 應用 (假設這是高維數據需要降維)
        # 提取標準化後的特徵部分進行 PCA
        features_for_pca = df_standardized[feature_cols]
        df_pca_result, _ = apply_pca(features_for_pca, n_components=2)
        if df_pca_result is not None:
            print("\nPCA 降維後的數據 (前2個主成分)：")
            print(df_pca_result.head())