# src/main.py
import pandas as pd
import numpy as np
from data_loader import load_public_data, load_private_data_placeholder
from preprocessing import standardize_features, apply_pca
from clustering_algorithms import apply_kmeans, apply_agglomerative_clustering, apply_dbscan, apply_gmm
from classification_algorithms import train_logistic_regression, train_svm, train_random_forest
from evaluation_metrics import evaluate_clustering_nmi, evaluate_clustering_internal, evaluate_classification
from sklearn.model_selection import train_test_split

def run_analysis(data_source, data_name, n_dimensions, is_public_data=True, feature_columns=None):
    """
    執行完整的數據分析流程：加載、預處理、聚類、分類、評估。
    """
    print(f"\n{'='*20} 開始分析: {data_name} ({n_dimensions} 維) {'='*20}")

    # 1. 加載數據
    if is_public_data:
        df = load_public_data(data_source)
        if df is None:
            return
        if feature_columns is None: # 默認公共數據集的特徵列
            feature_columns = ['1', '2', '3', '4']
    else: # 私人數據集 (概念性)
        df = load_private_data_placeholder() # 這裡返回 None，或您可以修改為生成隨機數據
        if df is None:
            print(f"未提供私人數據集 {data_name}，跳過分析。")
            # 為了演示流程，可以生成一個隨機的6維數據集
            print("為演示私人數據集流程，生成隨機6D數據...")
            df = pd.DataFrame(np.random.rand(200, 6), columns=[f'priv_feat_{i+1}' for i in range(6)])
            df['id'] = range(200)
            if feature_columns is None:
                 feature_columns = [f'priv_feat_{i+1}' for i in range(6)]
        # return # 如果不生成隨機數據則返回

    # 提取特徵用於後續處理
    features_df = df[feature_columns].copy()

    # 2. 數據預處理
    print("\n--- 數據預處理 ---")
    df_standardized_features, scaler = standardize_features(features_df, feature_columns)
    if df_standardized_features is None:
        return

    processed_features = df_standardized_features.copy()

    if not is_public_data and n_dimensions > 4: # 例如私人6維數據集
        print("\n對私人高維數據應用 PCA...")
        # 假設我們想保留95%的方差，或者指定一個較小的維度數，例如3或4
        processed_features_pca, pca_model = apply_pca(df_standardized_features, explained_variance_ratio=0.95)
        if processed_features_pca is not None:
            processed_features = processed_features_pca
        else:
            print("PCA 失敗，將使用標準化後的原始維度特徵。")
    
    # 3. 聚類分析
    print("\n--- 聚類分析 ---")
    num_clusters = 4 * n_dimensions - 1
    print(f"目標簇數: {num_clusters}")

    # K-Means
    print("\nK-Means:")
    kmeans_labels, kmeans_model = apply_kmeans(processed_features, n_clusters=num_clusters)
    s_score_kmeans, db_score_kmeans = evaluate_clustering_internal(processed_features, kmeans_labels)

    # Agglomerative Clustering
    print("\n層次聚類 (Agglomerative):")
    agg_labels, agg_model = apply_agglomerative_clustering(processed_features, n_clusters=num_clusters)
    s_score_agg, db_score_agg = evaluate_clustering_internal(processed_features, agg_labels)
    
    # 比較 K-Means 和 Agglomerative 的 NMI
    print("\nNMI (K-Means vs Agglomerative):")
    nmi_kmeans_vs_agg = evaluate_clustering_nmi(kmeans_labels, agg_labels)

    # DBSCAN - eps 和 min_samples 需要根據數據調整，這裡使用示例值
    # 對於n維數據，min_samples 通常建議 >= n_dimensions + 1
    print("\nDBSCAN:")
    # 調整 eps 可能有助於獲得不同數量的簇，但精確控制到 4n-1 很難
    # 這裡的 eps 是一個示例值，實際應用中需要調優
    optimal_eps = 0.5 # 示例值，需要根據數據調整
    if not processed_features.empty:
        # 嘗試根據數據標準差的倍數來估計 eps
        # 這只是一個非常粗略的啟發式方法，實際調優更複雜
        try:
            std_devs = processed_features.std(axis=0)
            # 如果是 PCA 後的數據，列名可能不是數字
            # mean_std = std_devs.mean() if not std_devs.empty else 0.5
            # optimal_eps = mean_std * 0.5 # 示例因子
            # 由於PCA後特徵的尺度可能變化，直接用0.5或1.0作為初始嘗試值
            if processed_features.shape[1] > 2: # 如果維度較高
                 optimal_eps = 1.0 + 0.1 * processed_features.shape[1] # 隨維度增加eps
            else:
                 optimal_eps = 0.5

        except Exception as e:
            print(f"計算DBSCAN的eps時出錯: {e}, 使用默認值0.5")
            optimal_eps = 0.5
        
        print(f"DBSCAN 試用 eps={optimal_eps:.2f}, min_samples={n_dimensions+1}")

    dbscan_labels, dbscan_model = apply_dbscan(processed_features, eps=optimal_eps, min_samples=max(2, n_dimensions + 1))
    # 只有當 DBSCAN 產生多個簇時才評估
    if len(set(dbscan_labels) - {-1}) > 1 :
         s_score_dbscan, db_score_dbscan = evaluate_clustering_internal(processed_features, dbscan_labels)
    else:
        print("DBSCAN 未能找到足夠的簇進行內部評估。")
        s_score_dbscan, db_score_dbscan = np.nan, np.nan


    # GMM
    print("\nGMM:")
    gmm_labels, gmm_model = apply_gmm(processed_features, n_components=num_clusters)
    s_score_gmm, db_score_gmm = evaluate_clustering_internal(processed_features, gmm_labels)

    # 假設我們選擇 K-Means 的結果作為分類任務的偽標籤
    classification_target_labels = kmeans_labels
    
    # 4. 分類分析 (使用 K-Means 聚類標籤作為目標)
    print("\n--- 分類分析 (使用 K-Means 標籤) ---")
    # 確保分類目標中至少有兩個類別
    if len(set(classification_target_labels)) < 2:
        print("K-Means 產生的簇少於2個，無法進行分類任務。")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            processed_features, classification_target_labels, test_size=0.3, random_state=42, stratify=classification_target_labels
        )
        
        n_unique_classes = len(set(y_train)) # 分類任務中的類別數

        print("\n訓練和評估邏輯回歸:")
        lr_model = train_logistic_regression(X_train, y_train)
        lr_preds = lr_model.predict(X_test)
        lr_proba = lr_model.predict_proba(X_test) if hasattr(lr_model, "predict_proba") else None
        evaluate_classification(y_test, lr_preds, y_proba=lr_proba, n_classes=n_unique_classes)

        print("\n訓練和評估 SVM:")
        svm_model = train_svm(X_train, y_train)
        svm_preds = svm_model.predict(X_test)
        svm_proba = svm_model.predict_proba(X_test) if hasattr(svm_model, "predict_proba") else None
        evaluate_classification(y_test, svm_preds, y_proba=svm_proba, n_classes=n_unique_classes)

        print("\n訓練和評估隨機森林:")
        rf_model = train_random_forest(X_train, y_train)
        rf_preds = rf_model.predict(X_test)
        rf_proba = rf_model.predict_proba(X_test) if hasattr(rf_model, "predict_proba") else None
        evaluate_classification(y_test, rf_preds, y_proba=rf_proba, n_classes=n_unique_classes)

    print(f"\n{'='*20} {data_name} 分析完成 {'='*20}")


if __name__ == "__main__":
    # 分析公共數據集
    public_data_file = "data/public_data.csv"
    public_feature_cols = ['1', '2', '3', '4'] # 根據 public_data.csv 的實際列名
    run_analysis(data_source=public_data_file, 
                 data_name="公共數據集", 
                 n_dimensions=4, 
                 is_public_data=True,
                 feature_columns=public_feature_cols)

    # 分析私人數據集 (概念性)
    # 實際應用中，您需要提供私人數據的路徑和特徵列
    # private_data_file = "data/private_data.csv" # 假設的路徑
    # private_feature_cols = [...] # 私人數據集的特徵列
    run_analysis(data_source=None, # 傳遞 None，因為 load_private_data_placeholder 會處理
                 data_name="私人數據集", 
                 n_dimensions=6, 
                 is_public_data=False,
                 feature_columns=None) # main函數中會為演示生成列名