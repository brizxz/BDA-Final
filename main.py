#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大數據分析期末專案 - 聚類分析
對public_data.csv進行聚類分析並輸出預測結果
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """載入數據"""
    try:
        # 嘗試從data目錄載入
        public_data = pd.read_csv("data/public_data.csv")
        print(f"成功載入公共數據: {public_data.shape}")
        return public_data
    except FileNotFoundError:
        try:
            # 嘗試從big data目錄載入
            public_data = pd.read_csv("big data/public_data.csv")
            print(f"成功載入公共數據: {public_data.shape}")
            return public_data
        except FileNotFoundError:
            print("錯誤: 找不到public_data.csv檔案")
            return None

def preprocess_data(data):
    """數據預處理"""
    print("開始數據預處理...")
    
    # 提取特徵（除了id列）
    feature_columns = [col for col in data.columns if col != 'id']
    features = data[feature_columns].values
    
    # 標準化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    print(f"特徵維度: {features_scaled.shape}")
    print(f"特徵列: {feature_columns}")
    
    return features_scaled, scaler

def find_optimal_clusters(features, max_k=15):
    """找到最佳聚類數量"""
    print("尋找最佳聚類數量...")
    
    silhouette_scores = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        silhouette_avg = silhouette_score(features, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"K={k}, Silhouette Score: {silhouette_avg:.4f}")
    
    optimal_k = K_range[np.argmax(silhouette_scores)]
    best_score = max(silhouette_scores)
    
    print(f"最佳聚類數量: K={optimal_k}, 最佳分數: {best_score:.4f}")
    return optimal_k, best_score

def apply_clustering_algorithms(features, n_clusters):
    """應用多種聚類算法並比較結果"""
    print(f"\n應用聚類算法 (目標聚類數: {n_clusters})...")
    
    results = {}
    
    # 1. K-Means
    print("1. K-Means聚類")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(features)
    kmeans_score = silhouette_score(features, kmeans_labels)
    results['kmeans'] = {
        'labels': kmeans_labels,
        'score': kmeans_score,
        'model': kmeans
    }
    print(f"   K-Means Silhouette Score: {kmeans_score:.4f}")
    
    # 2. Agglomerative Clustering
    print("2. 層次聚類")
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    agg_labels = agg_clustering.fit_predict(features)
    agg_score = silhouette_score(features, agg_labels)
    results['agglomerative'] = {
        'labels': agg_labels,
        'score': agg_score,
        'model': agg_clustering
    }
    print(f"   層次聚類 Silhouette Score: {agg_score:.4f}")
    
    # 3. DBSCAN
    print("3. DBSCAN聚類")
    # 調整eps參數
    eps_values = [0.3, 0.5, 0.7, 1.0, 1.5]
    best_dbscan_score = -1
    best_dbscan_labels = None
    best_dbscan_model = None
    
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        dbscan_labels = dbscan.fit_predict(features)
        
        # 檢查是否產生足夠的聚類
        n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        
        if n_clusters_dbscan >= 2:
            try:
                dbscan_score = silhouette_score(features, dbscan_labels)
                print(f"   DBSCAN (eps={eps}): {n_clusters_dbscan} 個聚類, Score: {dbscan_score:.4f}")
                
                if dbscan_score > best_dbscan_score:
                    best_dbscan_score = dbscan_score
                    best_dbscan_labels = dbscan_labels
                    best_dbscan_model = dbscan
            except:
                print(f"   DBSCAN (eps={eps}): 無法計算分數")
        else:
            print(f"   DBSCAN (eps={eps}): 聚類數不足 ({n_clusters_dbscan})")
    
    if best_dbscan_labels is not None:
        results['dbscan'] = {
            'labels': best_dbscan_labels,
            'score': best_dbscan_score,
            'model': best_dbscan_model
        }
    
    # 選擇最佳方法
    best_method = max(results.keys(), key=lambda x: results[x]['score'])
    best_result = results[best_method]
    
    print(f"\n最佳聚類方法: {best_method}")
    print(f"最佳分數: {best_result['score']:.4f}")
    
    return best_result, results

def create_submission_file(data, labels, filename="public_submission.csv"):
    """創建提交檔案"""
    print(f"\n創建提交檔案: {filename}")
    
    submission = pd.DataFrame({
        'id': data['id'],
        'label': labels
    })
    
    # 確保標籤從0開始
    unique_labels = sorted(set(labels))
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    submission['label'] = submission['label'].map(label_mapping)
    
    submission.to_csv(filename, index=False)
    
    print(f"提交檔案已保存: {filename}")
    print(f"總計 {len(submission)} 筆記錄")
    print(f"聚類標籤分佈:")
    print(submission['label'].value_counts().sort_index())
    
    return submission

def visualize_results(features, labels, method_name):
    """可視化結果（使用PCA降維）"""
    try:
        import matplotlib.pyplot as plt
        
        print(f"\n生成 {method_name} 聚類結果可視化...")
        
        # PCA降維到2D
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title(f'{method_name} 聚類結果 (PCA降維)', fontsize=14)
        plt.xlabel(f'第一主成分 (解釋方差: {pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'第二主成分 (解釋方差: {pca.explained_variance_ratio_[1]:.2%})')
        plt.grid(True, alpha=0.3)
        
        # 保存圖片
        filename = f"{method_name.lower()}_clustering_result.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可視化結果已保存: {filename}")
        
    except ImportError:
        print("無法載入matplotlib，跳過可視化")
    except Exception as e:
        print(f"可視化過程中發生錯誤: {e}")

def main():
    """主函數"""
    print("="*50)
    print("大數據分析期末專案 - 聚類分析")
    print("="*50)
    
    # 1. 載入數據
    data = load_data()
    if data is None:
        return
    
    # 2. 數據預處理
    features, scaler = preprocess_data(data)
    
    # 3. 找到最佳聚類數量
    optimal_k, _ = find_optimal_clusters(features)
    
    # 4. 應用聚類算法
    best_result, all_results = apply_clustering_algorithms(features, optimal_k)
    
    # 5. 創建提交檔案
    submission = create_submission_file(data, best_result['labels'])
    
    # 6. 可視化結果
    best_method = None
    for method, result in all_results.items():
        if result['score'] == best_result['score']:
            best_method = method
            break
    
    if best_method:
        visualize_results(features, best_result['labels'], best_method.upper())
    
    # 7. 輸出總結
    print("\n" + "="*50)
    print("分析完成！")
    print("="*50)
    print(f"最佳聚類方法: {best_method}")
    print(f"聚類數量: {len(set(best_result['labels']))}")
    print(f"Silhouette Score: {best_result['score']:.4f}")
    print(f"輸出檔案: public_submission.csv")
    
    return submission

if __name__ == "__main__":
    submission = main() 