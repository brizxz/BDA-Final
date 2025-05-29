import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import time

# 設置隨機種子以確保結果可重複
np.random.seed(42)

# 讀取數據
print("讀取數據...")
public_data = pd.read_csv("public_data.csv")
private_data = pd.read_csv("private_data.csv")

# 提取特徵
public_features = public_data.iloc[:, 1:].values
private_features = private_data.iloc[:, 1:].values

# 探索數據
print(f"Public data shape: {public_features.shape}")
print(f"Private data shape: {private_features.shape}")

# 數據預處理 - 嘗試不同的縮放方法
print("數據預處理...")
scalers = {
    'standard': StandardScaler(),
    'minmax': MinMaxScaler(),
    'robust': RobustScaler()
}

best_score = -1
best_labels_public = None
best_labels_private = None
best_method = None
best_scaler = None
best_params = None

# 降維以便可視化
print("降維以便可視化...")
pca = PCA(n_components=2)
public_pca = pca.fit_transform(public_features)

# 可視化原始數據
plt.figure(figsize=(10, 8))
plt.scatter(public_pca[:, 0], public_pca[:, 1], alpha=0.5)
plt.title('PCA of Public Data')
plt.savefig('pca_visualization.png')
plt.close()

# 嘗試不同的聚類方法
print("嘗試不同的聚類方法...")

# 1. KMeans
print("\n===== KMeans =====")
for scaler_name, scaler in scalers.items():
    print(f"\nUsing {scaler_name} scaler")
    public_scaled = scaler.fit_transform(public_features)
    private_scaled = scaler.transform(private_features)
    
    # 尋找最佳的K值
    silhouette_scores = []
    k_range = range(2, 15)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(public_scaled)
        score = silhouette_score(public_scaled, labels)
        silhouette_scores.append(score)
        print(f"K={k}, Silhouette Score: {score:.4f}")
    
    best_k = k_range[np.argmax(silhouette_scores)]
    print(f"Best K: {best_k}, Score: {max(silhouette_scores):.4f}")
    
    # 使用最佳的K值
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    public_labels = kmeans.fit_predict(public_scaled)
    private_labels = kmeans.predict(private_scaled)
    
    # 評估
    score = silhouette_score(public_scaled, public_labels)
    if score > best_score:
        best_score = score
        best_labels_public = public_labels
        best_labels_private = private_labels
        best_method = "KMeans"
        best_scaler = scaler_name
        best_params = {"n_clusters": best_k}

# 2. DBSCAN
print("\n===== DBSCAN =====")
for scaler_name, scaler in scalers.items():
    print(f"\nUsing {scaler_name} scaler")
    public_scaled = scaler.fit_transform(public_features)
    private_scaled = scaler.transform(private_features)
    
    # 嘗試不同的eps和min_samples參數
    eps_values = [0.1, 0.5, 1.0, 1.5, 2.0]
    min_samples_values = [5, 10, 15, 20]
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            try:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                public_labels = dbscan.fit_predict(public_scaled)
                
                # 檢查是否有足夠的聚類（DBSCAN可能將某些點標記為噪聲，標籤為-1）
                n_clusters = len(set(public_labels)) - (1 if -1 in public_labels else 0)
                if n_clusters < 2:
                    print(f"eps={eps}, min_samples={min_samples}: 聚類不足")
                    continue
                
                # 評估
                score = silhouette_score(public_scaled, public_labels)
                print(f"eps={eps}, min_samples={min_samples}, clusters={n_clusters}, Score: {score:.4f}")
                
                if score > best_score:
                    # 對private數據進行預測
                    private_labels = dbscan.fit_predict(private_scaled)
                    
                    best_score = score
                    best_labels_public = public_labels
                    best_labels_private = private_labels
                    best_method = "DBSCAN"
                    best_scaler = scaler_name
                    best_params = {"eps": eps, "min_samples": min_samples}
            except Exception as e:
                print(f"eps={eps}, min_samples={min_samples}: 錯誤 - {e}")

# 3. Agglomerative Clustering
print("\n===== Agglomerative Clustering =====")
for scaler_name, scaler in scalers.items():
    print(f"\nUsing {scaler_name} scaler")
    public_scaled = scaler.fit_transform(public_features)
    private_scaled = scaler.transform(private_features)
    
    # 嘗試不同的n_clusters
    for n_clusters in range(2, 15):
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        public_labels = agg.fit_predict(public_scaled)
        
        # 評估
        score = silhouette_score(public_scaled, public_labels)
        print(f"n_clusters={n_clusters}, Score: {score:.4f}")
        
        if score > best_score:
            # 對private數據進行預測（需要重新訓練，因為AgglomerativeClustering沒有predict方法）
            agg_private = AgglomerativeClustering(n_clusters=n_clusters)
            private_labels = agg_private.fit_predict(private_scaled)
            
            best_score = score
            best_labels_public = public_labels
            best_labels_private = private_labels
            best_method = "AgglomerativeClustering"
            best_scaler = scaler_name
            best_params = {"n_clusters": n_clusters}

# 輸出最佳結果
print("\n===== 最佳結果 =====")
print(f"方法: {best_method}")
print(f"縮放器: {best_scaler}")
print(f"參數: {best_params}")
print(f"分數: {best_score:.4f}")

# 可視化最佳聚類結果
plt.figure(figsize=(10, 8))
plt.scatter(public_pca[:, 0], public_pca[:, 1], c=best_labels_public, cmap='viridis', alpha=0.5)
plt.title(f'Best Clustering: {best_method} with {best_scaler} scaler')
plt.colorbar(label='Cluster')
plt.savefig('best_clustering.png')
plt.close()

# 創建提交文件
print("創建提交文件...")
public_submission = pd.DataFrame({
    "id": public_data["id"],
    "label": best_labels_public
})

private_submission = pd.DataFrame({
    "id": private_data["id"],
    "label": best_labels_private
})

public_submission.to_csv("public_submission.csv", index=False)
private_submission.to_csv("private_submission.csv", index=False)

print("完成！")

# 評估分數
try:
    from grader import score
    labels_pred = public_submission["label"].tolist()
    print(f"Public Score: {score(labels_pred):.4f}")
except Exception as e:
    print(f"無法評估分數: {e}") 