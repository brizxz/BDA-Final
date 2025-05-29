# src/clustering_algorithms.py
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
import numpy as np

def apply_kmeans(data, n_clusters, random_state=42):
    """
    應用 K-Means 聚類。

    Args:
        data (numpy.array or pandas.DataFrame): 輸入數據。
        n_clusters (int): 目標簇數。
        random_state (int): 隨機狀態種子。

    Returns:
        numpy.array: 聚類標籤。
        sklearn.cluster.KMeans: 訓練好的 K-Means 模型。
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(data)
    print(f"K-Means 聚類完成，分為 {n_clusters} 個簇。")
    return labels, kmeans

def apply_agglomerative_clustering(data, n_clusters, linkage='ward'):
    """
    應用層次聚類 (Agglomerative Clustering)。

    Args:
        data (numpy.array or pandas.DataFrame): 輸入數據。
        n_clusters (int): 目標簇數。
        linkage (str): 連接標準 ('ward', 'complete', 'average', 'single')。

    Returns:
        numpy.array: 聚類標籤。
        sklearn.cluster.AgglomerativeClustering: 訓練好的模型。
    """
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = agg_clustering.fit_predict(data)
    print(f"層次聚類完成，分為 {n_clusters} 個簇 (使用 {linkage} 連接)。")
    return labels, agg_clustering

def apply_dbscan(data, eps=0.5, min_samples=5):
    """
    應用 DBSCAN 聚類。
    注意：DBSCAN 自動確定簇數，要達到特定的簇數需要仔細調參。

    Args:
        data (numpy.array or pandas.DataFrame): 輸入數據。
        eps (float): DBSCAN 的 epsilon 參數。
        min_samples (int): DBSCAN 的 min_samples 參數。

    Returns:
        numpy.array: 聚類標籤。
        sklearn.cluster.DBSCAN: 訓練好的 DBSCAN 模型。
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_points = list(labels).count(-1)
    print(f"DBSCAN 聚類完成。找到 {n_clusters_found} 個簇和 {n_noise_points} 個噪聲點 (eps={eps}, min_samples={min_samples})。")
    print("注意：DBSCAN 的簇數是自動確定的，可能與目標 '4n-1' 不同。需要仔細調優 eps 和 min_samples 以嘗試接近目標簇數。")
    return labels, dbscan

def apply_gmm(data, n_components, covariance_type='full', random_state=42, n_init=10):
    """
    應用高斯混合模型 (GMM) 聚類。

    Args:
        data (numpy.array or pandas.DataFrame): 輸入數據。
        n_components (int): 目標簇數 (高斯分量的數量)。
        covariance_type (str): 協方差類型 ('full', 'tied', 'diag', 'spherical')。
        random_state (int): 隨機狀態種子。
        n_init (int): 初始化次數。

    Returns:
        numpy.array: 聚類標籤。
        sklearn.mixture.GaussianMixture: 訓練好的 GMM 模型。
    """
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, 
                          random_state=random_state, n_init=n_init)
    labels = gmm.fit_predict(data)
    print(f"GMM 聚類完成，分為 {n_components} 個簇 (協方差類型: {covariance_type})。")
    return labels, gmm

if __name__ == '__main__':
    # 示例數據 (假設已經標準化)
    sample_data = np.random.rand(100, 4) # 100 個樣本，4 維
    n_dim = 4
    target_clusters = 4 * n_dim - 1

    print(f"目標簇數: {target_clusters}")

    kmeans_labels, _ = apply_kmeans(sample_data, n_clusters=target_clusters)
    agg_labels, _ = apply_agglomerative_clustering(sample_data, n_clusters=target_clusters)
    # 對於 DBSCAN，eps 和 min_samples 需要根據數據特性調整
    # 這裡使用 scikit-learn 的默認值或示例值
    dbscan_labels, _ = apply_dbscan(sample_data, eps=0.5, min_samples=n_dim + 1) 
    gmm_labels, _ = apply_gmm(sample_data, n_components=target_clusters)

    print("\n聚類標籤示例 (K-Means):", kmeans_labels[:10])
    print("聚類標籤示例 (Agglomerative):", agg_labels[:10])
    print("聚類標籤示例 (DBSCAN):", dbscan_labels[:10])
    print("聚類標籤示例 (GMM):", gmm_labels[:10])