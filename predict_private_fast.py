#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大數據分析期末專案 - Private Dataset 預測 (高速版本)
使用GPU加速、平行處理和算法優化來快速完成預測
支援指定特定算法進行預測
"""

import pandas as pd
import numpy as np
import time
import warnings
import os
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
warnings.filterwarnings('ignore')

# 標準機器學習庫
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.utils import resample

# 進階聚類算法
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
    print("✓ 檢測到HDBSCAN")
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("⚠️  HDBSCAN未安裝，將跳過HDBSCAN算法")

# GPU加速庫 (可選)
try:
    import cuml
    from cuml.cluster import KMeans as cuKMeans
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    CUML_AVAILABLE = True
    print("✓ 檢測到GPU加速庫 cuML")
except ImportError:
    CUML_AVAILABLE = False
    print("⚠️  未安裝cuML，將使用CPU版本")

# 設置環境變數以提高性能
os.environ['OMP_NUM_THREADS'] = str(cpu_count())
os.environ['MKL_NUM_THREADS'] = str(cpu_count())

class FastPrivatePredictor:
    """高速版本的Private Dataset預測器"""
    
    def __init__(self, use_gpu=True, use_sampling=True, n_jobs=-1, force_algorithm=None):
        self.public_data = None
        self.private_data = None
        self.features_public = None
        self.features_private = None
        self.feature_columns = None
        self.scaler = None
        self.best_model = None
        self.best_method = None
        self.best_params = None
        self.use_gpu = use_gpu and CUML_AVAILABLE
        self.use_sampling = use_sampling
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        self.force_algorithm = force_algorithm
        
        print(f"🚀 初始化高速預測器")
        print(f"   GPU加速: {'✓' if self.use_gpu else '✗'}")
        print(f"   採樣優化: {'✓' if self.use_sampling else '✗'}")
        print(f"   CPU核心數: {self.n_jobs}")
        if force_algorithm:
            print(f"   指定算法: {force_algorithm.upper()}")
        
    def load_data(self):
        """快速載入數據"""
        print("\n📂 載入數據...")
        start_time = time.time()
        
        try:
            # 使用更快的讀取方法
            self.public_data = pd.read_csv("data/public_data.csv")
            self.private_data = pd.read_csv("data/private_data.csv")
            
            elapsed = time.time() - start_time
            print(f"✓ 公開數據: {self.public_data.shape}")
            print(f"✓ 私有數據: {self.private_data.shape}")
            print(f"✓ 載入完成，耗時: {elapsed:.1f}s")
            return True
            
        except Exception as e:
            print(f"✗ 載入失敗: {e}")
            return False
    
    def preprocess_data_fast(self):
        """高速數據預處理"""
        print("\n⚡ 快速數據預處理...")
        start_time = time.time()
        
        # 找到共同特徵列
        public_cols = set(self.public_data.columns)
        private_cols = set(self.private_data.columns)
        common_cols = public_cols.intersection(private_cols)
        self.feature_columns = [col for col in common_cols if col != 'id']
        
        print(f"✓ 共同特徵: {self.feature_columns}")
        
        # 提取特徵數據
        public_features_raw = self.public_data[self.feature_columns].values.astype(np.float32)
        private_features_raw = self.private_data[self.feature_columns].values.astype(np.float32)
        
        # 選擇縮放器
        if self.use_gpu:
            # 使用GPU縮放器
            self.scaler = cuStandardScaler()
            # 轉換為GPU格式
            import cudf
            public_df = cudf.DataFrame(public_features_raw)
            private_df = cudf.DataFrame(private_features_raw)
            
            self.features_public = self.scaler.fit_transform(public_df).values
            self.features_private = self.scaler.transform(private_df).values
        else:
            # 使用CPU縮放器，但優化設置
            self.scaler = StandardScaler()
            self.features_public = self.scaler.fit_transform(public_features_raw)
            self.features_private = self.scaler.transform(private_features_raw)
        
        elapsed = time.time() - start_time
        print(f"✓ 公開數據特徵: {self.features_public.shape}")
        print(f"✓ 私有數據特徵: {self.features_private.shape}")
        print(f"✓ 預處理完成，耗時: {elapsed:.1f}s")
        
        return True
    
    def find_best_model_fast(self):
        """快速找到最佳模型或使用指定算法"""
        print("\n🔍 快速模型選擇...")
        
        # 如果指定了特定算法，只使用該算法
        if self.force_algorithm:
            return self._train_specific_algorithm(self.force_algorithm)
        
        # 如果使用採樣，先在小樣本上快速測試
        if self.use_sampling and len(self.features_public) > 10000:
            print(f"📊 使用採樣優化 (原數據: {len(self.features_public)})")
            sample_size = min(10000, len(self.features_public) // 2)
            sample_indices = resample(range(len(self.features_public)), 
                                    n_samples=sample_size, 
                                    random_state=42)
            features_sample = self.features_public[sample_indices]
            print(f"📊 採樣大小: {features_sample.shape}")
        else:
            features_sample = self.features_public
        
        # 快速算法測試
        algorithms = [
            ('fast_kmeans', partial(self._test_fast_kmeans, features_sample)),
            ('minibatch_kmeans', partial(self._test_minibatch_kmeans, features_sample)),
            ('fast_gmm', partial(self._test_fast_gmm, features_sample))
        ]
        
        if HDBSCAN_AVAILABLE:
            algorithms.append(('hdbscan', partial(self._test_hdbscan, features_sample)))
        
        if self.use_gpu:
            algorithms.insert(0, ('gpu_kmeans', partial(self._test_gpu_kmeans, features_sample)))
        
        best_score = -1
        best_result = None
        
        for method_name, test_func in algorithms:
            print(f"\n🧪 測試 {method_name.upper()}...")
            start_time = time.time()
            
            try:
                result = test_func()
                elapsed = time.time() - start_time
                
                if result and result['score'] > best_score:
                    best_score = result['score']
                    best_result = result
                    self.best_method = method_name
                    # 用完整數據重新訓練最佳模型
                    if self.use_sampling and len(self.features_public) > 10000:
                        best_result = self._retrain_best_model(result)
                    self.best_model = best_result['model']
                    self.best_params = best_result['params']
                    
                    print(f"✓ {method_name}: Score={result['score']:.4f}, "
                          f"聚類數={result['n_clusters']}, 耗時={elapsed:.1f}s")
                else:
                    print(f"✗ {method_name}: 表現不佳或失敗")
                    
            except Exception as e:
                print(f"✗ {method_name}: 失敗 ({str(e)[:50]})")
        
        if best_result:
            print(f"\n🏆 最佳模型: {self.best_method.upper()}")
            print(f"   Silhouette Score: {best_score:.4f}")
            print(f"   聚類數: {best_result['n_clusters']}")
            return True
        else:
            print("\n✗ 無法找到有效模型")
            return False
    
    def _train_specific_algorithm(self, algorithm):
        """訓練指定的特定算法"""
        print(f"🎯 使用指定算法: {algorithm.upper()}")
        
        # 根據指定算法選擇對應的測試函數
        test_functions = {
            'kmeans': self._test_fast_kmeans,
            'fast_kmeans': self._test_fast_kmeans,
            'minibatch_kmeans': self._test_minibatch_kmeans,
            'gmm': self._test_fast_gmm,
            'fast_gmm': self._test_fast_gmm,
            'hdbscan': self._test_hdbscan,
            'dbscan': self._test_dbscan,
            'agglomerative': self._test_agglomerative,
            'spectral': self._test_spectral,
        }
        
        if self.use_gpu and algorithm in ['gpu_kmeans']:
            test_func = self._test_gpu_kmeans
        elif algorithm in test_functions:
            test_func = test_functions[algorithm]
        else:
            print(f"✗ 不支援的算法: {algorithm}")
            return False
        
        # 選擇訓練數據
        if self.use_sampling and len(self.features_public) > 10000 and algorithm != 'hdbscan':
            sample_size = min(10000, len(self.features_public) // 2)
            sample_indices = resample(range(len(self.features_public)), 
                                    n_samples=sample_size, 
                                    random_state=42)
            features_sample = self.features_public[sample_indices]
            print(f"📊 採樣大小: {features_sample.shape}")
        else:
            features_sample = self.features_public
            print(f"📊 使用完整數據: {features_sample.shape}")
        
        start_time = time.time()
        result = test_func(features_sample)
        elapsed = time.time() - start_time
        
        if result:
            # 如果使用了採樣，在完整數據上重新訓練
            if self.use_sampling and len(self.features_public) > 10000 and algorithm != 'hdbscan':
                self.best_method = algorithm
                result = self._retrain_best_model(result)
            
            self.best_model = result['model']
            self.best_method = algorithm
            self.best_params = result['params']
            
            print(f"✓ {algorithm.upper()}: Score={result['score']:.4f}, "
                  f"聚類數={result['n_clusters']}, 耗時={elapsed:.1f}s")
            return True
        else:
            print(f"✗ {algorithm.upper()}: 訓練失敗")
            return False
    
    def _test_hdbscan(self, features):
        """測試HDBSCAN算法"""
        if not HDBSCAN_AVAILABLE:
            print("✗ HDBSCAN不可用")
            return None
            
        min_cluster_sizes = [5, 10, 15, 20, 30, 50]
        best_size = 10
        best_score = -1
        best_model = None
        
        for min_cluster_size in min_cluster_sizes:
            try:
                hdbscan_model = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=5,
                    cluster_selection_epsilon=0.0
                )
                labels = hdbscan_model.fit_predict(features)
                
                # 過濾噪聲點
                valid_mask = labels != -1
                if sum(valid_mask) < len(features) * 0.1:  # 如果有效點太少，跳過
                    continue
                    
                valid_labels = labels[valid_mask]
                valid_features = features[valid_mask]
                
                if len(set(valid_labels)) >= 2:
                    score = silhouette_score(valid_features, valid_labels)
                    
                    if score > best_score:
                        best_score = score
                        best_size = min_cluster_size
                        best_model = hdbscan_model
                        
                    print(f"  HDBSCAN min_cluster_size={min_cluster_size}: "
                          f"Score={score:.4f}, 聚類數={len(set(valid_labels))}, "
                          f"噪聲點={sum(labels == -1)}")
                
            except Exception as e:
                print(f"  HDBSCAN min_cluster_size={min_cluster_size}: 失敗")
                continue
        
        if best_model:
            return {
                'model': best_model,
                'score': best_score,
                'n_clusters': len(set(best_model.labels_[best_model.labels_ != -1])),
                'params': {'min_cluster_size': best_size}
            }
        return None
    
    def _test_dbscan(self, features):
        """測試DBSCAN算法"""
        eps_values = [0.3, 0.5, 0.7, 1.0, 1.5]
        min_samples_values = [5, 10, 15]
        
        best_eps = 0.5
        best_min_samples = 5
        best_score = -1
        best_model = None
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(features)
                    
                    valid_mask = labels != -1
                    if sum(valid_mask) < len(features) * 0.1:
                        continue
                        
                    valid_labels = labels[valid_mask]
                    valid_features = features[valid_mask]
                    
                    if len(set(valid_labels)) >= 2:
                        score = silhouette_score(valid_features, valid_labels)
                        
                        if score > best_score:
                            best_score = score
                            best_eps = eps
                            best_min_samples = min_samples
                            best_model = dbscan
                            
                        print(f"  DBSCAN eps={eps}, min_samples={min_samples}: "
                              f"Score={score:.4f}")
                        
                except Exception:
                    continue
        
        if best_model:
            return {
                'model': best_model,
                'score': best_score,
                'n_clusters': len(set(best_model.labels_[best_model.labels_ != -1])),
                'params': {'eps': best_eps, 'min_samples': best_min_samples}
            }
        return None
    
    def _test_agglomerative(self, features):
        """測試層次聚類算法"""
        k_values = [2, 3, 4, 5, 6, 8, 10]
        best_k = 2
        best_score = -1
        best_model = None
        
        for k in k_values:
            try:
                agg = AgglomerativeClustering(n_clusters=k)
                labels = agg.fit_predict(features)
                
                score = silhouette_score(features, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_model = agg
                    
                print(f"  層次聚類 K={k}: Score={score:.4f}")
                
            except Exception:
                continue
        
        if best_model:
            return {
                'model': best_model,
                'score': best_score,
                'n_clusters': best_k,
                'params': {'n_clusters': best_k}
            }
        return None
    
    def _test_spectral(self, features):
        """測試譜聚類算法"""
        k_values = [2, 3, 4, 5, 6, 8]
        best_k = 2
        best_score = -1
        best_model = None
        
        for k in k_values:
            try:
                spectral = SpectralClustering(n_clusters=k, random_state=42)
                labels = spectral.fit_predict(features)
                
                score = silhouette_score(features, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_model = spectral
                    
                print(f"  譜聚類 K={k}: Score={score:.4f}")
                
            except Exception:
                continue
        
        if best_model:
            return {
                'model': best_model,
                'score': best_score,
                'n_clusters': best_k,
                'params': {'n_clusters': best_k}
            }
        return None

    def _test_gpu_kmeans(self, features):
        """測試GPU加速的K-Means"""
        if not self.use_gpu:
            return None
            
        k_values = [2, 3, 4, 5, 6]  # 只測試少數K值
        best_k = 2
        best_score = -1
        best_model = None
        
        for k in k_values:
            try:
                kmeans = cuKMeans(n_clusters=k, random_state=42, max_iter=100)
                labels = kmeans.fit_predict(features)
                
                # 轉換回CPU計算score
                labels_cpu = labels.to_pandas().values if hasattr(labels, 'to_pandas') else labels
                features_cpu = features.to_pandas().values if hasattr(features, 'to_pandas') else features
                
                score = silhouette_score(features_cpu, labels_cpu)
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_model = kmeans
                    
                print(f"  GPU K={k}: Score={score:.4f}")
                
            except Exception as e:
                print(f"  GPU K={k}: 失敗")
                continue
        
        if best_model:
            return {
                'model': best_model,
                'score': best_score,
                'n_clusters': best_k,
                'params': {'n_clusters': best_k}
            }
        return None
    
    def _test_fast_kmeans(self, features):
        """測試快速K-Means"""
        k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        best_k = 2
        best_score = -1
        best_model = None
        
        for k in k_values:
            try:
                # 使用較少的初始化次數和迭代次數
                kmeans = KMeans(n_clusters=k, random_state=42, 
                              n_init=3, max_iter=50, algorithm='elkan')
                labels = kmeans.fit_predict(features)
                
                score = silhouette_score(features, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_model = kmeans
                    
                print(f"  快速K={k}: Score={score:.4f}")
                
            except Exception:
                continue
        
        if best_model:
            return {
                'model': best_model,
                'score': best_score,
                'n_clusters': best_k,
                'params': {'n_clusters': best_k}
            }
        return None
    
    def _test_minibatch_kmeans(self, features):
        """測試MiniBatch K-Means (適合大數據)"""
        k_values = [2, 3, 4, 5, 6]
        best_k = 2
        best_score = -1
        best_model = None
        
        for k in k_values:
            try:
                # MiniBatch版本，速度更快
                kmeans = MiniBatchKMeans(n_clusters=k, random_state=42,
                                       batch_size=1000, max_iter=100)
                labels = kmeans.fit_predict(features)
                
                score = silhouette_score(features, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_model = kmeans
                    
                print(f"  MiniBatch K={k}: Score={score:.4f}")
                
            except Exception:
                continue
        
        if best_model:
            return {
                'model': best_model,
                'score': best_score,
                'n_clusters': best_k,
                'params': {'n_clusters': best_k}
            }
        return None
    
    def _test_fast_gmm(self, features):
        """測試快速GMM"""
        k_values = [2, 3, 4, 5]  # 減少測試數量
        best_k = 2
        best_score = -1
        best_model = None
        
        for k in k_values:
            try:
                # 減少迭代次數
                gmm = GaussianMixture(n_components=k, random_state=42, 
                                    max_iter=50, tol=1e-3)
                labels = gmm.fit_predict(features)
                
                score = silhouette_score(features, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_model = gmm
                    
                print(f"  快速GMM K={k}: Score={score:.4f}")
                
            except Exception:
                continue
        
        if best_model:
            return {
                'model': best_model,
                'score': best_score,
                'n_clusters': best_k,
                'params': {'n_components': best_k}
            }
        return None
    
    def _retrain_best_model(self, sample_result):
        """在完整數據上重新訓練最佳模型"""
        print(f"🔄 在完整數據上重新訓練最佳模型...")
        
        method = self.best_method
        params = sample_result['params']
        
        try:
            if 'gpu' in method and self.use_gpu:
                model = cuKMeans(**params, random_state=42)
            elif 'minibatch' in method:
                model = MiniBatchKMeans(**params, random_state=42, batch_size=1000)
            elif 'kmeans' in method:
                model = KMeans(**params, random_state=42, n_init=3, max_iter=100)
            elif 'gmm' in method:
                model = GaussianMixture(**params, random_state=42, max_iter=50)
            elif method == 'hdbscan':
                model = hdbscan.HDBSCAN(**params)
            elif method == 'dbscan':
                model = DBSCAN(**params)
            elif method == 'agglomerative':
                model = AgglomerativeClustering(**params)
            elif method == 'spectral':
                model = SpectralClustering(**params, random_state=42)
            else:
                return sample_result
            
            labels = model.fit_predict(self.features_public)
            
            # 計算新的分數
            if hasattr(labels, 'to_pandas'):
                labels = labels.to_pandas().values
            if hasattr(self.features_public, 'to_pandas'):
                features = self.features_public.to_pandas().values
            else:
                features = self.features_public
            
            # 處理有噪聲點的算法
            if method in ['hdbscan', 'dbscan']:
                valid_mask = labels != -1
                if sum(valid_mask) > 0:
                    valid_labels = labels[valid_mask]
                    valid_features = features[valid_mask]
                    if len(set(valid_labels)) >= 2:
                        score = silhouette_score(valid_features, valid_labels)
                    else:
                        score = sample_result['score']
                else:
                    score = sample_result['score']
            else:
                score = silhouette_score(features, labels)
            
            return {
                'model': model,
                'score': score,
                'n_clusters': sample_result['n_clusters'],
                'params': params
            }
            
        except Exception as e:
            print(f"⚠️  重新訓練失敗，使用採樣結果: {e}")
            return sample_result
    
    def predict_private_fast(self):
        """快速預測私有數據"""
        if self.best_model is None:
            print("✗ 沒有可用模型")
            return None
        
        print(f"\n🎯 使用 {self.best_method.upper()} 快速預測...")
        print(f"📊 私有數據大小: {len(self.private_data):,} 樣本")
        
        start_time = time.time()
        
        try:
            # 對於能直接預測的模型
            if any(name in self.best_method for name in ['kmeans', 'gmm']):
                if self.use_gpu and hasattr(self.best_model, 'predict'):
                    # GPU預測
                    if hasattr(self.features_private, 'to_pandas'):
                        private_labels = self.best_model.predict(self.features_private)
                        private_labels = private_labels.to_pandas().values
                    else:
                        import cudf
                        private_df = cudf.DataFrame(self.features_private)
                        private_labels = self.best_model.predict(private_df).to_pandas().values
                else:
                    # CPU預測
                    features = self.features_private.to_pandas().values if hasattr(self.features_private, 'to_pandas') else self.features_private
                    private_labels = self.best_model.predict(features)
            
            elif self.best_method in ['hdbscan', 'dbscan', 'agglomerative', 'spectral']:
                # 這些算法需要重新fit整個數據集
                print("⚠️  此算法需要重新fit整個數據集，可能需要較長時間...")
                
                # 合併數據
                combined_features = np.vstack([self.features_public, self.features_private])
                
                # 使用相同參數重新訓練
                if self.best_method == 'hdbscan':
                    new_model = hdbscan.HDBSCAN(**self.best_params)
                elif self.best_method == 'dbscan':
                    new_model = DBSCAN(**self.best_params)
                elif self.best_method == 'agglomerative':
                    new_model = AgglomerativeClustering(**self.best_params)
                elif self.best_method == 'spectral':
                    new_model = SpectralClustering(**self.best_params, random_state=42)
                
                combined_labels = new_model.fit_predict(combined_features)
                
                # 提取私有數據的標籤
                private_labels = combined_labels[len(self.features_public):]
            
            else:
                print("✗ 不支援的模型類型")
                return None
            
            elapsed = time.time() - start_time
            print(f"✓ 預測完成！耗時: {elapsed:.1f}s")
            
            # 顯示聚類分佈
            unique_labels, counts = np.unique(private_labels, return_counts=True)
            print(f"\n📈 私有數據聚類分佈:")
            for label, count in zip(unique_labels, counts):
                percentage = count / len(private_labels) * 100
                if label == -1:
                    print(f"  噪聲點: {count:,} 樣本 ({percentage:.1f}%)")
                else:
                    print(f"  聚類 {label}: {count:,} 樣本 ({percentage:.1f}%)")
            
            return private_labels
            
        except Exception as e:
            print(f"✗ 預測失敗: {e}")
            return None
    
    def save_results_fast(self, private_labels, filename="private_submission.csv"):
        """快速保存結果"""
        if private_labels is None:
            return None
        
        print(f"\n💾 保存結果: {filename}")
        
        submission = pd.DataFrame({
            'id': self.private_data['id'],
            'label': private_labels
        })
        
        # 確保數據類型優化
        submission['id'] = submission['id'].astype('int32')
        submission['label'] = submission['label'].astype('int32')
        
        # 標籤標準化（確保從0開始且連續，處理噪聲點-1）
        unique_labels = sorted(set(private_labels))
        if -1 in unique_labels:
            # 如果有噪聲點，將其重新標記為最大標籤+1
            max_label = max([l for l in unique_labels if l != -1])
            label_mapping = {-1: max_label + 1}
            for i, old_label in enumerate([l for l in unique_labels if l != -1]):
                label_mapping[old_label] = i
        else:
            label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        
        submission['label'] = submission['label'].map(label_mapping)
        
        # 快速保存
        submission.to_csv(filename, index=False)
        
        print(f"✓ 檔案已保存: {filename}")
        print(f"✓ 記錄數: {len(submission):,}")
        print(f"✓ 聚類數: {len(set(submission['label']))}")
        
        return submission

def parse_arguments():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='Private Dataset 聚類預測')
    parser.add_argument('--algorithm', '-a', type=str, 
                       choices=['kmeans', 'minibatch_kmeans', 'gmm', 'hdbscan', 
                               'dbscan', 'agglomerative', 'spectral', 'gpu_kmeans'],
                       help='指定要使用的聚類算法')
    parser.add_argument('--no-gpu', action='store_true', 
                       help='禁用GPU加速')
    parser.add_argument('--no-sampling', action='store_true',
                       help='禁用採樣優化')
    parser.add_argument('--output', '-o', type=str, default='private_submission.csv',
                       help='輸出檔案名稱')
    return parser.parse_args()

def main():
    """主函數"""
    # 解析命令行參數
    args = parse_arguments()
    
    print("="*70)
    print("🚀 大數據分析期末專案 - Private Dataset 高速預測")
    print("="*70)
    
    # 初始化高速預測器
    predictor = FastPrivatePredictor(
        use_gpu=not args.no_gpu,      # GPU加速
        use_sampling=not args.no_sampling,  # 採樣優化
        n_jobs=-1,                    # 使用所有CPU核心
        force_algorithm=args.algorithm # 指定算法
    )
    
    total_start_time = time.time()
    
    # 執行流程
    steps = [
        ("載入數據", predictor.load_data),
        ("數據預處理", predictor.preprocess_data_fast),
        ("模型選擇", predictor.find_best_model_fast),
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"✗ {step_name}失敗，程式結束")
            return None
    
    # 預測和保存
    private_labels = predictor.predict_private_fast()
    if private_labels is None:
        return None
    
    submission = predictor.save_results_fast(private_labels, args.output)
    
    # 總結
    total_elapsed = time.time() - total_start_time
    print("\n" + "="*70)
    print("🎉 高速預測完成！")
    print("="*70)
    print(f"⚡ 總耗時: {total_elapsed:.1f}s")
    print(f"🧠 使用模型: {predictor.best_method.upper()}")
    print(f"📊 預測樣本數: {len(private_labels):,}")
    print(f"🎯 聚類數量: {len(set(private_labels))}")
    print(f"📁 輸出檔案: {args.output}")
    
    if total_elapsed < 60:
        print(f"🏆 預測速度: {len(private_labels)/total_elapsed:.0f} 樣本/秒")
    
    return submission

if __name__ == "__main__":
    submission = main() 