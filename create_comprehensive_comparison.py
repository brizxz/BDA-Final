#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
綜合聚類算法比較和分析腳本
"""

import pandas as pd
import numpy as np
import os

def create_comprehensive_comparison():
    """創建綜合比較表格"""
    
    # 讀取聚類分析報告（如果存在）
    report_file = "clustering_analysis_report.txt"
    performance_metrics = {}
    
    if os.path.exists(report_file):
        print("讀取聚類分析報告...")
        with open(report_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 解析報告中的性能指標
        lines = content.split('\n')
        current_algorithm = None
        
        for line in lines:
            line = line.strip()
            if line.endswith(':') and line.replace(':', '').upper() in ['KMEANS', 'DBSCAN', 'HDBSCAN', 'SPECTRAL', 'GMM', 'AGGLOMERATIVE']:
                current_algorithm = line.replace(':', '').lower()
                performance_metrics[current_algorithm] = {}
            elif current_algorithm and '  - ' in line:
                if 'Silhouette Score:' in line:
                    score = line.split(':')[1].strip()
                    performance_metrics[current_algorithm]['silhouette'] = score
                elif '執行時間:' in line:
                    time_str = line.split(':')[1].strip().replace(' 秒', '')
                    performance_metrics[current_algorithm]['execution_time'] = time_str
                elif 'Calinski-Harabasz Score:' in line:
                    score = line.split(':')[1].strip()
                    performance_metrics[current_algorithm]['calinski_harabasz'] = score
                elif 'Davies-Bouldin Score:' in line:
                    score = line.split(':')[1].strip()
                    performance_metrics[current_algorithm]['davies_bouldin'] = score
    
    # 分析聚類分佈
    algorithms = ['kmeans', 'dbscan', 'hdbscan', 'spectral', 'gmm', 'agglomerative']
    comparison_data = []
    
    for alg in algorithms:
        filename = f'{alg}_submission.csv'
        if os.path.exists(filename):
            # 讀取結果
            df = pd.read_csv(filename)
            label_counts = df['label'].value_counts().sort_index()
            
            # 基本統計
            n_clusters = len(label_counts)
            total_samples = len(df)
            max_cluster_size = label_counts.max()
            min_cluster_size = label_counts.min()
            std_cluster_size = label_counts.std()
            imbalance_ratio = max_cluster_size / min_cluster_size if min_cluster_size > 0 else float('inf')
            
            # 從報告讀取性能指標
            metrics = performance_metrics.get(alg, {})
            
            # 計算平衡度評級
            if imbalance_ratio < 2:
                balance_grade = "優"
                balance_score = 5
            elif imbalance_ratio < 5:
                balance_grade = "良"
                balance_score = 4
            elif imbalance_ratio < 20:
                balance_grade = "中"
                balance_score = 3
            elif imbalance_ratio < 100:
                balance_grade = "差"
                balance_score = 2
            else:
                balance_grade = "極差"
                balance_score = 1
            
            # 計算聚類數評級（根據數據大小，合理的聚類數）
            if 5 <= n_clusters <= 15:
                cluster_grade = "優"
                cluster_score = 5
            elif 3 <= n_clusters <= 20:
                cluster_grade = "良" 
                cluster_score = 4
            elif 2 <= n_clusters <= 30:
                cluster_grade = "中"
                cluster_score = 3
            else:
                cluster_grade = "差"
                cluster_score = 2
            
            # 綜合評分（平衡度佔50%，聚類數適當性佔20%，Silhouette佔30%）
            silhouette_score = float(metrics.get('silhouette', '0'))
            if silhouette_score > 0.5:
                silhouette_grade = 5
            elif silhouette_score > 0.3:
                silhouette_grade = 4
            elif silhouette_score > 0.1:
                silhouette_grade = 3
            elif silhouette_score > 0:
                silhouette_grade = 2
            else:
                silhouette_grade = 1
            
            overall_score = balance_score * 0.5 + cluster_score * 0.2 + silhouette_grade * 0.3
            
            comparison_data.append({
                '算法': alg.upper(),
                '聚類數': n_clusters,
                '最大/最小比例': f"{imbalance_ratio:.1f}" if imbalance_ratio != float('inf') else "∞",
                '平衡度': balance_grade,
                'Silhouette分數': metrics.get('silhouette', 'N/A'),
                'Calinski-Harabasz': metrics.get('calinski_harabasz', 'N/A'),
                'Davies-Bouldin': metrics.get('davies_bouldin', 'N/A'),
                '執行時間(秒)': metrics.get('execution_time', 'N/A'),
                '綜合評分': f"{overall_score:.2f}",
                '推薦度': "★★★★★" if overall_score >= 4 else 
                         "★★★★☆" if overall_score >= 3.5 else
                         "★★★☆☆" if overall_score >= 3 else
                         "★★☆☆☆" if overall_score >= 2 else "★☆☆☆☆"
            })
    
    # 創建DataFrame並排序
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('綜合評分', ascending=False)
    
    return comparison_df

def generate_recommendations(comparison_df):
    """生成算法推薦"""
    print("\n" + "="*80)
    print("🎯 算法推薦和分析")
    print("="*80)
    
    best_algorithm = comparison_df.iloc[0]
    worst_algorithm = comparison_df.iloc[-1]
    
    print(f"🏆 最佳算法: {best_algorithm['算法']}")
    print(f"   - 綜合評分: {best_algorithm['綜合評分']}")
    print(f"   - 聚類數: {best_algorithm['聚類數']}")
    print(f"   - 平衡度: {best_algorithm['平衡度']}")
    print(f"   - Silhouette分數: {best_algorithm['Silhouette分數']}")
    
    print(f"\n❌ 表現最差算法: {worst_algorithm['算法']}")
    print(f"   - 綜合評分: {worst_algorithm['綜合評分']}")
    print(f"   - 主要問題: 平衡度{worst_algorithm['平衡度']}，比例{worst_algorithm['最大/最小比例']}")
    
    print(f"\n📊 各算法問題分析:")
    for _, row in comparison_df.iterrows():
        alg = row['算法']
        issues = []
        
        if row['平衡度'] in ['差', '極差']:
            issues.append(f"聚類不平衡({row['最大/最小比例']})")
        
        if row['聚類數'] < 3:
            issues.append("聚類數過少")
        elif row['聚類數'] > 20:
            issues.append("聚類數過多")
        
        try:
            silhouette = float(row['Silhouette分數'])
            if silhouette < 0.2:
                issues.append(f"Silhouette分數過低({silhouette:.3f})")
        except:
            pass
        
        if issues:
            print(f"   {alg}: {', '.join(issues)}")
        else:
            print(f"   {alg}: 表現良好")
    
    print(f"\n💡 改進建議:")
    
    # 針對層次聚類的特別建議
    agg_row = comparison_df[comparison_df['算法'] == 'AGGLOMERATIVE']
    if not agg_row.empty and agg_row.iloc[0]['平衡度'] == '極差':
        print(f"   🔧 層次聚類修復建議:")
        print(f"      1. 使用不同的連接方法 (ward, complete, average)")
        print(f"      2. 調整聚類數 (嘗試3-10個聚類)")
        print(f"      3. 考慮資料預處理 (降維、特徵選擇)")
        print(f"      4. 使用樣本數據進行參數調優")
    
    # 針對DBSCAN的建議
    dbscan_row = comparison_df[comparison_df['算法'] == 'DBSCAN']
    if not dbscan_row.empty and dbscan_row.iloc[0]['平衡度'] == '極差':
        print(f"   🔧 DBSCAN修復建議:")
        print(f"      1. 調整eps參數 (嘗試0.1-2.0)")
        print(f"      2. 調整min_samples參數 (嘗試3-20)")
        print(f"      3. 使用標準化或正規化的資料")
    
    print(f"\n✅ 建議使用順序:")
    for i, (_, row) in enumerate(comparison_df.iterrows(), 1):
        print(f"   {i}. {row['算法']} - {row['推薦度']} (評分: {row['綜合評分']})")

def main():
    """主函數"""
    print("="*80)
    print("🔍 綜合聚類算法比較分析")
    print("="*80)
    
    # 創建比較表格
    comparison_df = create_comprehensive_comparison()
    
    print("\n📊 綜合比較表格:")
    print("-" * 80)
    print(comparison_df.to_string(index=False))
    
    # 儲存表格
    comparison_df.to_csv('comprehensive_clustering_comparison.csv', index=False, encoding='utf-8')
    print(f"\n💾 詳細比較表格已儲存: comprehensive_clustering_comparison.csv")
    
    # 生成推薦
    generate_recommendations(comparison_df)
    
    print("\n" + "="*80)
    print("✨ 分析完成！建議使用評分最高的算法。")
    print("="*80)

if __name__ == "__main__":
    main() 