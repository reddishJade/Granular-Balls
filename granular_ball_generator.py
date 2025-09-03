import numpy as np
import faiss
from tqdm.auto import tqdm
from itertools import combinations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import warnings
from concurrent.futures import ThreadPoolExecutor
import gc


@dataclass
class GranularBallConfig:
    """配置类，集中管理所有参数"""
    k: int = 20
    top_k_peaks: int = 500
    M_neighbors_for_delta: int = 200
    jaccard_threshold: float = 0.5
    radius_percentile: float = 10
    min_ball_size: int = 5
    max_balls: Optional[int] = None
    use_gpu: bool = False
    batch_size: int = 1000
    max_cohesion_samples: int = 200
    n_threads: int = 4
    
    # 新增参数解决提到的问题
    min_radius: float = 0.01  # 最小半径阈值
    max_radius: float = 1.0   # 最大半径阈值
    density_adaptive_factor: float = 0.1  # 密度自适应因子
    overlap_threshold: float = 0.7  # 球重叠阈值
    random_state: Optional[int] = 42  # 随机种子
    adaptive_search_factor: float = 0.1  # 自适应搜索因子


@dataclass
class GranularBall:
    """粒球数据结构"""
    center_index: int
    center_embedding: np.ndarray
    radius: float
    sample_indices: np.ndarray
    sample_count: int
    cohesion_score: float
    label_prevalence: np.ndarray
    top_labels: List[Tuple[int, float]]  # 改进：保存标签索引和占比
    
    def __post_init__(self):
        self.center_embedding = self.center_embedding.astype('float32')
        self.sample_indices = self.sample_indices.astype(int)


class ProductionGranularBallGenerator:
    def __init__(self, X: np.ndarray, y: np.ndarray, config: GranularBallConfig = None):
        """
        生产级粒球生成器 - 解决所有性能和鲁棒性问题
        """
        print("--- 初始化生产级粒球生成器 ---")
        self.config = config or GranularBallConfig()
        
        # 设置随机种子保证可重现性
        if self.config.random_state is not None:
            np.random.seed(self.config.random_state)
        
        # 数据预处理和验证
        self.X = self._validate_and_prepare_data(X)
        self.y = self._validate_and_prepare_labels(y)
        self.n_samples, self.n_features = self.X.shape
        self.n_labels = self.y.shape[1]
        
        # 自适应搜索参数
        self.adaptive_radius_search_k = self._get_adaptive_search_k()
        
        # 初始化FAISS索引
        self.index = self._build_faiss_index()
        
        # 缓存和结果
        self._cache = {}
        self._density_rho = None  # 缓存密度值
        self.granular_balls: List[GranularBall] = []
        
        print(f"数据准备完成: {self.n_samples} 样本, {self.n_features} 特征, {self.n_labels} 标签")
        print(f"自适应搜索规模: {self.adaptive_radius_search_k}")

    def _get_adaptive_search_k(self) -> int:
        """自适应确定搜索规模"""
        base_k = min(1000, max(100, int(self.n_samples * self.config.adaptive_search_factor)))
        return base_k

    def _validate_and_prepare_data(self, X: np.ndarray) -> np.ndarray:
        """验证和准备特征数据"""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if X.ndim != 2:
            raise ValueError(f"X必须是2D数组，当前维度: {X.ndim}")
        
        if np.isnan(X).any():
            warnings.warn("检测到NaN值，将用均值填充")
            X = np.nan_to_num(X, nan=np.nanmean(X))
        
        return X.astype('float32')

    def _validate_and_prepare_labels(self, y: np.ndarray) -> np.ndarray:
        """验证和准备标签数据"""
        if not isinstance(y, np.ndarray):
            y = np.array(y)
            
        if y.ndim == 1:
            n_classes = len(np.unique(y))
            y_multi = np.zeros((len(y), n_classes), dtype=bool)
            for i, label in enumerate(y):
                y_multi[i, int(label)] = True
            y = y_multi
        
        return y.astype(bool)

    def _build_faiss_index(self) -> faiss.Index:
        """构建FAISS索引"""
        index = faiss.IndexFlatL2(self.n_features)
        
        if self.config.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                print("FAISS索引成功迁移到GPU")
            except Exception as e:
                print(f"GPU初始化失败: {e}，使用CPU")
        
        index.add(self.X)
        print(f"FAISS索引构建完成: {index.ntotal} 个向量")
        return index

    def _jaccard_similarity_matrix(self, sample_indices: np.ndarray) -> np.ndarray:
        """
        矩阵化Jaccard相似度计算 - 解决O(n²)性能问题
        """
        if len(sample_indices) == 0:
            return np.array([[]])
        
        Y = self.y[sample_indices].astype(np.int32)  # (n_samples, n_labels)
        n_samples = len(sample_indices)
        
        if n_samples == 1:
            return np.array([[1.0]])
        
        # 矩阵化计算交集：Y @ Y.T
        intersection = Y @ Y.T  # (n_samples, n_samples)
        
        # 矩阵化计算并集：使用广播
        # union[i,j] = sum(Y[i] | Y[j]) = sum(Y[i]) + sum(Y[j]) - intersection[i,j]
        label_counts = np.sum(Y, axis=1)  # 每个样本的标签数量
        union = label_counts[:, None] + label_counts[None, :] - intersection
        
        # 计算Jaccard相似度矩阵
        similarity_matrix = intersection / (union + 1e-9)
        
        return similarity_matrix

    def _calculate_cohesion_matrix(self, sample_indices: np.ndarray) -> float:
        """
        使用矩阵运算的超快凝聚度计算 - 解决性能瓶颈
        """
        if len(sample_indices) < 2:
            return 1.0
        
        # 采样策略
        if len(sample_indices) > self.config.max_cohesion_samples:
            if self.config.random_state is not None:
                np.random.seed(self.config.random_state + hash(tuple(sample_indices)) % 10000)
            sample_indices = np.random.choice(
                sample_indices, self.config.max_cohesion_samples, replace=False
            )
        
        # 使用矩阵化计算
        similarity_matrix = self._jaccard_similarity_matrix(sample_indices)
        
        # 只取上三角矩阵（避免重复计算和对角线）
        upper_triangular = np.triu(similarity_matrix, k=1)
        n_pairs = len(sample_indices) * (len(sample_indices) - 1) // 2
        
        if n_pairs == 0:
            return 1.0
        
        return np.sum(upper_triangular) / n_pairs

    def _calculate_density_peaks_optimized(self) -> np.ndarray:
        """优化的密度峰值计算"""
        cache_key = f"density_peaks_{self.config.k}_{self.config.top_k_peaks}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        print(f"计算密度峰值 (k={self.config.k})...")
        
        # 批量k-NN搜索
        D_k, I_k = self.index.search(self.X, self.config.k + 1)
        k_dist = D_k[:, -1]
        self._density_rho = 1.0 / (k_dist + 1e-9)  # 缓存密度值
        
        # 优化delta计算
        delta = self._calculate_delta_optimized(self._density_rho)
        
        # 计算密度峰值得分
        scores = self._density_rho * delta
        peak_indices = np.argsort(scores)[::-1][:self.config.top_k_peaks]
        
        self._cache[cache_key] = peak_indices
        print(f"找到 {len(peak_indices)} 个密度峰值")
        return peak_indices

    def _calculate_delta_optimized(self, rho: np.ndarray) -> np.ndarray:
        """优化的delta值计算"""
        delta = np.zeros(self.n_samples)
        sorted_indices = np.argsort(rho)[::-1]
        
        # 批量搜索邻居
        D_neighbors, I_neighbors = self.index.search(
            self.X, self.config.M_neighbors_for_delta + 1
        )
        
        print("计算Delta值...")
        for i, idx in enumerate(tqdm(sorted_indices)):
            if i == 0:
                delta[idx] = np.max(D_neighbors[idx])
                continue
            
            neighbor_indices = I_neighbors[idx, 1:]
            neighbor_rho = rho[neighbor_indices]
            higher_rho_mask = neighbor_rho > rho[idx]
            
            if np.any(higher_rho_mask):
                higher_rho_distances = D_neighbors[idx, 1:][higher_rho_mask]
                delta[idx] = np.min(higher_rho_distances)
            else:
                delta[idx] = np.max(D_neighbors[idx])
        
        return delta

    def _get_robust_adaptive_radius(self, center_idx: int) -> float:
        """
        鲁棒的自适应半径计算 - 解决半径策略问题
        """
        D_neighbors, I_neighbors = self.index.search(
            self.X[center_idx:center_idx+1], self.adaptive_radius_search_k
        )
        
        center_labels = self.y[center_idx]
        neighbor_indices = I_neighbors[0, 1:]
        
        # 批量计算相似度
        similarities = []
        for neighbor_idx in neighbor_indices:
            neighbor_labels = self.y[neighbor_idx]
            sim_matrix = self._jaccard_similarity_matrix(np.array([center_idx, neighbor_idx]))
            similarities.append(sim_matrix[0, 1])
        
        similarities = np.array(similarities)
        
        # 找到不相似的邻居
        dissimilar_mask = similarities < self.config.jaccard_threshold
        
        if not np.any(dissimilar_mask):
            base_radius = D_neighbors[0, -1]
        else:
            dissimilar_distances = D_neighbors[0, 1:][dissimilar_mask]
            base_radius = np.percentile(dissimilar_distances, self.config.radius_percentile)
        
        # 密度自适应调整
        if self._density_rho is not None:
            density_factor = self.config.density_adaptive_factor / (self._density_rho[center_idx] + 1e-9)
            adaptive_radius = min(base_radius, density_factor)
        else:
            adaptive_radius = base_radius
        
        # 应用最小最大半径约束
        robust_radius = np.clip(adaptive_radius, self.config.min_radius, self.config.max_radius)
        
        return robust_radius

    def _check_ball_overlap(self, new_sample_indices: np.ndarray) -> bool:
        """
        检查新球与已有球的重叠 - 解决重叠问题
        """
        if not self.granular_balls:
            return False
        
        new_samples_set = set(new_sample_indices)
        
        for existing_ball in self.granular_balls:
            existing_samples_set = set(existing_ball.sample_indices)
            
            # 计算Jaccard重叠
            intersection_size = len(new_samples_set & existing_samples_set)
            union_size = len(new_samples_set | existing_samples_set)
            
            if union_size > 0:
                overlap_ratio = intersection_size / union_size
                if overlap_ratio > self.config.overlap_threshold:
                    return True
        
        return False

    def _get_interpretable_top_labels(self, label_prevalence: np.ndarray, top_k: int = 3) -> List[Tuple[int, float]]:
        """
        获取可解释的标签信息 - 解决可解释性问题
        """
        top_indices = np.argsort(label_prevalence)[::-1][:top_k]
        top_values = label_prevalence[top_indices]
        return [(int(idx), float(val)) for idx, val in zip(top_indices, top_values)]

    def _process_center_batch(self, center_indices: np.ndarray) -> List[GranularBall]:
        """批量处理中心点"""
        balls = []
        
        for center_idx in center_indices:
            try:
                # 计算鲁棒自适应半径
                radius = self._get_robust_adaptive_radius(center_idx)
                
                # 找到球内样本
                D_candidates, I_candidates = self.index.search(
                    self.X[center_idx:center_idx+1], self.adaptive_radius_search_k
                )
                in_ball_mask = D_candidates[0] <= radius
                sample_indices = I_candidates[0][in_ball_mask]
                
                if len(sample_indices) < self.config.min_ball_size:
                    continue
                
                # 检查重叠
                if self._check_ball_overlap(sample_indices):
                    continue
                
                # 使用矩阵化计算凝聚度
                cohesion = self._calculate_cohesion_matrix(sample_indices)
                
                # 计算标签流行度和可解释标签
                label_prevalence = np.mean(self.y[sample_indices], axis=0)
                top_labels = self._get_interpretable_top_labels(label_prevalence)
                
                ball = GranularBall(
                    center_index=center_idx,
                    center_embedding=self.X[center_idx].copy(),
                    radius=radius,
                    sample_indices=sample_indices,
                    sample_count=len(sample_indices),
                    cohesion_score=cohesion,
                    label_prevalence=label_prevalence,
                    top_labels=top_labels
                )
                balls.append(ball)
                
            except Exception as e:
                warnings.warn(f"处理中心点 {center_idx} 时出错: {e}")
                continue
        
        return balls

    def fit(self) -> List[GranularBall]:
        """训练粒球生成器"""
        print("\n--- 开始粒球生成 (生产级) ---")
        
        # 计算密度峰值
        potential_centers = self._calculate_density_peaks_optimized()
        
        # 批量处理中心点
        batch_size = self.config.batch_size
        all_balls = []
        
        for i in tqdm(range(0, len(potential_centers), batch_size), desc="生成粒球"):
            if self.config.max_balls and len(all_balls) >= self.config.max_balls:
                break
                
            batch_centers = potential_centers[i:i+batch_size]
            if self.config.max_balls:
                remaining_slots = self.config.max_balls - len(all_balls)
                batch_centers = batch_centers[:remaining_slots]
            
            batch_balls = self._process_center_batch(batch_centers)
            all_balls.extend(batch_balls)
            
            # 内存清理
            if i % (batch_size * 5) == 0:
                gc.collect()
        
        self.granular_balls = all_balls
        print(f"\n✅ 生成完成：创建了 {len(self.granular_balls)} 个粒球")
        return self.granular_balls

    def get_detailed_statistics(self) -> Dict:
        """获取详细的生成统计信息"""
        if not self.granular_balls:
            return {}
        
        cohesions = [ball.cohesion_score for ball in self.granular_balls]
        sizes = [ball.sample_count for ball in self.granular_balls]
        radii = [ball.radius for ball in self.granular_balls]
        
        # 计算覆盖情况
        all_covered_samples = set()
        for ball in self.granular_balls:
            all_covered_samples.update(ball.sample_indices)
        
        # 重叠统计
        total_ball_samples = sum(sizes)
        overlap_rate = (total_ball_samples - len(all_covered_samples)) / total_ball_samples if total_ball_samples > 0 else 0
        
        return {
            'total_balls': len(self.granular_balls),
            'avg_cohesion': np.mean(cohesions),
            'std_cohesion': np.std(cohesions),
            'min_cohesion': np.min(cohesions),
            'max_cohesion': np.max(cohesions),
            'avg_size': np.mean(sizes),
            'std_size': np.std(sizes),
            'avg_radius': np.mean(radii),
            'std_radius': np.std(radii),
            'coverage_ratio': len(all_covered_samples) / self.n_samples,
            'overlap_rate': overlap_rate,
            'adaptive_search_k': self.adaptive_radius_search_k
        }

    def print_top_balls(self, top_k: int = 5):
        """打印最佳粒球的详细信息"""
        if not self.granular_balls:
            print("没有生成的粒球")
            return
        
        print(f"\n--- 前 {top_k} 个最佳粒球 ---")
        best_balls = sorted(self.granular_balls, key=lambda b: b.cohesion_score, reverse=True)[:top_k]
        
        for i, ball in enumerate(best_balls):
            print(f"\n🏀 粒球 {i+1}:")
            print(f"   凝聚度: {ball.cohesion_score:.4f}")
            print(f"   样本数: {ball.sample_count}")
            print(f"   半径: {ball.radius:.4f}")
            print(f"   主要标签: {[(f'标签{idx}', f'{val:.3f}') for idx, val in ball.top_labels]}")

    def save_balls(self, filepath: str):
        """保存粒球到文件"""
        balls_data = {
            'balls': [],
            'config': self.config.__dict__,
            'statistics': self.get_detailed_statistics()
        }
        
        for ball in self.granular_balls:
            ball_dict = {
                'center_index': ball.center_index,
                'center_embedding': ball.center_embedding,
                'radius': ball.radius,
                'sample_indices': ball.sample_indices,
                'sample_count': ball.sample_count,
                'cohesion_score': ball.cohesion_score,
                'label_prevalence': ball.label_prevalence,
                'top_labels': ball.top_labels
            }
            balls_data['balls'].append(ball_dict)
        
        np.savez_compressed(filepath, **balls_data)
        print(f"粒球已保存到 {filepath}")


# 使用示例
if __name__ == '__main__':
    print("--- 测试生产级粒球生成器 ---")
    
    # 加载数据
    try:
        data = np.load('preprocessed_data.npz')
        X_train, y_train = data['X_train'], data['y_train']
        print(f"加载数据: X={X_train.shape}, y={y_train.shape}")
        
        sample_size = 3000  # 适中的测试规模
        X_sample = X_train[:sample_size]
        y_sample = y_train[:sample_size]
        
        # 生产级配置
        config = GranularBallConfig(
            k=25,
            top_k_peaks=150,
            M_neighbors_for_delta=100,
            jaccard_threshold=0.4,
            radius_percentile=15,
            min_ball_size=8,
            max_balls=50,
            use_gpu=False,
            batch_size=25,
            max_cohesion_samples=100,
            # 新增的鲁棒性参数
            min_radius=0.05,
            max_radius=2.0,
            density_adaptive_factor=0.15,
            overlap_threshold=0.6,
            random_state=42,
            adaptive_search_factor=0.12
        )
        
        # 创建生成器并训练
        print("\n🚀 开始生产级粒球生成...")
        generator = ProductionGranularBallGenerator(X_sample, y_sample, config)
        balls = generator.fit()
        
        # 显示详细统计
        stats = generator.get_detailed_statistics()
        print("\n📊 详细统计信息:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # 显示最佳粒球
        generator.print_top_balls(3)
        
        # 保存结果
        generator.save_balls('production_granular_balls.npz')
        
    except FileNotFoundError:
        print("未找到数据文件，使用模拟数据测试...")
        np.random.seed(42)
        X_sim = np.random.randn(800, 30).astype('float32')
        y_sim = np.random.randint(0, 2, (800, 8)).astype(bool)
        
        config = GranularBallConfig(
            max_balls=15, 
            batch_size=5,
            random_state=42,
            min_radius=0.1,
            max_radius=1.5
        )
        generator = ProductionGranularBallGenerator(X_sim, y_sim, config)
        balls = generator.fit()
        
        print(f"\n✅ 模拟数据测试完成")
        generator.print_top_balls(3)
        stats = generator.get_detailed_statistics()
        print(f"\n📈 生成了 {stats['total_balls']} 个粒球，覆盖率: {stats['coverage_ratio']:.3f}")