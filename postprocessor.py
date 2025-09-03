import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import joblib
import warnings

# --- 数据结构 ---
@dataclass
class GranularBall:
    center_index: int
    center_embedding: np.ndarray
    radius: float
    sample_indices: np.ndarray
    sample_count: int
    cohesion_score: float
    label_prevalence: np.ndarray
    top_labels: List[Tuple[int, float]]

@dataclass
class PostprocessorConfig:
    """后处理器配置"""
    spatial_merge_threshold: float = 0.7
    semantic_merge_threshold: float = 0.85
    split_threshold: float = 0.5
    min_split_size: int = 30
    radius_percentile: int = 95
    max_cohesion_samples: int = 200
    random_state: int = 42
    max_merge_iterations: int = 5
    max_split_iterations: int = 5

class GranularBallPostprocessorFinal:
    """(V3 最终完整版) 粒球后处理器"""
    def __init__(self, balls: List[Dict], X: np.ndarray, y: np.ndarray, config: PostprocessorConfig = None):
        self.X = X.astype('float32')
        self.y = y.astype(bool)
        self.balls = [GranularBall(**b) for b in balls]
        self.config = config or PostprocessorConfig()
        np.random.seed(self.config.random_state)
        print(f"--- Initializing PostprocessorFinal with {len(self.balls)} balls ---")

    def _jaccard_similarity_sets(self, set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / (union + 1e-9)

    def _calculate_semantic_similarity(self, ball1, ball2):
        return cosine_similarity(
            ball1.label_prevalence.reshape(1, -1),
            ball2.label_prevalence.reshape(1, -1)
        )[0, 0]

    def _calculate_cohesion_matrix(self, sample_indices: np.ndarray) -> float:
        num_samples = len(sample_indices)
        if num_samples < 2: return 1.0
        if num_samples > self.config.max_cohesion_samples:
            sample_indices = np.random.choice(sample_indices, self.config.max_cohesion_samples, replace=False)
            num_samples = self.config.max_cohesion_samples
        Y_subset = self.y[sample_indices].astype(np.int32)
        intersection = Y_subset @ Y_subset.T
        label_counts = Y_subset.sum(axis=1)
        union = label_counts[:, None] + label_counts[None, :] - intersection
        sim_matrix = intersection / (union + 1e-9)
        total_similarity = np.sum(np.triu(sim_matrix, k=1))
        num_pairs = num_samples * (num_samples - 1) // 2
        return total_similarity / num_pairs if num_pairs > 0 else 1.0

    def _recalculate_ball_from_indices(self, sample_indices: np.ndarray) -> Optional[GranularBall]:
        if len(sample_indices) < self.config.min_split_size:
            return None
        ball_X, ball_y = self.X[sample_indices], self.y[sample_indices]
        new_center_embedding = np.mean(ball_X, axis=0)
        distances_to_centroid = np.linalg.norm(ball_X - new_center_embedding, axis=1)
        new_center_idx = sample_indices[np.argmin(distances_to_centroid)]
        distances_from_center = np.linalg.norm(ball_X - self.X[new_center_idx], axis=1)
        new_radius = np.percentile(distances_from_center, self.config.radius_percentile)
        cohesion = self._calculate_cohesion_matrix(sample_indices)
        label_prevalence = np.mean(ball_y, axis=0)
        top_3_indices = np.argsort(label_prevalence)[::-1][:3]
        top_labels = [(int(idx), float(label_prevalence[idx])) for idx in top_3_indices]
        return GranularBall(
            center_index=new_center_idx, center_embedding=self.X[new_center_idx],
            radius=new_radius, sample_indices=sample_indices,
            sample_count=len(sample_indices), cohesion_score=cohesion,
            label_prevalence=label_prevalence, top_labels=top_labels
        )

    def merge_balls(self):
        """合并空间和语义上都相似的粒球"""
        print(f"\n--- Starting Merge Process (Spatial > {self.config.spatial_merge_threshold}, Semantic > {self.config.semantic_merge_threshold}) ---")
        
        merged_balls_list = self.balls
        for i in range(self.config.max_merge_iterations):
            if len(merged_balls_list) < 2: break
            
            was_merged = False
            is_merged = np.zeros(len(merged_balls_list), dtype=bool)
            next_pass_balls = []
            ball_sets = [set(b.sample_indices) for b in merged_balls_list]

            for i in tqdm(range(len(merged_balls_list)), desc=f"Merging Pass {i+1}"):
                if is_merged[i]: continue
                
                indices_to_merge = {i}
                for j in range(i + 1, len(merged_balls_list)):
                    if is_merged[j]: continue
                    
                    spatial_overlap = self._jaccard_similarity_sets(ball_sets[i], ball_sets[j])
                    if spatial_overlap > self.config.spatial_merge_threshold:
                        semantic_sim = self._calculate_semantic_similarity(merged_balls_list[i], merged_balls_list[j])
                        if semantic_sim > self.config.semantic_merge_threshold:
                            indices_to_merge.add(j)
                            is_merged[j] = True
                            was_merged = True
                
                if len(indices_to_merge) > 1:
                    all_indices = set().union(*(ball_sets[idx] for idx in indices_to_merge))
                    new_ball = self._recalculate_ball_from_indices(np.array(list(all_indices)))
                    if new_ball: next_pass_balls.append(new_ball)
                    is_merged[i] = True
                else:
                    next_pass_balls.append(merged_balls_list[i])

            merged_balls_list = next_pass_balls
            if not was_merged:
                print("No more balls to merge. Converged.")
                break
            else:
                print(f"Merge pass complete. Ball count reduced to {len(merged_balls_list)}")

        self.balls = merged_balls_list
        print(f"--- Merge Process Finished. Final ball count: {len(self.balls)} ---")

    def split_balls(self):
        """分裂低内聚性的粒球，并验证分裂效果"""
        print(f"\n--- Starting Split Process (threshold={self.config.split_threshold}) ---")
        
        split_balls_list = self.balls
        for i in range(self.config.max_split_iterations):
            was_split = False
            next_pass_balls = []
            
            for ball in tqdm(split_balls_list, desc=f"Splitting Pass {i+1}"):
                if ball.cohesion_score < self.config.split_threshold and ball.sample_count >= self.config.min_split_size:
                    ball_X = self.X[ball.sample_indices]
                    kmeans = KMeans(n_clusters=2, random_state=self.config.random_state, n_init='auto')
                    clusters = kmeans.fit_predict(ball_X)
                    
                    indices1 = ball.sample_indices[clusters == 0]
                    indices2 = ball.sample_indices[clusters == 1]
                    
                    new_ball_1 = self._recalculate_ball_from_indices(indices1)
                    new_ball_2 = self._recalculate_ball_from_indices(indices2)

                    if (new_ball_1 and new_ball_2 and
                        new_ball_1.cohesion_score > ball.cohesion_score and
                        new_ball_2.cohesion_score > ball.cohesion_score):
                        next_pass_balls.extend([new_ball_1, new_ball_2])
                        was_split = True
                    else:
                        next_pass_balls.append(ball)
                else:
                    next_pass_balls.append(ball)
            
            split_balls_list = next_pass_balls
            if not was_split:
                print("No more effective splits found. Converged.")
                break
            else:
                print(f"Split pass complete. Ball count changed to {len(split_balls_list)}")

        self.balls = split_balls_list
        print(f"--- Split Process Finished. Final ball count: {len(self.balls)} ---")

    def process(self):
        """执行完整的后处理流程"""
        print("\n=== Starting Full Post-processing Pipeline (V3 Final) ===")
        initial_count = len(self.balls)
        
        self.merge_balls()
        self.split_balls()
        
        final_count = len(self.balls)
        print(f"\n✅ Post-processing complete. Ball count changed from {initial_count} to {final_count}.")
        return self.balls

# ===================================================================
# Main execution block for testing
# ===================================================================
if __name__ == '__main__':
    print("--- Testing GranularBallPostprocessorFinal ---")

    try:
        archive = np.load('production_granular_balls.npz', allow_pickle=True)
        balls_as_dicts = archive['balls']
        
        data = np.load('preprocessed_data.npz')
        X_train, y_train = data['X_train'], data['y_train']

        # 【关键更新】加载原始数据以获取标签名
        df = pd.read_excel('emotion.xlsx')
        LABEL_COLUMNS = df.columns.drop(['text', 'id']).tolist()

    except FileNotFoundError as e:
        print(f"Error: Required data file not found. {e}")
        print("Please run 'prepare_data.py' and 'granular_ball_generator_v3.py' first.")
        exit(1)

    print(f"Loaded {len(balls_as_dicts)} balls for post-processing.")

    # 使用配置类进行配置
    config = PostprocessorConfig(
        spatial_merge_threshold=0.7,
        semantic_merge_threshold=0.8,
        split_threshold=0.45,
        min_split_size=40,
        radius_percentile=90,
    )
    
    postprocessor = GranularBallPostprocessorFinal(balls_as_dicts, X_train, y_train, config)
    
    final_balls = postprocessor.process()

    if final_balls:
        # 【关键更新】将粒球和标签名一起保存
        final_data_to_save = {
            'balls': [b.__dict__ for b in final_balls], # 保存为字典列表
            'label_names': LABEL_COLUMNS
        }
        
        try:
            joblib.dump(final_data_to_save, 'final_refined_balls.joblib')
            print("\n--- Final refined balls and label names saved to 'final_refined_balls.joblib' ---")
        except Exception as e:
            print(f"Error saving with joblib: {e}")