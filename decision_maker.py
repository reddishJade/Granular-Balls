import numpy as np
import joblib
import logging
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Union

# --- 1. 全局配置与导入 ---
# 使用logging代替print
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 将faiss导入置于顶部
try:
    import faiss
except ImportError:
    logging.error("FAISS is not installed. Please run 'pip install faiss-cpu' or 'faiss-gpu'")
    exit(1)

# --- 2. 数据结构 ---
@dataclass
class GranularBall:
    # ... (与之前版本相同)
    center_index: int; center_embedding: np.ndarray; radius: float; sample_indices: np.ndarray
    sample_count: int; cohesion_score: float; label_prevalence: np.ndarray
    top_labels: List[Tuple[int, float]]

@dataclass
class DecisionConfigV2:
    """(V2版) 决策器配置，更灵活"""
    k_nearest_balls: int = 7
    weighting_method: str = 'softmax' # 'softmax' 或 'inverse'
    temperature: float = 0.5          # Softmax温度参数
    # 支持全局阈值或按标签定制的阈值字典
    alpha: Union[float, Dict[int, float]] = 0.5
    beta: Union[float, Dict[int, float]] = 0.2

@dataclass
class Decision:
    label_name: str
    decision: str
    score: float

class ThreeWayDecisionMakerFinal:
    """(最终版) 三域决策器，具备生产级质量"""
    def __init__(self, balls_filepath: str, encoder_model_name='all-MiniLM-L6-v2', config: DecisionConfigV2 = None):
        logging.info("--- Initializing ThreeWayDecisionMakerFinal ---")
        self.config = config or DecisionConfigV2()
        
        self.encoder = SentenceTransformer(encoder_model_name)
        
        # (核心改进) 从joblib文件加载数据和标签名，实现解耦
        saved_data = joblib.load(balls_filepath)
        self.balls: List[GranularBall] = saved_data['balls']
        self.label_names: List[str] = saved_data['label_names']
        logging.info(f"Loaded {len(self.balls)} balls and {len(self.label_names)} label names.")
        
        if not self.balls: raise ValueError("No granular balls found.")
            
        ball_centers = np.array([ball.center_embedding for ball in self.balls]).astype('float32')
        self.n_features = ball_centers.shape[1]
        self.index = faiss.IndexFlatL2(self.n_features)
        self.index.add(ball_centers)
        logging.info(f"FAISS index for {len(self.balls)} ball centers built.")

    def _calculate_label_scores(self, embedding: np.ndarray) -> np.ndarray:
        embedding = embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(embedding, self.config.k_nearest_balls)
        distances = distances[0]
        indices = indices[0]

        if self.config.weighting_method == 'softmax':
            # (核心改进) Softmax权重，更平滑
            weights = np.exp(-distances / self.config.temperature)
            weights /= np.sum(weights)
        else: # 'inverse'
            weights = 1.0 / (distances + 1e-9)
            weights /= np.sum(weights)
        
        weighted_prevalences = np.zeros(len(self.label_names))
        for i, ball_idx in enumerate(indices):
            weighted_prevalences += self.balls[ball_idx].label_prevalence * weights[i]
            
        return weighted_prevalences

    def predict(self, text: str) -> List[Decision]:
        embedding = self.encoder.encode(text, show_progress_bar=False)
        scores = self._calculate_label_scores(embedding)
        
        decisions = []
        for i, score in enumerate(scores):
            # (核心改进) 支持按标签定制阈值
            alpha = self.config.alpha.get(i, self.config.alpha) if isinstance(self.config.alpha, dict) else self.config.alpha
            beta = self.config.beta.get(i, self.config.beta) if isinstance(self.config.beta, dict) else self.config.beta
            
            if score >= alpha: decision_str = "ACCEPT"
            elif score <= beta: decision_str = "REJECT"
            else: decision_str = "DEFER"
            
            decisions.append(Decision(label_name=self.label_names[i], decision=decision_str, score=score))
            
        return decisions

# ===================================================================
# Main execution block
# ===================================================================
if __name__ == '__main__':
    BALLS_FILE = 'final_balls_v3.joblib'
    
    logging.info("--- Demonstrating ThreeWayDecisionMakerFinal ---")
    
    # 1. 配置决策器
    config = DecisionConfigV2(
        k_nearest_balls=10,
        weighting_method='softmax',
        temperature=0.1,
        alpha=0.4, # 全局alpha
        beta=0.15  # 全局beta
    )
    
    try:
        decision_maker = ThreeWayDecisionMaker(BALLS_FILE, config=config)
    except FileNotFoundError:
        logging.error(f"Error: Could not find '{BALLS_FILE}'. Please run previous scripts.")
        exit(1)

    # 2. 准备测试句子
    test_sentences = [
        "This is amazing! I'm so grateful for all your help.",
        "I am so angry and disappointed with the results.",
        "I'm not sure how I feel about this news.",
        "Wow, that's a huge surprise! I'm excited to see what happens next."
    ]
    
    # 3. 运行并展示结果
    for sentence in test_sentences:
        logging.info(f"===== Prediction for: '{sentence}' =====")
        decisions = decision_maker.predict(sentence)
        interesting_decisions = [d for d in decisions if d.decision != "REJECT"]
        
        if not interesting_decisions:
            logging.info("  - Model is confident in REJECTING all labels.")
        else:
            for d in sorted(interesting_decisions, key=lambda x: x.score, reverse=True):
                logging.info(f"  - Label: {d.label_name:<15} | Decision: {d.decision:<7} | Score: {d.score:.3f}")