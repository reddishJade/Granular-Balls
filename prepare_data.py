import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# 多标签分层抽样
from skmultilearn.model_selection import iterative_train_test_split

def load_and_inspect_data(filepath):
    """加载 Excel 并返回 DataFrame，找不到文件时返回 None。"""
    print(f"Loading data from: {filepath}")
    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    return df

def preprocess_and_embed_data(df, text_col, label_cols, model_name='all-MiniLM-L6-v2'):
    """将文本列编码为句向量，并返回 (X, y)。

    X: numpy array, 文本嵌入
    y: numpy array, 多标签二值矩阵
    """
    print("Preprocessing and embedding text...")

    # 文本和标签
    texts = df[text_col].astype(str).tolist()
    y = df[label_cols].values

    # 加载句向量模型并计算嵌入
    print(f"Load model: {model_name}")
    model = SentenceTransformer(model_name)
    X = model.encode(texts, show_progress_bar=True)

    print(f"Embeddings shape: {X.shape}")
    return X, y

def split_data(X, y, train_size=0.7, val_size=0.15):
    """按多标签分层抽样拆分为训练/验证/测试集。"""
    print("Splitting data...")
    test_size = 1.0 - train_size - val_size
    assert test_size > 0

    X_train, y_train, X_temp, y_temp = iterative_train_test_split(
        X, y, test_size=(val_size + test_size)
    )
    relative_test_size = test_size / (val_size + test_size)
    X_val, y_val, X_test, y_test = iterative_train_test_split(
        X_temp, y_temp, test_size=relative_test_size
    )

    print(f"Shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

# ===================================================================
# Main execution block
# ===================================================================
if __name__ == "__main__":
    FILEPATH = 'emotion.xlsx'

    df = load_and_inspect_data(FILEPATH)
    if df is None:
        exit(1)

    # 特征列与标签列
    TEXT_COLUMN = 'text'
    LABEL_COLUMNS = df.columns.drop(['text', 'id']).tolist()

    print(f"Text column: {TEXT_COLUMN}, labels: {len(LABEL_COLUMNS)} columns")

    X, y = preprocess_and_embed_data(df, TEXT_COLUMN, LABEL_COLUMNS)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    print("Data preparation complete.")

    print("\nSaving datasets to disk...")
    np.savez_compressed(
        'preprocessed_data.npz',
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )
    print("Datasets saved to 'preprocessed_data.npz'")