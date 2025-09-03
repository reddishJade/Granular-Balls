import pandas as pd
import requests
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def download_goemotions_dataset():
    """
    下载GoEmotions数据集的三个CSV文件
    """
    print("开始下载GoEmotions数据集...")
    
    # 创建数据目录
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # 下载链接
    urls = [
        'https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv',
        'https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv',
        'https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv'
    ]
    
    files = []
    for i, url in enumerate(urls, 1):
        filename = f'data/goemotions_{i}.csv'
        print(f"正在下载文件 {i}/3: {filename}")
        
        # 下载文件
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f:
            for data in tqdm(response.iter_content(chunk_size=1024), total=total_size//1024, unit='KB'):
                f.write(data)
        
        files.append(filename)
        print(f"文件 {filename} 下载完成")
    
    return files

def process_goemotions_data(files):
    """
    处理GoEmotions数据集，合并CSV文件并创建多标签格式
    """
    print("开始处理GoEmotions数据集...")
    
    # 定义情感类别
    emotion_categories = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    
    # 读取所有CSV文件
    dfs = []
    for file in files:
        print(f"正在读取文件: {file}")
        df = pd.read_csv(file)
        dfs.append(df)
    
    # 合并所有数据框
    print("合并数据框...")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    print(f"原始数据集大小: {len(combined_df)} 行")
    
    # 创建处理后的数据框
    processed_data = []
    
    # 按评论ID分组，因为同一评论可能有多个标注者
    grouped = combined_df.groupby('id')
    
    print("处理多标签数据...")
    for comment_id, group in tqdm(grouped, desc="处理评论"):
        # 获取评论文本（取第一个即可，因为同一评论文本相同）
        text = group['text'].iloc[0]
        
        # 计算每个情感类别的多数投票
        emotion_labels = {}
        for emotion in emotion_categories:
            # 计算该情感的标注者数量
            positive_votes = group[emotion].sum()
            total_votes = len(group)
            
            # 如果超过一半的标注者认为有这个情感，则标记为1
            emotion_labels[emotion] = 1 if positive_votes > total_votes / 2 else 0
        
        # 只保留至少有一个情感标签的样本
        if sum(emotion_labels.values()) > 0:
            row = {'text': text, 'id': comment_id}
            row.update(emotion_labels)
            processed_data.append(row)
    
    # 创建处理后的数据框
    processed_df = pd.DataFrame(processed_data)
    
    print(f"处理后的数据集大小: {len(processed_df)} 行")
    
    # 显示每个情感类别的样本数量
    print("\n各情感类别的样本数量:")
    emotion_counts = processed_df[emotion_categories].sum().sort_values(ascending=False)
    for emotion, count in emotion_counts.items():
        print(f"{emotion}: {count}")
    
    return processed_df, emotion_categories

def save_to_excel(df, emotion_categories, filename='emotion.xlsx'):
    """
    将处理后的数据保存为Excel文件
    """
    print(f"正在保存数据到 {filename}...")
    
    # 重新排列列的顺序：文本列在前，情感类别列在后
    columns_order = ['text', 'id'] + emotion_categories
    df = df[columns_order]
    
    # 保存为Excel文件
    df.to_excel(filename, index=False, engine='openpyxl')
    
    print(f"数据已成功保存到 {filename}")
    print(f"数据集包含 {len(df)} 个样本")
    print(f"包含 {len(emotion_categories)} 个情感类别")

def main():
    """
    主函数：下载、处理并保存GoEmotions数据集
    """
    try:
        # 1. 下载数据集
        files = download_goemotions_dataset()
        
        # 2. 处理数据
        processed_df, emotion_categories = process_goemotions_data(files)
        
        # 3. 保存为Excel文件
        save_to_excel(processed_df, emotion_categories, 'emotion.xlsx')
        
        print("\n数据处理完成！")
        print("生成的 emotion.xlsx 文件包含:")
        print("- text: 评论文本")
        print("- id: 评论唯一标识符")
        print("- 28个情感类别的二进制标签（0或1）")
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()