# 导入基础库与深度学习相关模块
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
from datetime import datetime
from scipy.signal import savgol_filter
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 设置 matplotlib 字体为黑体，解决中文乱码
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# 判断是否有可用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用设备：{device}")


# 通道注意力机制模块定义（与训练时保持一致）
class ChannelAttention(nn.Module):
    def __init__(self, input_channels=8, reduction=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_channels, input_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_channels // reduction, input_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_perm = x.permute(0, 2, 1)
        pooled = self.avg_pool(x_perm).view(x.size(0), -1)
        attn = self.fc(pooled).unsqueeze(2)
        out = x_perm * attn
        return out.permute(0, 2, 1), attn


# 带通道注意力的LSTM分类器定义（与训练时保持一致）
class EMGLSTMClassifierWithAttention(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, num_classes=3, dropout=0.3):
        super(EMGLSTMClassifierWithAttention, self).__init__()
        self.channel_attention = ChannelAttention(input_channels=input_size)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x_attn, attn_weights = self.channel_attention(x)
        lstm_out, _ = self.lstm(x_attn)
        last_step = lstm_out[:, -1, :]
        out = self.fc(last_step)
        return out, attn_weights


# 卡尔曼滤波器类
class KalmanFilter:
    def __init__(self, num_classes=3, process_noise=1e-2, measurement_noise=1e-1):
        """
        卡尔曼滤波器用于平滑预测结果

        Args:
            num_classes: 分类数量
            process_noise: 过程噪声方差
            measurement_noise: 测量噪声方差
        """
        self.num_classes = num_classes

        # 状态向量 (概率分布)
        self.x = np.ones(num_classes) / num_classes  # 初始化为均匀分布

        # 状态协方差矩阵
        self.P = np.eye(num_classes) * 0.1

        # 过程噪声协方差矩阵
        self.Q = np.eye(num_classes) * process_noise

        # 测量噪声协方差矩阵
        self.R = np.eye(num_classes) * measurement_noise

        # 状态转移矩阵 (假设状态相对稳定)
        self.F = np.eye(num_classes) * 0.9 + np.ones((num_classes, num_classes)) * 0.1 / num_classes

        # 观测矩阵
        self.H = np.eye(num_classes)

        self.is_initialized = False

    def update(self, measurement):
        """
        更新卡尔曼滤波器

        Args:
            measurement: 当前测量值 (softmax概率)

        Returns:
            filtered_probs: 滤波后的概率分布
        """
        measurement = np.array(measurement)

        if not self.is_initialized:
            self.x = measurement.copy()
            self.is_initialized = True
            return self.x

        # 预测步骤
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # 更新步骤
        y = measurement - self.H @ x_pred  # 残差
        S = self.H @ P_pred @ self.H.T + self.R  # 残差协方差
        K = P_pred @ self.H.T @ np.linalg.inv(S)  # 卡尔曼增益

        self.x = x_pred + K @ y
        self.P = (np.eye(self.num_classes) - K @ self.H) @ P_pred

        # 确保概率和为1且非负
        self.x = np.maximum(self.x, 0)
        self.x = self.x / np.sum(self.x)

        return self.x.copy()


# 移动平均滤波器（作为对比）
class MovingAverageFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = []

    def update(self, prediction):
        self.history.append(prediction)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        return np.mean(self.history, axis=0)


# 从单个CSV文件加载数据
def load_single_csv_sliding(csv_file_path, seq_len=200, step=50, mode='last', max_samples=None):
    """
    从单个CSV文件加载数据并进行滑动窗口切片

    Args:
        csv_file_path: CSV文件路径
        seq_len: 序列长度
        step: 滑动步长
        mode: 标签选择模式 ('last' 或 'majority')
        max_samples: 最大样本数量限制（None表示不限制）

    Returns:
        X_tensor: 特征张量
        y_tensor: 标签张量
        segment_info: 每个片段的信息
        class_info: 类别信息字典
    """
    print(f"正在加载CSV文件: {csv_file_path}")

    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    print(f"原始数据形状: {df.shape}")

    # 检查必要的列
    channels = [f"Channel {i}" for i in range(1, 9)]

    if not all(col in df.columns for col in channels + ['label']):
        missing_cols = [col for col in channels + ['label'] if col not in df.columns]
        raise ValueError(f"CSV文件缺少必要的列: {missing_cols}")

    # 获取通道数据与标签
    data = df[channels].values
    labels = df['label'].values

    # 分析类别信息
    unique_labels = np.unique(labels)
    print(f"数据范围: {np.min(data):.4f} ~ {np.max(data):.4f}")
    print(f"原始标签分布: {dict(zip(unique_labels, np.bincount(labels)))}")
    print(f"发现的类别: {unique_labels}")

    X_list, y_list, segment_info = [], [], []

    # 使用滑动窗口方式切片
    total_segments = (len(data) - seq_len) // step + 1
    print(f"总共可生成 {total_segments} 个片段")

    if max_samples and total_segments > max_samples:
        print(f"限制样本数量为 {max_samples}")
        # 均匀采样
        indices = np.linspace(0, total_segments - 1, max_samples, dtype=int)
        selected_starts = [i * step for i in indices]
    else:
        selected_starts = range(0, len(data) - seq_len + 1, step)

    for i, start in enumerate(selected_starts):
        end = start + seq_len
        segment = data[start:end]
        label_seg = labels[start:end]

        # 标签策略：取最后一个样本的标签或窗口中多数类别
        if mode == 'last':
            label = label_seg[-1]
        else:  # majority
            label = np.bincount(label_seg).argmax()

        X_list.append(segment)
        y_list.append(label)
        segment_info.append({
            'segment_id': i,
            'start_idx': start,
            'end_idx': end,
            'label': label,
            'original_file': os.path.basename(csv_file_path)
        })

        # 打印进度
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1} 个片段")

    print(f"最终生成 {len(X_list)} 个片段")

    # 分析最终的类别分布
    final_labels = np.array(y_list)
    unique_final = np.unique(final_labels)
    label_counts = np.bincount(final_labels)

    print(f"片段标签分布: {dict(zip(unique_final, label_counts[unique_final]))}")

    # 创建类别信息字典
    class_info = {
        'unique_classes': unique_final,
        'num_classes': len(unique_final),
        'class_counts': {int(cls): int(label_counts[cls]) for cls in unique_final},
        'class_names': {0: "正常", 1: "左摔", 2: "右摔"}  # 默认映射
    }

    # 转换为Torch张量
    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_list), dtype=torch.long)

    return X_tensor.to(device), y_tensor.to(device), segment_info, class_info


# 实时预测函数（模拟实时流式预测）
def predict_with_filters(model, X_test, y_test, segment_info):
    """
    使用不同滤波器进行预测对比

    Returns:
        results: 包含所有预测结果的字典
    """
    model.eval()

    # 初始化滤波器
    kalman_filter = KalmanFilter(num_classes=3, process_noise=1e-3, measurement_noise=1e-1)
    moving_avg_filter = MovingAverageFilter(window_size=5)

    # 存储结果
    results = {
        'raw_predictions': [],
        'raw_probabilities': [],
        'kalman_predictions': [],
        'kalman_probabilities': [],
        'moving_avg_predictions': [],
        'moving_avg_probabilities': [],
        'true_labels': [],
        'attention_weights': [],
        'timestamps': [],
        'segment_info': segment_info
    }

    print("开始逐样本预测...")

    with torch.no_grad():
        for i, (sample, true_label) in enumerate(zip(X_test, y_test)):
            start_time = time.time()

            # 模型预测
            sample_batch = sample.unsqueeze(0)  # 添加batch维度
            output, attention = model(sample_batch)

            # 原始预测结果
            raw_probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            raw_pred = np.argmax(raw_probs)

            # 卡尔曼滤波
            kalman_probs = kalman_filter.update(raw_probs)
            kalman_pred = np.argmax(kalman_probs)

            # 移动平均滤波
            moving_avg_probs = moving_avg_filter.update(raw_probs)
            moving_avg_pred = np.argmax(moving_avg_probs)

            # 存储结果
            results['raw_predictions'].append(raw_pred)
            results['raw_probabilities'].append(raw_probs)
            results['kalman_predictions'].append(kalman_pred)
            results['kalman_probabilities'].append(kalman_probs)
            results['moving_avg_predictions'].append(moving_avg_pred)
            results['moving_avg_probabilities'].append(moving_avg_probs)
            results['true_labels'].append(true_label.cpu().item())
            results['attention_weights'].append(attention.cpu().numpy()[0])
            results['timestamps'].append(time.time() - start_time)

            # 每50个样本打印一次进度
            if (i + 1) % 50 == 0:
                print(f"已处理 {i + 1}/{len(X_test)} 个样本")

    return results


# 评估和可视化结果（修复版本）
def evaluate_and_visualize(results, class_info, output_dir):
    """评估不同方法的性能并可视化"""

    # 根据实际存在的类别动态生成类别名称
    unique_classes = class_info['unique_classes']
    all_class_names = {0: "正常", 1: "左摔", 2: "右摔"}
    class_names = [all_class_names[cls] for cls in unique_classes]

    print(f"实际存在的类别: {unique_classes}")
    print(f"类别名称: {class_names}")

    true_labels = results['true_labels']

    # 计算各方法的准确率
    raw_acc = accuracy_score(true_labels, results['raw_predictions'])
    kalman_acc = accuracy_score(true_labels, results['kalman_predictions'])
    moving_avg_acc = accuracy_score(true_labels, results['moving_avg_predictions'])

    print(f"\n=== 性能对比 ===")
    print(f"原始预测准确率: {raw_acc:.4f}")
    print(f"卡尔曼滤波准确率: {kalman_acc:.4f}")
    print(f"移动平均滤波准确率: {moving_avg_acc:.4f}")
    print(f"卡尔曼滤波提升: {kalman_acc - raw_acc:.4f}")
    print(f"移动平均滤波提升: {moving_avg_acc - raw_acc:.4f}")

    # 创建大图
    plt.figure(figsize=(20, 15))

    # 1. 准确率对比柱状图
    plt.subplot(4, 4, 1)
    methods = ['原始预测', '卡尔曼滤波', '移动平均']
    accuracies = [raw_acc, kalman_acc, moving_avg_acc]
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4']

    bars = plt.bar(methods, accuracies, color=colors, alpha=0.7)
    plt.title('不同方法准确率对比', fontsize=12, fontweight='bold')
    plt.ylabel('准确率')
    plt.ylim([min(accuracies) - 0.02, max(accuracies) + 0.02])

    # 在柱状图上添加数值
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

    # 2-4. 混淆矩阵对比
    methods_data = [
        ('原始预测', results['raw_predictions']),
        ('卡尔曼滤波', results['kalman_predictions']),
        ('移动平均', results['moving_avg_predictions'])
    ]

    for idx, (method_name, predictions) in enumerate(methods_data):
        plt.subplot(4, 4, idx + 2)
        cm = confusion_matrix(true_labels, predictions, labels=unique_classes)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{method_name} - 混淆矩阵')
        plt.xlabel('预测')
        plt.ylabel('真实')

    # 5-6. 预测概率时序图
    sample_range = min(300, len(results['raw_probabilities']))
    x_axis = range(sample_range)

    plt.subplot(4, 4, 5)
    for class_idx, class_name in enumerate(class_names):
        actual_class_idx = unique_classes[class_idx]
        raw_probs = [p[actual_class_idx] for p in results['raw_probabilities'][:sample_range]]
        plt.plot(x_axis, raw_probs, label=f'{class_name}', alpha=0.8, linewidth=1.5)

    plt.title('原始预测概率时序图')
    plt.xlabel('样本序号')
    plt.ylabel('概率')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 4, 6)
    for class_idx, class_name in enumerate(class_names):
        actual_class_idx = unique_classes[class_idx]
        kalman_probs = [p[actual_class_idx] for p in results['kalman_probabilities'][:sample_range]]
        plt.plot(x_axis, kalman_probs, label=f'{class_name}', alpha=0.8, linewidth=1.5)

    plt.title('卡尔曼滤波概率时序图')
    plt.xlabel('样本序号')
    plt.ylabel('概率')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 7. 预测结果时序对比
    plt.subplot(4, 4, 7)
    plt.plot(x_axis, [results['true_labels'][i] for i in range(sample_range)],
             'k-', label='真实标签', linewidth=2, alpha=0.8)
    plt.plot(x_axis, [results['raw_predictions'][i] for i in range(sample_range)],
             'r--', label='原始预测', alpha=0.7)
    plt.plot(x_axis, [results['kalman_predictions'][i] for i in range(sample_range)],
             'g-', label='卡尔曼滤波', alpha=0.7)

    plt.title('预测结果时序对比')
    plt.xlabel('样本序号')
    plt.ylabel('类别')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 设置y轴刻度为实际存在的类别
    plt.yticks(unique_classes, [class_names[i] for i in range(len(unique_classes))])

    # 8. 预测稳定性分析
    plt.subplot(4, 4, 8)

    def calculate_changes(predictions):
        changes = 0
        for i in range(1, len(predictions)):
            if predictions[i] != predictions[i - 1]:
                changes += 1
        return changes / len(predictions) if len(predictions) > 1 else 0

    raw_changes = calculate_changes(results['raw_predictions'])
    kalman_changes = calculate_changes(results['kalman_predictions'])
    moving_avg_changes = calculate_changes(results['moving_avg_predictions'])

    change_rates = [raw_changes, kalman_changes, moving_avg_changes]
    bars = plt.bar(methods, change_rates, color=colors, alpha=0.7)
    plt.title('预测稳定性对比\n(变化率越低越稳定)')
    plt.ylabel('预测变化率')

    for bar, rate in zip(bars, change_rates):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f'{rate:.4f}', ha='center', va='bottom', fontweight='bold')

    # 9. 注意力权重可视化
    plt.subplot(4, 4, 9)
    avg_attention = np.mean(results['attention_weights'], axis=0).flatten()
    channels = [f'通道{i + 1}' for i in range(len(avg_attention))]

    bars = plt.bar(channels, avg_attention, color='skyblue', alpha=0.7)
    plt.title('平均注意力权重分布')
    plt.xlabel('EMG通道')
    plt.ylabel('注意力权重')
    plt.xticks(rotation=45)

    # 标出最重要的通道
    max_idx = np.argmax(avg_attention)
    bars[max_idx].set_color('red')
    bars[max_idx].set_alpha(0.8)

    # 10. 处理时间统计
    plt.subplot(4, 4, 10)
    avg_time = np.mean(results['timestamps']) * 1000
    std_time = np.std(results['timestamps']) * 1000

    plt.bar(['平均处理时间'], [avg_time], color='orange', alpha=0.7)
    plt.errorbar(['平均处理时间'], [avg_time], yerr=[std_time],
                 fmt='none', color='black', capsize=5)
    plt.title(f'单样本处理时间\n{avg_time:.2f}±{std_time:.2f} ms')
    plt.ylabel('时间 (ms)')

    # 11. 类别分布
    plt.subplot(4, 4, 11)
    unique, counts = np.unique(true_labels, return_counts=True)
    plt.pie(counts, labels=[all_class_names[i] for i in unique], autopct='%1.1f%%',
            colors=['lightblue', 'lightgreen', 'lightcoral'][:len(unique)])
    plt.title('测试数据类别分布')

    # 12. 准确率提升详细对比
    plt.subplot(4, 4, 12)
    improvements = [0, kalman_acc - raw_acc, moving_avg_acc - raw_acc]
    colors_imp = ['gray', 'green' if improvements[1] > 0 else 'red',
                  'green' if improvements[2] > 0 else 'red']

    bars = plt.bar(['基准', '卡尔曼滤波', '移动平均'], improvements,
                   color=colors_imp, alpha=0.7)
    plt.title('准确率提升量对比')
    plt.ylabel('准确率提升')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    for bar, imp in zip(bars, improvements):
        if imp != 0:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                     f'{imp:+.4f}', ha='center', va='bottom', fontweight='bold')

    # 13. 各类别预测准确率对比
    plt.subplot(4, 4, 13)
    class_accuracies = {}
    for method_name, predictions in methods_data:
        accs = []
        for cls in unique_classes:
            mask = np.array(true_labels) == cls
            if np.sum(mask) > 0:
                acc = accuracy_score(np.array(true_labels)[mask], np.array(predictions)[mask])
                accs.append(acc)
            else:
                accs.append(0)
        class_accuracies[method_name] = accs

    x_pos = np.arange(len(class_names))
    width = 0.25

    for i, (method_name, accs) in enumerate(class_accuracies.items()):
        plt.bar(x_pos + i * width, accs, width, label=method_name, alpha=0.7)

    plt.title('各类别预测准确率对比')
    plt.xlabel('类别')
    plt.ylabel('准确率')
    plt.xticks(x_pos + width, class_names)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 14. 预测置信度分析
    plt.subplot(4, 4, 14)
    raw_confidences = [np.max(p) for p in results['raw_probabilities']]
    kalman_confidences = [np.max(p) for p in results['kalman_probabilities']]

    plt.hist(raw_confidences, bins=20, alpha=0.5, label='原始预测', density=True)
    plt.hist(kalman_confidences, bins=20, alpha=0.5, label='卡尔曼滤波', density=True)
    plt.title('预测置信度分布')
    plt.xlabel('最大概率值')
    plt.ylabel('密度')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 15. 错误分析
    plt.subplot(4, 4, 15)
    raw_errors = np.array(true_labels) != np.array(results['raw_predictions'])
    kalman_errors = np.array(true_labels) != np.array(results['kalman_predictions'])

    error_types = ['原始预测错误', '卡尔曼滤波错误', '两者都错误', '两者都正确']
    error_counts = [
        np.sum(raw_errors & ~kalman_errors),  # 只有原始错误
        np.sum(~raw_errors & kalman_errors),  # 只有卡尔曼错误
        np.sum(raw_errors & kalman_errors),  # 两者都错误
        np.sum(~raw_errors & ~kalman_errors)  # 两者都正确
    ]

    plt.pie(error_counts, labels=error_types, autopct='%1.1f%%', startangle=90)
    plt.title('错误分析')

    # 16. 滤波器效果对比（移动平均 vs 卡尔曼）
    plt.subplot(4, 4, 16)
    sample_subset = min(100, len(results['raw_probabilities']))
    x_subset = range(sample_subset)

    # 选择第一个存在的类别进行展示
    class_to_show = unique_classes[0]

    raw_probs_subset = [p[class_to_show] for p in results['raw_probabilities'][:sample_subset]]
    kalman_probs_subset = [p[class_to_show] for p in results['kalman_probabilities'][:sample_subset]]
    moving_avg_probs_subset = [p[class_to_show] for p in results['moving_avg_probabilities'][:sample_subset]]

    plt.plot(x_subset, raw_probs_subset, 'r-', alpha=0.7, label='原始预测', linewidth=1)
    plt.plot(x_subset, kalman_probs_subset, 'g-', alpha=0.8, label='卡尔曼滤波', linewidth=2)
    plt.plot(x_subset, moving_avg_probs_subset, 'b-', alpha=0.8, label='移动平均', linewidth=2)

    plt.title(f'滤波效果对比 - {all_class_names[class_to_show]}类概率')
    plt.xlabel('样本序号')
    plt.ylabel('概率')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prediction_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 图像已保存到: {os.path.join(output_dir, 'prediction_comparison.png')}")

    # 保存详细报告
    with open(os.path.join(output_dir, "prediction_report.txt"), "w", encoding="utf-8") as f:
        f.write("=== EMG单文件预测性能对比报告 ===\n\n")
        f.write(f"测试文件: {results['segment_info'][0]['original_file']}\n")
        f.write(f"测试样本数量: {len(true_labels)}\n")
        f.write(f"实际存在的类别: {unique_classes}\n")
        f.write(f"类别分布: {class_info['class_counts']}\n\n")

        f.write("=== 准确率对比 ===\n")
        f.write(f"原始预测准确率: {raw_acc:.4f}\n")
        f.write(f"卡尔曼滤波准确率: {kalman_acc:.4f} (提升: {kalman_acc - raw_acc:+.4f})\n")
        f.write(f"移动平均滤波准确率: {moving_avg_acc:.4f} (提升: {moving_avg_acc - raw_acc:+.4f})\n\n")

        f.write("=== 预测稳定性 ===\n")
        f.write(f"原始预测变化率: {raw_changes:.4f}\n")
        f.write(f"卡尔曼滤波变化率: {kalman_changes:.4f}\n")
        f.write(f"移动平均滤波变化率: {moving_avg_changes:.4f}\n\n")

        f.write("=== 处理性能 ===\n")
        f.write(f"平均处理时间: {avg_time:.2f} ± {std_time:.2f} ms\n")
        f.write(f"总处理时间: {sum(results['timestamps']):.2f} 秒\n\n")

        f.write("=== 注意力权重分析 ===\n")
        for i, weight in enumerate(avg_attention):
            f.write(f"通道{i + 1}: {weight:.4f}\n")
        f.write(f"最重要通道: 通道{np.argmax(avg_attention) + 1} (权重: {np.max(avg_attention):.4f})\n\n")

        # 只为实际存在的类别生成报告
        for method_name, predictions in methods_data:
            f.write(f"=== {method_name} 详细报告 ===\n")
            try:
                report = classification_report(
                    true_labels,
                    predictions,
                    labels=unique_classes,
                    target_names=class_names,
                    digits=4,
                    zero_division=0
                )
                f.write(report)
            except Exception as e:
                f.write(f"报告生成错误: {e}\n")
            f.write("\n\n")


# 保存预测结果到CSV（修复版本）
def save_predictions_to_csv(results, class_info, output_dir):
    """保存所有预测结果到CSV文件"""

    # 创建结果DataFrame
    df_results = pd.DataFrame()

    # 基本信息
    for i, info in enumerate(results['segment_info']):
        row_data = {
            'segment_id': info['segment_id'],
            'original_file': info['original_file'],
            'start_idx': info['start_idx'],
            'end_idx': info['end_idx'],
            'true_label': results['true_labels'][i],
            'raw_prediction': results['raw_predictions'][i],
            'kalman_prediction': results['kalman_predictions'][i],
            'moving_avg_prediction': results['moving_avg_predictions'][i],
            'processing_time_ms': results['timestamps'][i] * 1000,
        }

        # 添加所有概率（确保有3个类别的概率）
        for j in range(3):
            row_data[f'raw_prob_class_{j}'] = results['raw_probabilities'][i][j]
            row_data[f'kalman_prob_class_{j}'] = results['kalman_probabilities'][i][j]
            row_data[f'moving_avg_prob_class_{j}'] = results['moving_avg_probabilities'][i][j]

        # 注意力权重
        for j in range(8):
            row_data[f'attention_ch_{j + 1}'] = results['attention_weights'][i][j]

        df_results = pd.concat([df_results, pd.DataFrame([row_data])], ignore_index=True)

    # 保存到CSV
    output_file = os.path.join(output_dir, "prediction_results.csv")
    df_results.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✅ 预测结果已保存到: {output_file}")


# 主预测函数
def main_predict():
    # 设置路径
    model_path = r"C:\Users\EMG\Desktop\肌电数据\训练结果_左腿\run_20251021_011700\best_model.pth" # 替换为你的模型路径
    csv_file_path = r"C:\Users\EMG\Desktop\肌电数据\标签完\左腿\L1745487557_labeled.csv"  # 替换为你的CSV文件路径

    # 创建输出目录
    output_root = r"C:\Users\EMG\Desktop\肌电数据\预测结果_左腿"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    output_dir = os.path.join(output_root, f"prediction_{csv_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"预测结果将保存到: {output_dir}")

    # 加载模型
    print("正在加载模型...")
    model = EMGLSTMClassifierWithAttention(input_size=8, num_classes=3).to(device)

    # 加载训练好的权重
    try:
        state_dict = torch.load(model_path, map_location=device)
        # 如果模型是用DataParallel训练的，需要处理键名
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        model.load_state_dict(state_dict)
        print("✅ 模型加载成功!")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 加载测试数据
    print("正在加载测试数据...")
    try:
        X_test, y_test, segment_info, class_info = load_single_csv_sliding(
            csv_file_path=csv_file_path,
            seq_len=200,
            step=50,
            mode='last',
            max_samples=500  # 限制最大样本数，可以根据需要调整或设为None
        )
        print(f"✅ 测试数据加载完成! 样本数量: {len(X_test)}")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 进行预测
    print("开始预测...")
    start_time = time.time()
    try:
        results = predict_with_filters(model, X_test, y_test, segment_info)
        total_time = time.time() - start_time

        print(f"✅ 预测完成! 总耗时: {total_time:.2f} 秒")
        print(f"平均每样本耗时: {total_time / len(X_test) * 1000:.2f} ms")
    except Exception as e:
        print(f"❌ 预测过程出错: {e}")
        return

    # 评估和可视化
    print("正在生成分析报告...")
    try:
        evaluate_and_visualize(results, class_info, output_dir)
    except Exception as e:
        print(f"❌ 可视化过程出错: {e}")
        return

    # 保存详细结果
    try:
        save_predictions_to_csv(results, class_info, output_dir)
    except Exception as e:
        print(f"❌ 保存CSV过程出错: {e}")
        return

    print(f"✅ 所有结果已保存到: {output_dir}")


if __name__ == '__main__':
    main_predict()