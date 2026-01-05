import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用设备：{device}")

# ==== 注意力模块 ====
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

# ==== LSTM 模型 ====
class EMGLSTMWithAttention(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, num_classes=3, dropout=0.3):
        super(EMGLSTMWithAttention, self).__init__()
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
        logits = self.fc(last_step)
        return logits, attn_weights

# ==== 数据加载 ====
def load_emg_dataset_sliding(folder, seq_len=200, step=50, mode='last'):
    X_list, y_list = [], []
    for file in os.listdir(folder):
        if file.endswith("_labeled.csv"):
            df = pd.read_csv(os.path.join(folder, file))
            channels = [f"Channel {i}" for i in [1, 2, 3, 4, 5, 6, 7, 8]]

            if not all(col in df.columns for col in channels + ['label']):
                continue
            data = df[channels].values
            labels = df['label'].values
            for start in range(0, len(data) - seq_len + 1, step):
                end = start + seq_len
                segment = data[start:end]
                label_seg = labels[start:end]
                label = label_seg[-1] if mode == 'last' else np.bincount(label_seg).argmax()
                X_list.append(segment)
                y_list.append(label)
    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_list), dtype=torch.long)
    return X_tensor.to(device), y_tensor.to(device)

# ==== 训练函数 ====
def train_and_validate(model, X, y, output_dir, epochs=1000, lr=1e-3, batch_size=1024):
    y_np = y.cpu().numpy()
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_np), y=y_np)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_cpu, y_cpu = X.cpu(), y.cpu()
    X_train, X_val, y_train, y_val = train_test_split(X_cpu, y_cpu, test_size=0.2, stratify=y_cpu)
    X_train, X_val, y_train, y_val = X_train.to(device), X_val.to(device), y_train.to(device), y_val.to(device)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, y_val),
        batch_size=batch_size, shuffle=False
    )

    train_loss_list, val_loss_list, val_acc_list = [], [], []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            out, _ = model(xb)
            loss = loss_fn(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_loss_list.append(train_loss)

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                out, _ = model(xb)
                pred = torch.argmax(out, dim=1)
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(yb.cpu().numpy())
        val_acc = accuracy_score(val_labels, val_preds)
        val_loss = loss_fn(model(X_val)[0], y_val).item()
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        duration = time.time() - start_time
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 第 {epoch+1}/{epochs} 轮 - "
              f"训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f} | "
              f"验证准确率: {val_acc:.4f} | 耗时: {duration:.2f} 秒")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))
    return train_loss_list, val_loss_list, val_acc_list, val_preds, val_labels

# ==== 保存图表和 Attention 可视化 ====
def save_training_outputs(output_dir, train_loss, val_loss, val_acc, val_preds, val_labels):
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='训练损失')
    plt.plot(val_loss, label='验证损失')
    plt.title("训练 vs 验证损失")
    plt.xlabel("轮次")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(val_acc, label='验证准确率')
    plt.title("验证准确率变化")
    plt.xlabel("轮次")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    plt.close()

    cm = confusion_matrix(val_labels, val_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
    plt.xlabel("预测")
    plt.ylabel("真实")
    plt.title("混淆矩阵")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    report = classification_report(val_labels, val_preds, target_names=["正常", "左摔", "右摔"], digits=4)
    with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

def visualize_attention(model, X_val, output_dir):
    model.eval()
    with torch.no_grad():
        x_sample = X_val[0].unsqueeze(0)  # 取一条样本
        _, attn_weights = model(x_sample)
        attn = attn_weights.squeeze().cpu().numpy()
        plt.figure(figsize=(8, 5))
        plt.bar(range(1, len(attn) + 1), attn)
        plt.xlabel("肌肉通道编号")
        plt.ylabel("注意力权重")
        plt.title("模型关注的肌肉通道")
        plt.xticks(range(1, len(attn) + 1))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "attention_weights.png"))
        plt.close()
        print("✅ 已保存 Attention 权重图")

# ==== 主入口 ====
if __name__ == '__main__':
    folder_path = r"C:\Users\EMG\Desktop\肌电数据\标签完\左腿"
    output_root = r"C:\\Users\\EMG\\Desktop\\肌电数据\\训练结果_左腿"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_root, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    X, y = load_emg_dataset_sliding(folder=folder_path, seq_len=200, step=50, mode='last')
    model = EMGLSTMWithAttention(input_size=8, num_classes=3).to(device)
    train_loss, val_loss, val_acc, val_preds, val_labels = train_and_validate(model, X, y, output_dir)
    save_training_outputs(output_dir, train_loss, val_loss, val_acc, val_preds, val_labels)
    visualize_attention(model, X, output_dir)
    print(f"✅ 训练完成，所有结果保存在: {output_dir}")




