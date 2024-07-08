import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_fscore_support, average_precision_score
import os

# 1. 准备数据集
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels - 1  # 将标签减去1，使其从0开始编号

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

step = 80
num_classes = 4 #四类数据
data = []
labels = []

for move_num in range(1, 5):
    move_folder = f"Step_{step}/Move{move_num}"
    for index in range(1, 250):  # 读取x个文件
        file_path = f"{move_folder}/Move{move_num}_{index}.xlsx"
        df = pd.read_excel(file_path, usecols=["Channel"])
        data.append(df.values.flatten())
        labels.append(move_num)

data = np.array(data)
labels = np.array(labels)

# 2. 数据预处理
scaler = StandardScaler()
data = scaler.fit_transform(data)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=42)

train_dataset = CustomDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_dataset = CustomDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) #随机打乱顺序
test_loader = DataLoader(test_dataset, batch_size=32)

#4. 构建RNN 模型
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout_prob=0.0, l2_reg=0.0):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        self.l2_reg = l2_reg

    def forward(self, x):
        x = x.unsqueeze(1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        x = self.fc_1(x)
        x = self.relu(x)
        out, _ = self.rnn(x, h0)
        #out = self.dropout(out)  # 添加 dropout
        out = self.fc(out[:, -1, :])
        return out

# 定义训练参数
input_dim = X_train.shape[1]  # 输入维度
hidden_dim = 128  # 隐藏状态的维度
num_layers = 4  # RNN层数
num_classes = 4  # 分类数量
batch_size = 32
num_epochs = 10 #完整遍历num_epochs次数据集
learning_rate = 0.001

#5. 初始化模型、损失函数和优化器初始化模型、损失函数和优化器
model = RNN(input_dim, hidden_dim, num_layers, num_classes, dropout_prob=0.02, l2_reg=0.01)
criterion = nn.CrossEntropyLoss() #交叉熵作为损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. 训练模型
train_losses = []
train_accuracies = []
epochs = 1000 #训练次数
# 定义日志文件的路径
log_dir = '../logs'
log_path = os.path.join(log_dir, f'RNN_training_logs_Step{step}_Epochs{epochs}.txt')

# 检查日志目录是否存在，如果不存在，则创建它
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

#将数据写入日志文件中
with open(log_path, 'w') as f:
    for epoch in range(epochs):
        # 打印每一轮信息
        print(f"Epoch {epoch + 1}")
        f.write(f"Epoch {epoch + 1}\n")

        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # 梯度清零
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # 打印训练损失到控制台并写入文件
        print(f"Training Loss: {train_loss}")
        f.write(f"Training Loss: {train_loss}\n")

        # 在所有epochs完成后评估模型
        model.eval()
        correct = 0
        total = 0
        predicted_labels = []
        true_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        accuracy = correct / total
        train_accuracies.append(accuracy)

        # 打印测试准确率到控制台并写入文件
        print(f"Test Accuracy: {accuracy}")
        f.write(f"Test Accuracy: {accuracy}\n")

        # 使用 Scikit-learn 计算精确率、召回率和 F1 分数
        report = classification_report(true_labels, predicted_labels, output_dict=True)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='macro')
        # 打印指标到控制台并写入文件
        print("Macro-average metrics:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        f.write("Macro-average metrics:\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

        # 计算平均精确率 (mAP)
        # 需要将类别标签转换为 one-hot 编码
        true_labels_onehot = np.eye(num_classes)[true_labels]
        predicted_scores = np.eye(num_classes)[predicted_labels]
        mAP = average_precision_score(true_labels_onehot, predicted_scores, average='macro')
        # 打印mAP到控制台并写入文件
        print(f"Mean Average Precision (mAP): {mAP:.4f}")
        f.write(f"Mean Average Precision (mAP): {mAP:.4f}\n")
        # 打印轮之间的分割线
        print("-----------------------------------")
        f.write("-----------------------------------\n")

# 8. 可视化波形数据和分类结果
def visualize_data(data, labels, class_names):
    num_samples = min(5, len(data)) #最大可视化数量
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 禁用梯度计算
        for i in range(num_samples):
            print(f"Sample {i+1}:")
            print("Label:", labels[i])
            print("Class name:", class_names)
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(data[i])
            plt.title(f"Waveform - Class {class_names[labels[i] - 1]}")
            plt.xlabel("Time")
            plt.ylabel("Amplitude")

            # 计算预测概率
            input_tensor = torch.tensor(data[i], dtype=torch.float32).unsqueeze(0)
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze().detach().numpy()

            plt.subplot(1, 2, 2)
            plt.bar(range(len(class_names)), probabilities)
            plt.xticks(range(len(class_names)), class_names)
            plt.title("Predicted Probabilities")
            plt.xlabel("Class")
            plt.ylabel("Probability")
            #plt.show()
        plt.tight_layout()
        plt.show()

class_names = ['Move 1', 'Move 2', 'Move 3', 'Move 4']
visualize_data(X_test, y_test, class_names)