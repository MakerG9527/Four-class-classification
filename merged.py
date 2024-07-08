import matplotlib.pyplot as plt

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['Times New Roman']

# 设置全局字体大小
plt.rcParams.update({'font.size': 20})

Space = 15 # 假设我们每隔space个epoch取一次数据
Step = 80
Epochs = 1000

def read_txt(method,space=Space, step=Step, epochs=Epochs):
    file_path = f"./logs/{method}_training_logs_Step{step}_Epochs{epochs}.txt"
    with open(file_path, 'r') as file:
        lines = file.readlines()

    epoch = []
    metrics = {
        'Precision': [],
        'Recall': [],
        'F1 Score': [],
        'Mean Average Precision (mAP)': []
    }
    # 确保有足够的行来读取指标
    if len(lines) >= 5:
        for i, line in enumerate(lines):
            if line.startswith('Epoch'):
                #print(i, "\n")
                #stop point * 9
                if i > 1000 * 9:
                    break
                if (i / 9) % space == 0:
                    #print(i, "\n")
                    epoch.append(int(line.split()[1]))
                    # 假设指标始终在接下来的四行中
                    for metric_line in range(4,8):
                        next_line = lines[i + metric_line]
                        metric_name, value = next_line.split(':')
                        if metric_name in metrics:
                            metrics[metric_name].append(100 * float(value.strip()))

    return epoch, metrics

def read_loss_txt(method, space=30, step=Step, epochs=Epochs):
    file_path = f"{method}_training_logs_Step{step}_Epochs{epochs}.txt"
    with open(file_path, 'r') as file:
        lines = file.readlines()

    epoch = []
    value = []
    for i, line in enumerate(lines):
        if line.startswith('Epoch'):
            # 计算当前行是第几个epoch，然后根据space决定是否添加到列表
            if i > 1000 * 9:
                break
            if (i / 9 ) % space == 0:
                epoch.append(int(line.split()[1]))
                # 读取下一行以获取对应的Training Loss值
                next_line = lines[i + 1]
                value.append(float(next_line.split(':')[1].strip()))
    return epoch, value

def draw_subplot_1(method, i, fig):
    # 检查是否是Training Loss，并获取数据
    epoch, metrics = read_txt(method)

    # 获取当前的子图轴
    ax = fig.add_subplot(3, 2, i,sharex=axs[0,0],sharey=axs[0,0])

    # 绘制不同指标的曲线
    metrics_head = ['Precision', 'Recall', 'F1 Score', 'Mean Average Precision (mAP)']
    colors = ['b', 'g', 'r', 'c']  # 颜色列表
    markers = ['o', '^', 's', 'p']  # 标记列表
    for label, color, marker in zip(metrics_head, colors, markers):
        if metrics[label]:  # 确保有数据
            ax.plot(epoch, metrics[label], f'-{marker}', color=color, ms=8, label=label, linewidth=2)

    # 自定义图表
    ax.set_title(f'{method}', fontsize=24)
    if ax.get_xlabel() == "":
        ax.set_xlabel('Epoch', fontsize=20)
    if ax.get_ylabel() == "":
        ax.set_ylabel("Metrics (%)", fontsize=24)

    # 自定义轴样式
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.grid(True)
    ax.grid(color='lightgray', linestyle='--', linewidth=0.8, alpha=1)
    ax.legend(loc='lower right')  # 添加图例

def draw_subplot_2(method_list, i, fig):
    ax = fig.add_subplot(3, 2, i)
    # 绘制不同方法的损失曲线
    colors = ['b', 'g', 'r', 'c', 'm']  # 颜色列表
    markers = ['o', '^', 's', 'p', '*']  # 标记列表

    for method, color, marker in zip(method_list, colors, markers):
        epoch, value = read_loss_txt(method)
        ax.plot(epoch, value, f'-{marker}', color=color, ms=8, label=method, linewidth=2)

    # 自定义图表
    ax.set_title(f'Training Loss', fontsize=24)
    ax.set_xlabel('Epoch', fontsize=21)
    ax.set_ylabel("Loss", fontsize=24)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.grid(True)
    ax.grid(color='lightgray', linestyle='--', linewidth=0.8, alpha=1)
    ax.legend()  # 添加图例

# 创建整个图像
# 设置整个图形的大小和子图的布局
fig, axs = plt.subplots(3, 2, figsize=(10*2, 7*3),sharey=True,sharex=True,)  # 3行2列的子图布局，整个图形大小为30x14英寸393

method_list = ['MLP','CNN','RNN','LSTM','Transformer']

# 绘制所有子图
for i, method in enumerate(method_list, start=1):  # start=1 因为索引从1开始
    draw_subplot_1(method, i, fig)
    draw_subplot_2(method_list,6, fig)

index = ['A','B','C','D','E','F']
# 调整子图间距，确保序号有足够的空间显示
plt.subplots_adjust(bottom=0)
for i, ax in enumerate(axs.flatten(), start=1):
    if i <= len(index):  # 确保序号在列表范围内
        ax.text(0.5, -0.17, f'({index[i-1]})', fontsize=27, ha='center', transform=ax.transAxes, verticalalignment='top')

#plt.subplots_adjust(bottom=0.3, wspace=0.1, hspace=0.6)
plt.tight_layout()

# 保存图形为PDF文件
plt.savefig('merged.pdf', format='pdf', bbox_inches='tight')
# 显示图形
plt.show()