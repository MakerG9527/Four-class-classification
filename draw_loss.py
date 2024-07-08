import matplotlib.pyplot as plt

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['Times New Roman']

# 设置全局字体大小
plt.rcParams.update({'font.size': 20})

Title = 'Training Loss'
Space = 25  # 假设我们每隔space个epoch取一次数据
Step = 80
Epochs = 1000

def read_loss_txt(method, space=Space, step=Step,epochs=Epochs):
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

# 绘制不同方法的损失曲线
methods = ['MLP', 'CNN', 'RNN', 'LSTM', 'Transformer']
colors = ['b', 'g', 'r', 'c', 'm']  # 颜色列表
markers = ['o', '^', 's', 'p', '*']  # 标记列表

plt.figure(figsize=(10, 7))  # 可以指定图表大小

for method, color, marker in zip(methods, colors, markers):
    x, y = read_loss_txt(method)
    plt.plot(x, y, f'-{marker}', color=color, ms=5, label=method)

# 获取当前的Axes对象
ax = plt.gca()
# 只加粗左边和下边的坐标轴线
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

'''
# 隐藏上边和右边的坐标轴线
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
'''

# 添加网格线
plt.grid(True)
# 添加自定义网格线
plt.grid(color='lightgray', linestyle='--', linewidth=0.8, alpha=1)
#设置为浅灰色的虚线，线宽，半透明度。

plt.xlabel('Epoch', fontsize=24)
plt.ylabel(Title,fontsize=24)
plt.title(f'{Title}',fontsize=24)
plt.legend()

plt.savefig(f'{Title}.pdf', bbox_inches='tight')
plt.savefig(f'{Title}.svg', bbox_inches='tight')
plt.savefig(f'{Title}.png', bbox_inches='tight')
plt.show()