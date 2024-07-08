import pandas as pd
import os

# 定义步长
step = 100

# 创建大文件夹
step_folder = f"Step_{step}"
os.makedirs(step_folder, exist_ok=True)

# 遍历每个Move文件
for move_num in range(1, 5):
    move_file = f"./Datas/Move{move_num}.xlsx"
    move_df = pd.read_excel(move_file, usecols=["Channel"])

    # 创建Move文件夹
    move_folder = os.path.join(step_folder, f"Move{move_num}")
    os.makedirs(move_folder, exist_ok=True)

    # 分割数据并保存到新的文件
    for index, start in enumerate(range(0, len(move_df), step), 1):
        end = min(start + step, len(move_df))
        step_df = move_df.iloc[start:end]
        output_file = os.path.join(move_folder, f"Move{move_num}_{index}.xlsx")
        step_df.to_excel(output_file, index=False)

print(f"已完成step为{step}的分组")