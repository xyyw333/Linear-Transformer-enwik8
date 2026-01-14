import re
import matplotlib.pyplot as plt

# 1. 读取日志文件
log_path = 'train.log'
losses = []

# 2. 使用正则表达式提取 loss 数值
# 匹配格式如: training loss: 1.23456
pattern = re.compile(r"training loss: (\d+\.\d+)")

with open(log_path, 'r') as f:
    for line in f:
        match = pattern.search(line)
        if match:
            losses.append(float(match.group(1)))

# 3. 绘图
if losses:
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss', color='blue', alpha=0.7)
    
    # 如果数据点太多，可以做个平滑处理（移动平均）
    if len(losses) > 100:
        window = 50
        smooth_losses = [sum(losses[i:i+window])/window for i in range(len(losses)-window)]
        plt.plot(smooth_losses, label=f'Smoothed (win={window})', color='red')

    plt.title('Linear Transformer Training Loss (Improved Architecture)')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 4. 保存图片
    plt.savefig('loss.png', dpi=300)
    print(f"成功从日志提取了 {len(losses)} 个数据点，图片已保存为 loss.png")
else:
    print("未能从日志中找到 loss 数据，请检查日志格式。")
