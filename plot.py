import re
import matplotlib.pyplot as plt

# 定义要读取的三个日志文件路径
log_files = [
    'train_Baseline.log',
    'train_Conv_Only.log',
    'train_Full_Improved.log'
]

# 定义每条曲线的样式（颜色、标签），方便区分
plot_styles = [
    {'color': 'blue', 'label': 'train_Baseline Loss'},
    {'color': 'red', 'label': 'train_Conv_Only Loss'},
    {'color': 'green', 'label': 'train_Full_Improved Loss'}
]

# 正则表达式匹配 loss 数值（匹配 training loss: 1.23456 格式）
pattern = re.compile(r"training loss: (\d+\.\d+)")

# 创建画布
plt.figure(figsize=(10, 6))

# 循环读取每个日志文件并绘图
for idx, log_path in enumerate(log_files):
    losses = []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    losses.append(float(match.group(1)))
        
        if not losses:
            print(f"警告：{log_path} 中未找到 loss 数据，跳过该文件")
            continue
        
        # 绘制原始 loss 曲线
        plt.plot(losses, 
                 label=plot_styles[idx]['label'], 
                 color=plot_styles[idx]['color'], 
                 alpha=0.5,  # 原始曲线透明度调低，避免重叠看不清
                 linewidth=1)
        
        # 对数据量多的文件做平滑处理（移动平均）
        if len(losses) > 100:
            window = 50
            smooth_losses = [sum(losses[i:i+window])/window for i in range(len(losses)-window)]
            plt.plot(smooth_losses, 
                     label=f"{plot_styles[idx]['label']} (smoothed)", 
                     color=plot_styles[idx]['color'], 
                     linewidth=2)
        
        print(f"成功从 {log_path} 提取了 {len(losses)} 个 loss 数据点")
    
    except FileNotFoundError:
        print(f"错误：未找到文件 {log_path}，跳过该文件")

# 设置图表样式
plt.title('Training Loss Comparison')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend(loc='best')  # 自动选择最佳图例位置
plt.grid(True, linestyle='--', alpha=0.6)

# 保存图片
plt.savefig('loss_comparison.png', dpi=300, bbox_inches='tight')  # bbox_inches='tight' 防止图例被裁剪
plt.show()

print("对比图已保存为 loss_comparison.png")