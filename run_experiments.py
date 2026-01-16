import os
import subprocess
import matplotlib.pyplot as plt
import re
import pandas as pd
import math

# --- 实验配置 ---
EXPERIMENTS = {
    "Baseline": {"n_local_heads": 0, "gate": "False"},
    "Conv_Only": {"n_local_heads": 4, "gate": "False"},
    "Full_Improved": {"n_local_heads": 4, "gate": "True"}
}
SEQ_LEN_LIST = [512, 1024, 2048, 4096] # 用于生成效率对比图 

def run_training():
    """依次执行三个版本的训练"""
    for name, config in EXPERIMENTS.items():
        print(f"🚀 开始实验: {name}...")
        log_file = f"train_{name}.log"
        # 通过环境变量传递参数给 train.py
        env = os.environ.copy()
        env["EXP_VERSION"] = name
        env["LOCAL_HEADS"] = str(config["n_local_heads"])
        env["USE_GATE"] = config["gate"]
        
        # 执行训练脚本 (假设运行 3000 steps)
        with open(log_file, "w") as f:
            subprocess.run(["python", "./examples/enwik8_simple/train.py"], env=env, stdout=f, stderr=subprocess.STDOUT)
        print(f"✅ {name} 训练完成，日志已保存至 {log_file}")

def plot_convergence(log_files):
    """可视化收敛曲线 (模仿论文 Figure 2)"""
    plt.figure(figsize=(10, 6))
    for name, path in log_files.items():
        steps, bpc = [], []
        if not os.path.exists(path): continue
        with open(path, 'r') as f:
            for line in f:
                res = re.findall(r"step: (\d+),.*loss: ([\d.]+)", line)
                if res:
                    step, loss = int(res[0][0]), float(res[0][1])
                    steps.append(step)
                    bpc.append(loss / math.log(2)) # 转换为 BPC 
        
        plt.plot(steps, bpc, label=name)
    
    plt.yscale('log')
    plt.xlabel('Gradient Steps')
    plt.ylabel('BPC (Bits Per Character)')
    plt.title('Convergence Comparison on enwik8')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig('convergence_comparison.png')
    print("📈 收敛曲线已生成: convergence_comparison.png")

# --- 执行流程 ---
if __name__ == "__main__":
    # 1. 运行训练 (如果已经有日志可以跳过)
    # run_training() 
    
    # 2. 绘图
    log_map = {k: f"train_{k}.log" for k in EXPERIMENTS.keys()}
    plot_convergence(log_map)