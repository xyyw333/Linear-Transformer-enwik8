import torch
from linear_attention_transformer import LinearAttentionTransformerLM
from linear_attention_transformer.autoregressive_wrapper import AutoregressiveWrapper

# --- 1. 设备与配置 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = '/home/wxy/linear-attention-transformer/improved_model.pt'

# --- 2. 初始化模型 ---
model = LinearAttentionTransformerLM(
    num_tokens = 256,
    dim = 512,
    depth = 6,
    heads = 8,
    max_seq_len = 4096, # 匹配你训练时的长度
    reversible = True,
    n_local_attn_heads = 4,      
    local_attn_window_size = 128 
)

# --- 3. 使用包装器 (这一步是关键) ---
# 包装后，模型会多出一层 .net 结构，这正好对应你之前的键名报错
model = AutoregressiveWrapper(model)
model.to(device)

# --- 4. 加载权重 ---
try:
    state_dict = torch.load(MODEL_PATH, map_location=device)
    # 如果你的权重里有 net.net. 前缀，而这里只包装了一层，可能需要简单处理
    # 尝试直接加载，如果还报错，使用之前提到的 new_state_dict 去除多余前缀
    model.load_state_dict(state_dict)
    print("✅ 权重加载成功")
except Exception as e:
    print(f"⚠️ 尝试自动修复键名加载...")
    new_state_dict = {k.replace('net.net.', 'net.'): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)

model.eval()

# --- 5. GPU 优化推理 ---
@torch.no_grad()
def infer(prime_text, length=800, temperature=0.7):
    input_ids = torch.tensor([ord(c) for c in prime_text], dtype=torch.long, device=device).unsqueeze(0)
    
    # 适配新的 autocast 写法以消除警告
    with torch.amp.autocast('cuda'):
        # 包装器提供的 generate 方法
        output_ids = model.generate(input_ids, length, temperature=temperature)
    
    generated_ids = output_ids[0].cpu().tolist()
    return "".join(map(chr, generated_ids))

if __name__ == "__main__":
    result = infer("[[Artificial Intelligence]] is", length=800)
    print(result)
