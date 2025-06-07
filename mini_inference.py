# mini_inference.py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 量化配置
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 加载本地模型（自动处理缺失分片）
model = AutoModelForCausalLM.from_pretrained(
    "E:/deepseek-moe-16b",
    quantization_config=quant_config,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("E:/deepseek-moe-16b")

# 操作系统领域问答
question = "解释Linux中的进程调度算法CFS"
inputs = tokenizer(question, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))