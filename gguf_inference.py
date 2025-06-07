import os
import sys
import time
import traceback

# 强制使用 CPU 模式
os.environ["LLAMA_CPP_FORCE_CPU"] = "1"

# 使用绝对路径确保正确
MODEL_PATH = os.path.abspath(r"E:\deepseek-moe-16b\deepseek-moe-16b-chat.i1-Q4_K_M.gguf")

# 验证模型文件是否存在
if not os.path.exists(MODEL_PATH):
    print(f"错误: 模型文件不存在 - {MODEL_PATH}")
    print("请确保模型文件已下载并放在正确位置")
    sys.exit(1)

# 验证模型文件大小
expected_size = 9150441984  # 约 8.5GB
actual_size = os.path.getsize(MODEL_PATH)
if actual_size < expected_size * 0.95:  # 允许5%的差异
    print(f"警告: 模型文件大小异常 ({actual_size/1024/1024:.2f}MB)")
    print(f"预期大小: {expected_size/1024/1024:.2f}MB")
    print("文件可能不完整，建议重新下载")

print(f"尝试加载模型: {MODEL_PATH}")

try:
    from llama_cpp import Llama
except ImportError:
    print("错误: 无法导入 llama_cpp")
    print("尝试重新安装: pip install --force-reinstall llama-cpp-python")
    sys.exit(1)

try:
    print("正在加载模型...")
    start_time = time.time()
    
    # 尝试初始化模型
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,        # 减少上下文长度以节省内存
        n_threads=4,        # 减少线程数
        n_batch=128,        # 减少批处理大小
        use_mlock=False,    # 禁用内存锁定
        verbose=True,       # 启用详细日志
        n_gpu_layers=0      # 确保不使用 GPU
    )
    
    # 测试模型是否加载成功
    test_output = llm("测试", max_tokens=1)
    if "text" not in test_output['choices'][0]:
        raise RuntimeError("模型响应格式异常")
    
    load_time = time.time() - start_time
    print(f"模型加载成功! 耗时: {load_time:.2f}秒")
    
except Exception as e:
    print(f"\n加载模型失败: {str(e)}")
    print("\n详细错误信息:")
    traceback.print_exc()
    
    # 提供解决方案建议
    print("\n可能的解决方案:")
    print("1. 确保模型文件完整且未损坏")
    print("2. 尝试更小的模型 (如 Phi-3-mini)")
    print("3. 增加系统虚拟内存")
    print("4. 使用更轻量级的替代方案:")
    print("   pip install ctransformers")
    print("   或")
    print("   pip install transformers")
    
    sys.exit(1)

# 操作系统专家提示模板
def format_prompt(question):
    return f"""<|im_start|>system
你是一个操作系统专家，请用简洁专业的方式回答问题。
回答应包含核心概念和实际应用示例。<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""

def ask_os_expert(question):
    prompt = format_prompt(question)
    print(f"\n问题: {question}")
    print("思考中...")
    start = time.time()
    
    try:
        output = llm(
            prompt,
            max_tokens=150,    # 减少生成长度
            temperature=0.5,   # 降低创造性
            top_p=0.85,
            stop=["<|im_end|>"]
        )
        
        response = output['choices'][0]['text'].strip()
        latency = time.time() - start
        print(f"生成耗时: {latency:.2f}秒")
        return response
        
    except Exception as e:
        return f"生成回答时出错: {str(e)}"

# 简单测试问题
print("\n===== 运行简单测试 =====")
test_questions = [
    "什么是操作系统？",
    "解释进程和线程的区别"
]

for i, q in enumerate(test_questions, 1):
    print(f"\n测试问题 {i}: {q}")
    answer = ask_os_expert(q)
    print(f"测试回答: {answer[:200]}...")  # 只显示部分回答

# 内存状态报告
try:
    import psutil
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024 * 1024)
    print(f"\n当前内存使用: {mem_usage:.2f} MB")
except ImportError:
    print("\n安装 psutil 以查看内存使用: pip install psutil")

print("\n测试完成。如果成功，您可以开始完整问答。")