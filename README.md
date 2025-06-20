# 一、模型选择

需要对操作系统领域进行“专家级”的知识问答与检索，首先考虑到了deepseek 的大模型，然而在本地部署需要追求轻量化，直接使用全精度模型，既占用磁盘和内存，又没有必要，因为我们只需要专精于操作系统领域。因此，一个选择是deepseek 的稀疏模型moe-16b。  
另一个选择则是，Microsoft 开源的phi-3-mini或phi-4-mini，可以很好的适应操作系统方面的需求。而且，所需的模型参数更少，且同样支持中英文。一个可能的缺点是，这些模型在中文模式下，准确性还未得到全面的验证。  
综合考虑后，先选择了deepseek 的稀疏模型进行部署，后续可能会增加其他模型。

# 二、环境准备

# 1.安装Python
deepseek 官方推荐的版本在Python 3.8及以上，但经过实测，后续一些步骤需要Python ≥ 3.9，所以保险起见这里安装了3.10版本。注意不能使用过于新的版本，比如Python 3.13，因为很多库还没有适配新的版本  
安装完之后，把Python 3.10添加到环境变量中。（下一步会创建新的虚拟环境，所以也可以不添加，但为了方便以及后续的报错排查，这里还是选择添加了）  
# 2.创建一个新的虚拟环境
这里因为要使用cpu部署，所以命名为了deepseek-cpu。使用cmd（命令提示符）进行创建，创建完成后可以发现"C:\Users\Lenovo"多了一个文件夹\deepseek-cpu  
运行该新的虚拟环境，cmd指令为
```cmd
deepseek-cpu\Scripts\activate
```
成功进入后，cmd显示  
`(deepseek-cpu) C:\Users\Lenovo>`
# 3.安装基础依赖
在新生成的cmd对话框中，输入
```cmd
pip install torch>=2.1.1 transformers>=4.35.0 accelerate sentencepiece
```   
这是使用pip工具安装pytorch，如果未安装pip或者版本过低，可以
`install --uograde pip`  
torch包很大，需要长时间的下载  
这一步可能会产生报错，但只要torch下载完成就不影响。比如在transformers处报错，可以在cmd执行
```cmd
pip install --uograde transformers
```
某次尝试时这里没有报错，但后续步骤中会有提示“找不到transformers”，所以这里还是install一下transformers比较好
# 4.为了实现GPU加速推理
继续在cmd执行
```cmd
pip install vllm==0.3.0
```
等待进度条跑完，可能第二个进度条会报错error，但只要第一个进度条跑完是done就可以
# 5.为了实现CPU部署
继续在cmd执行
```cmd
pip install llama-cpp-python
```
等待进度条跑完，上一步的进度条即使有报错wheel（轮子）没有构建成功，这一步的进度条也会把wheel构建完成（done）
# 三、下载模型
一般采用hugging face下载比较多。虽然也可以继续通过cmd进行下载，但是有两个麻烦：一是要输入hf_打头的token，这个还是需要注册、登录hugging face并验证邮箱；二是国内无法访问，需要VPN  
即使这时打开VPN，cmd也会报错  
`InvalidSchema: missing dependencies for socks support`  
解决方案一是安装缺少的socks依赖，但是会报错  
`ERROR: Could not install packages due to an OSError: Missing dependencies for SOCKS support.`  
解决方案二是清除代理设置
```cmd
#Windows (CMD)
set http_proxy=
set https_proxy=
set ALL_PROXY=
```  
```psl
#Windows (PowerShell)
Remove-Item Env:http_proxy
Remove-Item Env:https_proxy
Remove-Item Env:ALL_PROXY
```
但是我们不知道代理的具体信息，没法重新设置  
所以我开启了VPN后直接在hugging face 官网上下载，网址如下：  
https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat/tree/main  
进来以后是这个样子：  
![image](https://github.com/user-attachments/assets/22c426e7-22cb-4f92-8402-3050d29e2728)  
我们点击file,下载其中的model。  
可以发现，因为模型很大，他分了7个分卷进行上传。我们为了轻量化，只需下载其中的  
```
pytorch_model-00001-of-00007.safetensors  # 主模型分片1

pytorch_model-00002-of-00007.safetensors  # 主模型分片2

tokenizer.json                           # 分词器配置

config.json                              # 模型配置

modeling_deepseek.py                     # 模型组装

configuration_deepseek.py                # 构造配置文件
```
即可。为什么可以省略其他文件？

其他分片包含的是模型的不同层，量化时会自动重建结构，只需要部分分片即可开始处理。

# 四、调试模型

## 1.编写推理文件mini_inference.py
```python

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
#量化配置
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

#加载本地模型（自动处理缺失分片）
model = AutoModelForCausalLM.from_pretrained(
    "./deepseek-moe-16b",
    quantization_config=quant_config,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("./deepseek-moe-16b")

#操作系统领域问答
question = "解释Linux中的进程调度算法CFS"
inputs = tokenizer(question, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

#加载本地模型（自动处理缺失分片）
model = AutoModelForCausalLM.from_pretrained(
    "./deepseek-moe-16b",
    quantization_config=quant_config,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("./deepseek-moe-16b")

#操作系统领域问答
question = "解释Linux中的进程调度算法CFS"
inputs = tokenizer(question, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 2.在deepseek-cpu中运行
```cmd
python mini_inference.py
```
提示bitsandbytes库没有正确安装或者无法识别到  
![image](https://github.com/user-attachments/assets/ab376fa9-5b0a-4a82-b7d5-dc358cd99f09)
若没有此报错，可以跳过2、3步骤  
解决方法：  
首先输入cmd指令，对原有bitsandbytes库进行卸载（如果有的话），我这里显示  
![image](https://github.com/user-attachments/assets/3f78fd22-f5ca-444e-8cfb-8f03f4c9f6f1)
说明我没有安装bitsandbytes库。所以第二步，安装预编译的Windows版bitsandbytes库,cmd指令  
```cmd
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
```
![image](https://github.com/user-attachments/assets/64e008c1-bab9-472d-950d-eb765aeaf052)
安装成功！
如果还是报错
![image](https://github.com/user-attachments/assets/db81525a-2cfb-4b88-abdc-ea72a9f2f8d3)
则根据提示升级
```cmd
pip install -U bitsandbytes
```
## 3.安装必要的编译依赖
```cmd
pip install setuptools wheel ninja
```
![image](https://github.com/user-attachments/assets/018a25bb-324f-4218-8787-062932b4962a)
如上，安装或更新了这个工具
```cmd
pip install accelerate
```
![image](https://github.com/user-attachments/assets/71e01935-0ca9-42d0-b1c4-bc05e55e4336)

## 4.若仍是出现报错，例如
![image](https://github.com/user-attachments/assets/fae1fe0f-44db-41ae-8e75-134912bac3a1)
则是因为bitsandbytes 0.46.0需要pytorch 2.7.0，但torchvision 0.16.0和torchaudio 2.1.0需要pytorch 2.1.0，它们冲突了
根据提示，解决方法是把所有的包都统一到2.3.0对应版本

依赖名称 | 版本 | 说明  
 ---- | ----- | :------:
 **torch** | 2.3.0 | 基础版本  
 **torchvision** | 0.18.0 | 专为PyTorch 2.3.x设计 
 **torchaudio** | 2.3.0 | 与PyTorch 2.3.x匹配 
 **bitsandbytes** | 0.43.0 | 兼容PyTorch 2.3.x 
 **transformers** | 4.40.0 | 最新稳定版 

都统一到2.3.0版本后，重新执行python mini_inference.py  
这里若通过，则继续使用使用这个即可顺利完成部署。（注意把.py里面模型的地址改为自己下载到的路径）  
然而，我这里报错RuntimeError:
```cmd
None of the available devices `available_devices = None` are supported by the bitsandbytes version you have installed:
`bnb_supported_devices = {'"cpu" (needs an Intel CPU and intel_extension_for_pytorch installed and compatible with the PyTorch version)',
'hpu', 'cuda', 'mps', 'xpu', 'npu'}`. Please check the docs to see if the backend you intend to use is available and how to install it:
https://huggingface.co/docs/bitsandbytes/main/en/installation#multi-backend
```
这个报错意思是，bitsandbytes 在 Windows 上默认不支持 CPU 推理，需要 Intel CPU 和特定扩展。  
可能的解决方法是，使用 CPU 优化的替代库
```cmd
pip uninstall bitsandbytes -y
pip install intel-extension-for-pytorch
pip install transformers[torch] accelerate
```
也就是安装intel-extension-for-pytorch优化，以达到支持的性能。
## 5.根据安装intel优化是否成功，分为两种情况：
### （1）安装成功
此时原来下载的模型可以继续使用，但需要修改推理文件，把bitsandbytes量化改为Intel优化：
```python
#cpu_inference.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import intel_extension_for_pytorch as ipex
import torch

#加载模型（不使用量化）
model = AutoModelForCausalLM.from_pretrained(
    "./deepseek-moe-16b",
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

#应用 Intel 优化
model = ipex.optimize(model)

tokenizer = AutoTokenizer.from_pretrained("./deepseek-moe-16b")

#操作系统领域问答
question = "解释Linux中的进程调度算法CFS"
inputs = tokenizer(question, return_tensors="pt").to(model.device)

#优化推理
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200)
    
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
并注意把.py里面模型的地址改为自己下载到的路径
### （2）安装不成功
像我这里，报错“找不到合适的版本”，说明cpu达不到intel优化包所需的要求
只能改用 GGUF 格式的模型 DeepSeek-MoE-16b-chat GGUF
deepseek的hugging face官方发布界面上这一模型暂时处于404-找不到的状态，我是从这里下载的

https://huggingface.co/mradermacher/deepseek-moe-16b-chat-i1-GGUF
![屏幕截图 2025-06-01 163655](https://github.com/user-attachments/assets/eb163cd2-0c94-409d-a395-a27912dddccb)
找到i1-Q4_K_M版本（它的特点是快速并且被推荐），点击左边的GGUF下载，如上图，共10.1GB，需要VPN

然后检查依赖是否安装，cmd执行命令pip install llama-cpp-python（前面安装过的话会提示already satisfied）

最后，修改配套的推理脚本（为了区分上面的Intel CPU模式，命名为gguf_inference.py）

```python
#gguf_inference.py
from llama_cpp import Llama
import time

#配置模型
model_path = "deepseek-moe-16b-chat.Q4_K_M.gguf"

print("正在加载模型...")
start_time = time.time()

llm = Llama(
    model_path=model_path,
    n_ctx=4096,        # 上下文长度
    n_threads=8,       # 使用8个CPU线程 (根据您的CPU核心数调整)
    n_batch=512,       # 批处理大小
    use_mlock=True,    # 锁定内存防止交换
    verbose=False
)

load_time = time.time() - start_time
print(f"模型加载完成，耗时: {load_time:.2f}秒")

#操作系统专家提示模板
def format_prompt(question):
    return f"""<|im_start|>system
你是一个操作系统专家，请用专业但易懂的语言回答以下问题。
请确保回答准确、详细，并包含实际应用示例。<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""

def ask_os_expert(question):
    prompt = format_prompt(question)
    print("\n思考中...")
    start = time.time()
    
    output = llm(
        prompt,
        max_tokens=300,    # 最大生成长度
        temperature=0.7,   # 创造性程度 (0-1)
        top_p=0.9,         # 核采样
        stop=["<|im_end|>"]  # 停止标记
    )
    
    latency = time.time() - start
    response = output['choices'][0]['text'].strip()
    
    print(f"生成耗时: {latency:.2f}秒")
    return response

#操作系统专业问题测试
questions = [
    "详细解释Linux的CFS(Completely Fair Scheduler)调度算法",
    "什么是虚拟内存？解释其工作原理和优点",
    "比较ext4和NTFS文件系统的优缺点",
    "解释DMA(直接内存访问)在I/O操作中的作用",
    "什么是写时复制(Copy-on-Write)？举例说明其在操作系统中的应用"
]

for i, q in enumerate(questions, 1):
    print(f"\n{'='*50}")
    print(f"问题 {i}: {q}")
    answer = ask_os_expert(q)
    print(f"\n答案: {answer}")

#交互模式
print("\n\n进入操作系统专家问答模式 (输入 '退出' 结束)")
while True:
    user_input = input("\n您的问题: ")
    if user_input.lower() in ["退出", "exit", "quit"]:
        break
    answer = ask_os_expert(user_input)
    print(f"\n专家回答: {answer}")
```
把模型加载的路径换成自己下载到的路径，然后cmd执行命令 python gguf_inference.py
# 五、测试结果
![测试结果1](https://github.com/user-attachments/assets/eed6907a-10ee-4f71-b9d8-828642e70525)
![测试结果2](https://github.com/user-attachments/assets/447c59cc-3f4d-4aa4-a6e2-228dbaf15570)
经过20s的模型加载，最终成功回答出了测试的问题
