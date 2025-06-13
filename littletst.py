import os
import time
import langchain # 用于 langchain.debug
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_huggingface import HuggingFaceEmbeddings # 为了测试LLM，暂时不需要这个

# 开启 LangChain 的全局 Debug 模式，会打印更多内部信息
langchain.debug = True

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Direct LLM Call Test Script started.")

# 0. 检查 Google API Key
if not os.getenv("GOOGLE_API_KEY"):
    # 尝试从 GEMINI_API_KEY 获取 (如果你的环境变量是这个)
    if os.getenv("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] GOOGLE_API_KEY set from GEMINI_API_KEY.")
    else:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 错误: GOOGLE_API_KEY 或 GEMINI_API_KEY 环境变量未设置！")
        exit()
else:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] GOOGLE_API_KEY 环境变量已找到。")

# 1. 初始化 LLM (Gemini)
llm_model_name = "gemini-1.5-flash-latest" # 使用你通过 google.genai.Client() 测试成功的模型名称
print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 步骤 1: 初始化 LLM...")
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 尝试初始化 Gemini LLM，模型: {llm_model_name}...")
llm = None # 先声明
try:
    llm = ChatGoogleGenerativeAI(
        model=llm_model_name,
        temperature=0,
        # request_timeout=60 # 可以尝试添加超时参数，具体名称可能需要查文档或根据库版本调整
    )
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Gemini LLM ({llm_model_name}) 初始化成功。")
except Exception as e:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 初始化 Gemini LLM 失败: {e}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 请确保 GOOGLE_API_KEY 环境变量已正确设置并指向有效的 Gemini API Key，")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 并且模型名称 '{llm_model_name}' 是你账户下可用的。")
    import traceback
    traceback.print_exc()
    exit()

# 2. 构造 Prompt 并直接调用 LLM
print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 步骤 2: 直接测试 LLM 调用...")

# 从你之前 RAG debug 输出中提取的上下文和问题 (为了测试，可以先用这个)
context_from_debug = """Apples are a type of fruit. They are often red or green.
Bananas are also fruits, and they are yellow when ripe.

Apples are a type of fruit. They are often red or green.
Bananas are also fruits, and they are yellow when ripe.

Apples are a type of fruit. They are often red or green.
Bananas are also fruits, and they are yellow when ripe."""
question_from_debug = "what is apple"

# LangChain StuffDocumentsChain 的默认模板格式通常包含一个系统消息和后续的人类问题+上下文
# 方式一：模拟 LangChain Debug 输出中看到的 "prompts" 列表里的单个字符串结构
# 有些 LLM 封装会将整个内容构建为一个 HumanMessage 传递给底层 SDK
# 这是 LangChain debug 输出中 `prompts[0]` 的内容
direct_prompt_string = f"""System: Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----------------
{context_from_debug}
Human: {question_from_debug}"""

messages_for_llm_v1 = [HumanMessage(content=direct_prompt_string)]

# 方式二：更标准的 SystemMessage + HumanMessage 结构
system_instruction = "Use the following pieces of context to answer the user's question. \nIf you don't know the answer, just say that you don't know, don't try to make up an answer."
human_question_with_context = f"----------------\n{context_from_debug}\nQuestion: {question_from_debug}" # 将 "Human:" 改为 "Question:" 或直接融入

messages_for_llm_v2 = [
    SystemMessage(content=system_instruction),
    HumanMessage(content=human_question_with_context)
]

# --- 选择一种方式进行测试 ---
# 我们先用方式一，因为它最接近你 LangChain debug 输出中 `prompts` 数组里的内容
messages_to_send = messages_for_llm_v1
# messages_to_send = messages_for_llm_v2 # 如果方式一不行，可以尝试方式二

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 准备发送给 LLM 的消息 (messages_to_send):")
for msg in messages_to_send:
    print(f"  {type(msg).__name__} (内容预览): {msg.content[:350]}...") # 打印预览，增加长度

llm_call_start_time = time.time()
try:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 正在调用 llm.invoke(messages_to_send)...")
    response = llm.invoke(messages_to_send)
    llm_call_end_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 直接 LLM 调用完成，耗时: {llm_call_end_time - llm_call_start_time:.2f} 秒。")
    if response:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] LLM 响应对象类型: {type(response)}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] LLM 响应内容: {response.content}")
    else:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] LLM 调用返回了 None 或空响应。")

except Exception as e:
    llm_call_end_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 直接 LLM 调用失败，耗时: {llm_call_end_time - llm_call_start_time:.2f} 秒。错误: {e}")
    import traceback
    traceback.print_exc() # 打印详细的错误堆栈信息
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 请检查错误信息，特别是与 API 调用、模型限制或 Prompt 相关的内容。")

print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Direct LLM Call Test Script finished.")