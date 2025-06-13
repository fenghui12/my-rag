import os
import time
import langchain # 用于 langchain.debug

# LangChain 组件导入 (根据你之前的修复)
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # 确保已安装 langchain-huggingface
from langchain_community.vectorstores import Chroma
# from langchain.chains import RetrievalQA # 我们不再直接使用这个链
# from langchain_google_genai import ChatGoogleGenerativeAI # 我们将用 google.genai.Client

# 导入 Google GenAI SDK
from google import genai as google_genai_sdk # 使用别名以区分

langchain.debug = True # 可以保留，看看检索部分的日志

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Semi-manual RAG Script started.")

# 0. API Key 设置 (确保 GOOGLE_API_KEY 或 GEMINI_API_KEY 已设置)
google_api_key_value = os.getenv("GOOGLE_API_KEY")
if not google_api_key_value and os.getenv("GEMINI_API_KEY"):
    google_api_key_value = os.getenv("GEMINI_API_KEY")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Using GEMINI_API_KEY for Google API.")

if not google_api_key_value:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 错误: GOOGLE_API_KEY 或 GEMINI_API_KEY 环境变量未设置！")
    exit()
else:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Google API Key 环境变量已找到。")

# --- RAG 的检索部分 (与之前类似) ---
# 1. 加载文档
print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 步骤 1: 加载本地文档...")
# ... (与之前相同的文档加载逻辑) ...
knowledge_base_dir = "./knowledge_base"
loader = DirectoryLoader(knowledge_base_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
try:
    documents = loader.load()
    if not documents: exit() # 简化错误处理
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 成功加载了 {len(documents)} 个文档。")
except Exception as e: print(f"加载文档失败: {e}"); exit()

# 2. 文档分块
print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 步骤 2: 文档分块...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
try:
    split_docs = text_splitter.split_documents(documents)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 文档被分成了 {len(split_docs)} 个块。")
except Exception as e: print(f"文档分块失败: {e}"); exit()

# 3. 文本嵌入与存入向量数据库
print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 步骤 3: 文本嵌入与存入向量数据库...")
embedding_model_name = "all-MiniLM-L6-v2"
try:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    persist_directory = 'db_chroma_manual' # 用一个新目录，避免与之前的冲突
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 正在创建/加载 Chroma 向量数据库于 '{persist_directory}'...")
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    # vectordb.persist() # Chroma 0.4.x 后自动持久化
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 文本块已嵌入并存入 Chroma 数据库。")
except Exception as e: print(f"文本嵌入或存入向量数据库失败: {e}"); import traceback; traceback.print_exc(); exit()

# 4. 创建检索器
print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 步骤 4: 创建检索器...")
try:
    retriever = vectordb.as_retriever(search_kwargs={"k": 3}) # 获取3个相关块
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 检索器创建成功。")
except Exception as e: print(f"创建检索器失败: {e}"); exit()

# --- 手动调用 Gemini API 进行生成 ---
# 5. 初始化 Google GenAI Client
print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 步骤 5: 初始化 Google GenAI Client...")
gemini_model_to_use = "gemini-2.0-flash" # 使用你测试成功的模型名称
try:
    # 注意：google.genai.Client() 初始化时可以直接传 api_key，
    # 或者如果 genai.configure(api_key=...) 已被调用过也可以。
    # 为确保清晰，我们在这里直接传递。
    client = google_genai_sdk.Client(api_key=google_api_key_value)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Google GenAI Client ({gemini_model_to_use}) 初始化成功。")
except Exception as e:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 初始化 Google GenAI Client 失败: {e}")
    import traceback; traceback.print_exc(); exit()


# 6. 进行提问与手动生成
print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 步骤 6: 开始提问 (手动 RAG)！输入 'exit' 退出程序。")
while True:
    query = input(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 请输入你的问题: ")
    if query.lower() == 'exit':
        break
    if not query.strip():
        continue

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 用户查询: '{query}'")

    # 6a. 使用 LangChain 检索器获取上下文
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 正在使用 LangChain 检索器获取相关文档块...")
    try:
        retrieved_docs = retriever.invoke(query)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 检索器返回了 {len(retrieved_docs)} 个文档块。")
        if not retrieved_docs:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 未检索到相关文档，将直接向 LLM 提问（无上下文）。")
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 调用检索器失败: {e}")
        retrieved_docs = [] # 如果检索失败，则没有上下文

    # 6b. 构造 Prompt (上下文 + 问题)
    context_for_llm = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # 你可以设计自己的 Prompt 模板
    # 这是 LangChain StuffDocumentsChain 的一个简化版本
    system_instruction = "Use the following pieces of context to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer."
    # 对于 google.genai.Client().models.generate_content(), contents 是一个列表
    # 我们可以把所有内容都放在一个字符串里，作为列表的唯一元素
    # 或者，如果模型支持聊天格式，可以构造更复杂的列表
    # 我们先尝试简单的方式：将所有内容合并成一个字符串
    if context_for_llm:
        full_prompt_content = f"{system_instruction}\n\nContext:\n{context_for_llm}\n\nQuestion: {query}\n\nAnswer:"
    else:
        full_prompt_content = f"{system_instruction}\n\nQuestion: {query}\n\nAnswer: (No context provided)" # 无上下文时的提示

    contents_for_gemini = [full_prompt_content] # 官方文档格式

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 准备发送给 Gemini API 的 Contents (预览):")
    print(f"  {contents_for_gemini[0][:350]}...") # 打印部分内容

    # 6c. 使用 google.genai.Client() 调用 LLM
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 正在使用 google.genai.Client() 调用 Gemini API ({gemini_model_to_use})...")
    api_call_start_time = time.time()
    try:
        response = client.models.generate_content(
            model=gemini_model_to_use,
            contents=contents_for_gemini
            # 你也可以在这里添加 config=google_genai_sdk.types.GenerateContentConfig(...) 来控制参数
        )
        api_call_end_time = time.time()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Gemini API 调用完成，耗时: {api_call_end_time - api_call_start_time:.2f} 秒。")

        if response and hasattr(response, 'text'):
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Gemini 回答:")
            print(response.text)
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Gemini API 调用返回了意外的响应格式或空响应。响应对象: {response}")
            if hasattr(response, 'prompt_feedback'):
                 print(f"  Prompt Feedback: {response.prompt_feedback}")


    except Exception as e:
        api_call_end_time = time.time()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Gemini API 调用失败，耗时: {api_call_end_time - api_call_start_time:.2f} 秒。错误: {e}")
        import traceback
        traceback.print_exc()

print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Semi-manual RAG Script finished.")