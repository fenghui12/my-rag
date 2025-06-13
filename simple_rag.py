import os
import time # 导入 time 模块，可以用来打印时间戳或计算耗时

# 确保导入路径已根据之前的讨论更新
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings # 旧的，如果还没改
from langchain_huggingface import HuggingFaceEmbeddings # 新的，确保已安装 langchain-huggingface
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI # 确保导入正确

import langchain
langchain.debug = True # 开启 LangChain 的全局 Debug 模式，会打印更多内部信息

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Script started.")

# 0. 设置你的 Google API Key
# 确保 GOOGLE_API_KEY 环境变量已设置
# 例如: os.environ["GOOGLE_API_KEY"] = "你的GEMINI_API_KEY_值"
# 如果你的环境变量名是 GEMINI_API_KEY，可以这样设置：
# if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
#     os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
#     print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] GOOGLE_API_KEY set from GEMINI_API_KEY.")

if not os.getenv("GOOGLE_API_KEY"):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 错误: GOOGLE_API_KEY 环境变量未设置！")
    exit()
else:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] GOOGLE_API_KEY 环境变量已找到。")


# 1. 加载本地文档 (Load)
print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 步骤 1: 加载本地文档...")
knowledge_base_dir = "./knowledge_base"
loader = DirectoryLoader(knowledge_base_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
try:
    documents = loader.load()
    if not documents:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 在 '{knowledge_base_dir}' 目录下没有找到 .txt 文档。请创建一些文档。")
        exit()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 成功加载了 {len(documents)} 个文档。")
    # for i, doc in enumerate(documents):
    #     print(f"  文档 {i} metadata: {doc.metadata}")
    #     print(f"  文档 {i} content preview: {doc.page_content[:100]}...") # 打印前100个字符
except Exception as e:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 加载文档失败: {e}")
    exit()

# 2. 文档分块 (Split)
print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 步骤 2: 文档分块...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
try:
    split_docs = text_splitter.split_documents(documents)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 文档被分成了 {len(split_docs)} 个块。")
    # if split_docs:
    #     print(f"  第一个块 metadata: {split_docs[0].metadata}")
    #     print(f"  第一个块 content preview: {split_docs[0].page_content[:100]}...")
except Exception as e:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 文档分块失败: {e}")
    exit()

# 3. 文本嵌入与存入向量数据库 (Store)
print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 步骤 3: 文本嵌入与存入向量数据库...")
embedding_model_name = "all-MiniLM-L6-v2" # 或者你选择的中文模型
try:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 正在初始化 Embedding 模型: {embedding_model_name}...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Embedding 模型初始化成功。")

    persist_directory = 'db_chroma'
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 正在创建/加载 Chroma 向量数据库于 '{persist_directory}'...")
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist() # 确保持久化
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 文本块已嵌入并存入 Chroma 数据库。")
except Exception as e:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 文本嵌入或存入向量数据库失败: {e}")
    import traceback
    traceback.print_exc()
    exit()

# 4. 创建检索器 (Retriever)
print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 步骤 4: 创建检索器...")
try:
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 检索器创建成功 (k=3)。")
except Exception as e:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 创建检索器失败: {e}")
    exit()

# 5. 初始化 LLM
print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 步骤 5: 初始化 LLM...")
# 使用你通过 google.genai.Client() 测试成功的模型名称
llm_model_name = "gemini-2.0-flash" # 或者 "gemini-1.5-flash-latest" 如果那个也能工作
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 尝试初始化 Gemini LLM，模型: {llm_model_name}...")
try:
    llm = ChatGoogleGenerativeAI(
        model=llm_model_name,
        temperature=0
        # 如果 GOOGLE_API_KEY 环境变量已确认设置，则无需在此显式传递 google_api_key
    )
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Gemini LLM ({llm_model_name}) 初始化成功。")
except Exception as e:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 初始化 Gemini LLM 失败: {e}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 请确保 GOOGLE_API_KEY 环境变量已正确设置并指向有效的 Gemini API Key，")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 并且模型名称 '{llm_model_name}' 是你账户下可用的。")
    import traceback
    traceback.print_exc()
    exit()

# 6. 创建 RetrievalQA 链 (Chain)
print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 步骤 6: 创建 RetrievalQA 链...")
try:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" 是最简单直接的类型
        retriever=retriever,
        return_source_documents=True,
        # verbose=True # 可以尝试开启，看是否有更多链的内部日志
    )
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] RetrievalQA 链创建成功。")
except Exception as e:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 创建 RetrievalQA 链失败: {e}")
    exit()

# 7. 进行提问 (Query)
print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 步骤 7: 开始提问！输入 'exit' 退出程序。")
while True:
    query = input(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 请输入你的问题: ")
    if query.lower() == 'exit':
        break
    if not query.strip():
        continue

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 用户查询: '{query}'")

    # 在调用链之前，我们可以尝试手动检索一下，看看检索器返回什么
    try:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 正在使用检索器获取相关文档块...")
        retrieved_docs = retriever.invoke(query) # 使用 invoke
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 检索器返回了 {len(retrieved_docs)} 个文档块。")
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs):
                print(f"  --- 检索片段 {i+1} (来自: {doc.metadata.get('source', '未知来源')}) ---")
                print(f"  内容: {doc.page_content[:200]}...") # 打印前200字符
                print(f"  --------------------------------------")
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 检索器没有为查询 '{query}' 返回任何文档块。")
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 调用检索器失败: {e}")
        continue # 继续下一次提问

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 正在调用 QA 链进行思考 (qa_chain.invoke)...")
    chain_call_start_time = time.time()
    try:
        # 使用 invoke 替代旧的 __call__
        result = qa_chain.invoke({"query": query})
        chain_call_end_time = time.time()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] QA 链调用完成，耗时: {chain_call_end_time - chain_call_start_time:.2f} 秒。")

        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 答案:")
        print(result["result"])

        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 参考的源文档片段:")
        if result.get("source_documents"):
            for i, source_doc in enumerate(result["source_documents"]):
                print(f"  --- 片段 {i+1} (来自: {source_doc.metadata.get('source', '未知来源')}) ---")
                print(f"  {source_doc.page_content}")
                print(f"  --------------------------------------------------")
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 未找到源文档信息。")

    except Exception as e:
        chain_call_end_time = time.time()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] QA 链调用失败，耗时: {chain_call_end_time - chain_call_start_time:.2f} 秒。错误: {e}")
        import traceback
        traceback.print_exc() # 打印详细的错误堆栈信息
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 请检查错误信息，特别是与 API 调用、模型限制或 Prompt 相关的内容。")


print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 程序结束。")

# 清理 (可选)
# vectordb = None
# import shutil
# if os.path.exists(persist_directory):
#     shutil.rmtree(persist_directory)
# print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 已清理数据库目录 '{persist_directory}'")