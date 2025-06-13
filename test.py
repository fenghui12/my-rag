from transformers import BertTokenizer, BertModel
import torch

# 1. 加载模型和分词器 (Tokenizer)
# 分词器就像一个翻译官的助手，先把我们的话切成一个个词或者小片段，模型才好理解
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# 模型本体，我们的“Transformer老师”
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True) # 注意这里，我们要让它输出注意力！

print("模型和分词器加载好啦！")
# 2. 准备一句话
sentence = "I love playing games with my friends."
print(f"我们要分析的句子是: {sentence}")

# 3. 用分词器处理这句话
# 模型不直接认识英文单词，要先变成它能懂的数字ID
inputs = tokenizer(sentence, return_tensors='pt') # 'pt' 表示返回 PyTorch 张量（一种特殊的数字数组）
print(f"分词后的结果 (模型能看懂的数字): {inputs}")

# 看看分词器把我们的句子切成了哪些词 (tokens)
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
print(f"分词后的词元 (tokens): {tokens}")
# 4. 把处理好的句子喂给模型
# model.eval() 会让模型进入“评估模式”，关闭一些训练时才用的功能，这里可以不加，但好习惯
model.eval()
with torch.no_grad(): # 告诉 PyTorch 我们只是看看，不需要计算梯度（学习的时候才用）
    outputs = model(**inputs) # **inputs 把字典里的东西自动对应到模型的参数

# 5. 从模型的输出中拿到“注意力小本本”
# 'outputs.attentions' 是一个元组 (tuple)，里面有很多层的注意力
# BERT-base 有 12 层 Transformer block，所以会有 12 个注意力张量
attentions = outputs.attentions
print(f"模型输出了 {len(attentions)} 层的注意力权重。")

# 我们先看看第一层的注意力 (索引是0)，第一个注意力头 (索引也是0)
# 注意力张量的形状通常是: (batch_size, num_heads, sequence_length, sequence_length)
# batch_size: 我们一次处理几个句子 (这里是1)
# num_heads: 多头注意力的“头”的数量 (bert-base有12个头)
# sequence_length: 句子的长度 (分词后的词元数量)
first_layer_attentions = attentions[0] # 取出第一层的所有头的注意力
one_head_attention = attentions[1][0][1] # 从第一层中取出第一个头的注意力 (形状是 seq_len x seq_len)

print(f"我们选取的单个注意力头的权重形状: {one_head_attention.shape}")
print("这个注意力权重告诉我们，每个词对其他词的关注程度。")
import matplotlib.pyplot as plt
import numpy as np # numpy 是一个处理数字数组的强大工具

# 6. 可视化注意力权重 (简单版)
# 我们需要把 PyTorch 张量转成 NumPy 数组才能用 matplotlib 画图
attention_scores_np = one_head_attention.detach().cpu().numpy() # .detach() .cpu() .numpy() 是标准转换流程

# 创建一个图
fig, ax = plt.subplots(figsize=(8, 6)) # figsize可以调整图片大小
im = ax.imshow(attention_scores_np, cmap='viridis') # cmap是颜色方案，可以换别的

# 设置坐标轴的标签，显示我们分词后的词元
ax.set_xticks(np.arange(len(tokens)))
ax.set_yticks(np.arange(len(tokens)))
ax.set_xticklabels(tokens)
ax.set_yticklabels(tokens)

# 让x轴的标签旋转一下，避免重叠
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# 给图加上标题
ax.set_title("Attention Head 0 - Layer 0")
fig.tight_layout() # 自动调整布局，让所有东西都显示出来
plt.colorbar(im) # 显示颜色条，表示关注强度
plt.show() # 把图显示出来！

print("请看弹出的图片，颜色越亮的地方，表示行对应的词对列对应的词关注度越高！")
print("比如，看看 'friends' 这一行，它可能对 'my' 和 'games' 的关注度比较高。")
print("你可以尝试不同的句子，或者看看模型不同层、不同头的注意力，会有很多有趣的发现！")