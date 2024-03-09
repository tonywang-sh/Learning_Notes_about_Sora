## Transformer LLM学习笔记
####1. Attention

Attention是


####2.Tokenizer分词


####3. Transformer 案例实践
文本分类实验，二分类文本，近5K文本样本

模型case1: transformer结构， Embedding + 单层SelfAttention + MLP, 
![img.png](img-case1-0.png)
![img.png](img-case1-1.png)
![img.png](img-case-1-2.png)
模型case2: transformer结构, Embedding + 单层SelfAttention + LayerNorm + MLP
![img.png](img-case2-0.png)
![img.png](img-case2-1.png)
![img.png](img-case2-2.png)

####4. 结合Numpy实现的llama实例
下载llama numpy实现
![img.png](img-3.png)
LLaMA模型结构

 1) Tokenizer
 2) Decoder
 3) Transformer Block
![img.png](img-llm-arch.png)