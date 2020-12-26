1. 加载并结构化数据,数据结构如下:

- examples(list)
  - qid(list)
  - question(str)
  - contexts(list)
    - split_para(list of str max_num=40)
    - length(int invalid)
    - str(str)
  - labels(list of int 0 or 1)
  - answer(str)

2. 构建 sentence 字典

- sentence_dict(Dictionary)
  - ind2sen(dict)
  - sen2ind(dict)


3. 加载 inferSent 模型

```python
# reflaction ?
# with class BLSTMEncoder((enc_lstm):LSTM(300,2048,bidirectional=True))
# in @/models.py 
# in what way to find it?
model_embed = torch.load('infersent.allnli.pickle')
```


1. batch 切分,以及句向量生成

- dicts(list of Dictionary, batch_size 512)
  - 

```pyhon
model_embed.set_glove_path()
model_embed.build_vocab()

with torch.no_grad():
```