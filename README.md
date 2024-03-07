# Self-evolution-learning-for-mixup
## Get Started
```bash
pip install -r requirements.txt
```
数据的准备
可在read_data/preprocess.py中更改读取数据集的路径
论文中所使用数据为从Huggingface中下载载入到本地，在本地加载数据集
```python
import datasets
data=datasets.load_dataset('glue','sst2')
data.save_to_disk('sst2')
#本地加载
data=datasets.load_from_disk('sst2')

```
## 使用
### 先使用模型跑出baseline
```bash
python run_train.py --dataset  --seed  --model  --mode 'normal' --low 1 --num 10
```
参数low设置为1表明是在小样本情况下（默认为0），num表示在每种标签的样本所抽取的数量
### 在baseline基础上使用mixup进行dataaugmentation,提升模型效果
```bash
python run_train.py  --dataset  --model  --seed  --mode ''   --low 1 --num 10 --evo 1
```
此时参数mode指定为各种mixup方法，evo参数为1表明进行self-evolution learning

  
