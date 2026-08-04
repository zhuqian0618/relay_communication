# 初学者版：MLP神经网络辅助CE编码优化

## 1. 先认识当前网络

当前网络是**多层感知机（MLP）**，也叫全连接深度神经网络（DNN）。

它不是卷积神经网络CNN。CNN主要用于图像或具有明显局部邻域结构的数据。它也不是循环神经网络或GRU，因为本版本不读取历史位置序列。

整个流程只有四步：

```text
控制角度 → MLP预测编码概率 → 小规模CE搜索 → 频谱仪实测最佳编码
```

网络只是给CE一个较好的搜索起点。最终编码必须来自频谱仪真正测量过的候选。

## 2. 输入、标签和输出

### 输入

原始角度首先转换为24组正弦和余弦：

```python
[sin(angle), cos(angle), sin(2*angle), cos(2*angle), ..., sin(24*angle), cos(24*angle)]
```

因此单个样本的输入维度是48。它们称为**傅里叶角度特征**，仍然全部由同一个角度确定，并没有增加新的传感器输入。

只使用第一组`sin(angle)、cos(angle)`时，输入变化过于平滑，难以表示2-bit编码在量化边界处的频繁跳转。加入高阶正弦和余弦后，普通MLP也能学习这些快速变化。

### 标签

两块超表面各有16个编码变量，组合成32个标签。每个变量有4个状态：

```text
状态0 = 0°
状态1 = 90°
状态2 = 180°
状态3 = 270°
```

### 输出

网络为32个变量分别输出4个logits，张量形状为：

```text
[batch_size, 32, 4]
```

Softmax将每组4个logits转换为4种状态的概率。

## 3. 网络结构

```text
输入48维傅里叶角度特征
  ↓
Linear(48, 128)
  ↓
ReLU
  ↓
Linear(128, 256)
  ↓
ReLU
  ↓
Linear(256, 128)
  ↓
reshape为 [batch_size, 32, 4]
```

`Linear`实现 `y = Wx + b`。ReLU为网络加入非线性，否则多个Linear仍等价于一个Linear。

## 4. 仿真数据如何产生

默认训练角度为：

```text
−60° 到 +60°，每0.5°一个样本，共241个样本
```

验证角度相对训练网格偏移0.25°，用于检查网络能否预测没有直接见过的中间角度。

可以把训练间隔改成0.25°，得到481个训练样本。但相邻编码通常重复，所以更密的数据并不等于同等比例增加的新信息。

发射功率和噪声功率不作为第一版网络输入，因为在线性信道模型中它们通常不会改变理论最优相位编码。它们只用于比较高、中、低SNR下CE搜索的稳定性。

## 5. 模型如何训练

安装依赖：

```powershell
pip install -r requirements-pytorch.txt
```

当前已验证环境为Python 3.11.15和PyTorch 2.5.1。

运行仿真预训练：

```powershell
python Main_simple_neural_ce.py --mode train
```

每个batch执行：

```python
optimizer.zero_grad()       # 清空上一轮梯度
logits = model(features)    # 前向传播
loss = criterion(logits, labels)
loss.backward()             # 反向传播，计算梯度
optimizer.step()            # Adam更新参数
```

训练会生成：

- `simple_code_net.pt`：训练后的模型；
- `simple_training_history.png`：训练/验证损失和准确率；
- `simple_code_heatmap.png`：真实编码与预测编码热力图。

如果训练损失和验证损失一起下降，说明模型正常学习。若训练损失继续下降而验证损失上升，则可能过拟合。程序会记录验证损失最低的epoch，并在训练结束后自动恢复该时刻的网络参数；图中的灰色虚线表示这个最佳epoch。

默认最多训练300个epoch；如果验证损失连续40轮没有改善，则提前结束，避免继续过拟合。

## 6. 如何加入实测数据

实测CSV格式必须为：

```text
angle_deg,c0,c1,...,c31
```

每行保存一个角度的完整CE最优联合编码。变量`c0`和`c16`固定为0。

微调命令：

```powershell
python Main_simple_neural_ce.py --mode train --real-data measured_codes.csv
```

程序会先使用仿真数据预训练，再使用较小学习率对实测数据微调，并生成`simple_finetune_history.png`。

## 7. 冷启动和热启动是什么

### 冷启动CE

没有网络帮助，4种状态初始概率都是0.25：

```text
[0.25, 0.25, 0.25, 0.25]
```

### 热启动CE

使用网络和上一位置编码：

```text
初始概率 = 70%网络概率 + 20%上一编码 + 10%均匀探索
```

两种方法都固定运行3代，每代实测12组，共36次频谱仪读数。每代选择功率最高的3组更新概率。

运行冷热启动对比：

```powershell
python Main_simple_neural_ce.py --mode demo --model simple_code_net.pt
```

程序分别在高SNR、默认SNR和低SNR下比较两种方法。

## 8. 如何接入真实频谱仪

只需要替换测量函数：

```python
def measure(joint_code):
    # joint_code[:16] 下发给MS1
    # joint_code[16:] 下发给MS2
    # 等待硬件和频谱仪稳定
    return measured_power_dBm
```

调用方式：

```python
result = warm_start_ce(
    model=model,
    angle_deg=current_commanded_angle,
    previous_code=last_best_code,
    measure=measure,
)

new_code = result.best_code
new_power_dBm = result.best_power_dBm
```

## 9. 两种“交叉熵”不要混淆

- `CrossEntropyLoss`：训练神经网络分类器的损失函数。
- CE优化算法：采样编码、测量功率、选择精英并更新概率的黑盒搜索算法。

它们使用了相同的中文名称，但在代码中承担完全不同的任务。

## 10. 推荐学习顺序

1. NumPy数组、形状、索引和`reshape`；
2. 特征、标签、训练集和验证集；
3. 单个神经元和 `y = Wx + b`；
4. 全连接层与MLP；
5. ReLU激活函数；
6. logits、Softmax和四分类；
7. CrossEntropyLoss；
8. 前向传播、反向传播和梯度；
9. epoch、batch size和learning rate；
10. PyTorch的`nn.Module`、DataLoader、Adam、保存和加载；
11. CE算法的采样、精英选择和概率更新；
12. 冷启动与热启动。

CNN、GRU、DeepSets、模型集成和不确定度可以等基础版本稳定后再学习。

## 11. 自动化检查

```powershell
python -m unittest discover -s tests -v
```
