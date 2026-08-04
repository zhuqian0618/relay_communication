# ±60°圆弧神经网络辅助CE使用说明

## 组成

- `relay_sim/Neural_CE.py`：轨迹状态、20/32/44次预算CE、完整CE、0°校准和纯NumPy备用先验。
- `relay_sim/Torch_Neural_Prior.py`：PyTorch 2.12.0 GRU、DeepSets、编码/变化/功率与不确定度输出头。
- `relay_sim/Arc_Experiment.py`：±60°圆弧功率测量仿真、域随机化和训练数据生成。
- `Main_neural_ce_arc_simulation.py`：端到端训练、轨迹测试、完整CE对照和绘图入口。

## 安装与仿真

安装PyTorch环境：

```powershell
pip install -r requirements-pytorch.txt
```

快速检查流程（使用快速解析标签，不作为正式实验结果）：

```powershell
python Main_neural_ce_arc_simulation.py --backend torch --quick-labels --epochs 10 --no-plot
```

正式生成完整CE标签、训练模型并保存：

```powershell
python Main_neural_ce_arc_simulation.py --backend torch --epochs 120 --model-out arc_prior.pt
```

在每个测试角度额外运行1800次完整CE对照：

```powershell
python Main_neural_ce_arc_simulation.py --backend torch --epochs 120 --full-baseline
```

## 接入频谱仪

实验侧只需实现一个同步函数：

```python
def measure(joint_code: np.ndarray) -> float:
    # 1. 将 joint_code[:16] 和 joint_code[16:] 下发给两块超表面
    # 2. 等待编码和频谱仪稳定
    # 3. 读取标量功率
    return power_dBm
```

然后将该函数传入：

```python
result = optimizer.optimize(
    trajectory_state,
    angle_cmd_deg=current_angle,
    delta_angle_cmd_deg=commanded_step,
    measure=measure,
)
```

`result.best_code` 一定来自实际测量过的候选，`result.measurement_count` 只能为20、32或44。若
`result.calibration_required` 为真，应在下一个安全静止窗口安排完整CE，而不是继续突破44次在线预算。

## 实验顺序

1. UAV2停在0°，调用完整CE并将结果保存为`CalibrationRecord(0.0, ...)`。
2. 首次到达+60°和−60°时分别保存完整CE边界记录，用于真实域训练。
3. 正常移动时只调用预算优化器。
4. 返回0°后调用`optimizer.check_zero_anchor(...)`重复测量起始编码；功率下降超过3 dB时重新完整校准。
5. 完整CE和预算CE的全部候选、功率、时间戳应持久化，供下一轮PyTorch微调使用。

## 自动化检查

```powershell
python -m unittest discover -s tests -v
```
