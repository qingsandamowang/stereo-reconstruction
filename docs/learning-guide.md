# 学习说明：从视差到离线点云

## 先跑一个能手算的场景

按 README 安装 contrib 环境，设置 `PYTHONPATH=src` 后执行：

```powershell
python -m stereolab.demo --out outputs/synthetic
```

打开 `viewer.html`，拖动旋转，滚轮缩放，再点 Reset。输入是固定种子的随机纹理，右图相对左图平移 12 像素。标定焦距 180 px，基线 60 mm，所以预期 `Z=60×180/12=900 mm`。报告中位深度应约 900 mm。左侧匹配搜索边界仍然无效，因此即使是理想平面，覆盖率也不会达到 100%。

用下面的代码读取数值，检查 NaN，不要从 PNG 颜色猜深度：

```python
import numpy as np
d = np.load("outputs/synthetic/disparity.npy")
z = np.load("outputs/synthetic/depth_mm.npy")
print(np.nanmedian(d), np.nanmedian(z), np.isfinite(d).mean())
```

## 图像、标定与几何

左图横坐标减右图横坐标得到视差 `d`。同一点经过水平校正后应在同一图像行，匹配主要沿 x 轴搜索。`io.py` 读取彩色/灰度图片、PFM 和 Middlebury 标定；PFM 要处理大小端和自下而上的行顺序。

项目几何为 `Z=baseline×fx/(d+doffs)`，`X=(x-cx)Z/fx`，`Y=(y-cy)Z/fy`，单位 mm。`doffs` 是右、左主点横坐标之差；fx/fy 可以不同，主点可能位于裁切图片之外。基线若用米填写却当毫米读取，尺度会错 1000 倍；程序只能验证有限/正数，无法替你核实实物单位。非正分母和非有限深度不会变成点。

动手练习：把 `fx=100,fy=200,cx=-200,cy=500,baseline=20,doffs=5`，像素 `(0,0)` 的视差设为 5，手算应得 `(X,Y,Z)=(400,-500,200) mm`。对照 `tests/test_reconstruct.py` 的对应例子。

自己的照片必须先正确校正；这里不从任意两张照片恢复内外参，也不自动校正。只支持当前正视差搜索。图像缩放后焦距和主点也要按同一比例调整；水平裁切会影响主点及 `doffs`。

## 四种方法各解决什么

BM 用局部窗口比较相似度；SGBM 沿多方向累计代价，通常能输出更多点。视差范围越大，搜索代价和左侧无效边界也越大。窗口越大，匹配更稳定但细节更容易被平滑。

左右一致性：左点先通过 `xR=round(xL-dL)` 找到右点，再检查 `abs(dL+dR)<=1`。OpenCV 输出是 16 倍定点数；右 matcher 最小视差 -63 时，-64 是“没有匹配”的哨兵，不能因为它是负数就接受。回归测试用 `dL=63,dR=-64` 展示旧错误；修复还要保留有效 -63 与 -0.25。过滤通常降低覆盖率，EPE 变小可能只因拒绝了难点。

WLS 用左图边缘和左右匹配置信度引导视差平滑。它来自 OpenCV contrib，不是自己训练的模型。`lambda` 控制正则化强度，`sigmaColor` 控制对图像边缘的敏感程度。实验保留原 SGBM 的无效掩码，不能通过把洞补成任意值获得看似更好的覆盖率。

## 学会阅读报告，尤其是退步

`coverage` 只统计有效 GT 掩码内有预测的比例；`epe_valid_px` 只看共同有效点。Bad1/2/3-all 把未预测的点也视为错误，误差恰好等于阈值不计坏点。所有指标必须一起读。

v3 中 SGBM 覆盖率 88.76%，EPE 1.768 px，Bad2-all 18.49%；WLS 覆盖率 88.70%，EPE 1.167 px，Bad2-all 16.97%，但本次中位时间约 2.31 倍。WLS 的 14 个留出场景中，7 个 Bad2 改善、7 个退步；`forest_1s` EPE 从 0.316 升到 0.705。先读[逐场景表](experiments-v3.md)，再讨论为什么总体改善与局部退步可以同时成立。

还要看派生深度：SGBM 深度 MAE/RMSE 为 683.0/3348.8 mm，WLS 为 526.8/5855.6 mm，平均绝对误差下降但平方误差变差。自己计算 `d=1` 与 `d=0.5` 的深度，可理解远处小视差为什么放大误差。深度评估复用 GT 掩码，并报告有限正深度覆盖率，未裁剪远点。**由真值视差换算的深度误差，不等同硬件实测精度。**

选参使用 13 个验证场景，整个场景族与 14 个留出场景隔离。BM/SGBM 各搜索 9 组，WLS 在已选 SGBM 上搜索 6 组，完整候选在 JSON。`benchmark.py` 在加载留出图片前写出参数检查点。练习：自己根据 `bad2_scene_mean` 复算选中候选，确认它不是按测试 EPE 挑出来的。

## 将真实数据换成自己的输入

```powershell
python -m stereolab.reconstruct --left LEFT.png --right RIGHT.png --calibration calib.txt --out outputs/my_pair --method sgbm_wls --num-disp 64 --max-points 10000
```

无 GT 时只查看结构、输出比例、深度分布和孔洞；`report.json` 不会凭空出现 accuracy。可选择 `--stride 2` 减少 PLY 点数、`--max-depth-mm 20000` 裁剪远处离群点。HTML 的 `--max-points` 只影响页面抽样，不能改变全分辨率误差结果；默认不进行点云远平面裁剪。

若想复现 ETH3D，将自己的下载数据放在仓库外，按 README 传 `--data`。不要提交原始图、衍生图片、点云或嵌有数据的 HTML。只有数值报告可安全随源码保存。旧 v2 结果用于历史对照，不用于覆盖新修复的证据。

## 建议的代码阅读顺序

1. `io.py` 与 `tests/test_io.py`：标定、PFM、非遮挡掩码的意义。
2. `matching.py` 与 `tests/test_matching.py`：定点缩放、无效哨兵、左右检查及 WLS 接线。
3. `metrics.py` 与 `benchmark.py`：缺失预测的惩罚、验证选参、像素加权与逐场景均值。
4. `geometry.py`：毫米深度与 XYZ；用几个小数组手工核对坐标。
5. `reconstruct.py` / `viewer.py`：CLI 编排与无外部依赖的 Canvas。Node 事件桩测交互代码路径，真正的视觉效果仍需浏览器查看。

面试时可以说清：公开 ETH3D 数据、传统 OpenCV 方法、Codex 的协助、真实的验证/留出划分、总体改进与场景退步、无效值漏洞及回归测试。不应称为新模型、生产级测量系统或已完成的双目结构光硬件。
