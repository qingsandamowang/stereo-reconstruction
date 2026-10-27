# 双目视差估计与三维重建

**Stereo Reconstruction · BM / SGBM / left-right consistency**

基于 ETH3D 真实双目数据，完成相机标定解析、BM/SGBM 参数选择、视差评估、左右一致性检查、深度换算与彩色 PLY 点云导出。代码由 Codex Agent 协助开发和核验。

## 流程

左右校正图像 → 双目匹配 → 视差图 → 标定参数换算深度 → 三维彩色点云

项目使用 OpenCV 中的传统匹配算法，没有训练新的深度学习模型。数据和标定来自公开数据集；没有自行采集数据或实现结构光投影硬件。

## 已完成实验

在 ETH3D 公开低分辨率双目训练集的 27 组场景内自行划分：

- 参数选择：`delivery_area` / `electro` / `facade` 共 13 组。
- 留出测试：`forest` / `playground` / `terrace` / `terrains` 共 14 组。
- 同一场景族不跨两组；本结果不是 ETH3D 官方隐藏测试榜成绩。

| 方法 | 覆盖率 ↑ | 有效预测点 EPE ↓ | Bad 2px，含未预测点 ↓ |
|---|---:|---:|---:|
| BM | 68.0% | 0.682 px | 34.5% |
| SGBM | **88.8%** | 1.768 px | **18.5%** |
| SGBM + 左右一致性 | 79.0% | 1.155 px | 25.0% |

只评估真值有限且为正、`mask0nocc == 255` 的双目可见像素。Bad 2px 将误差超过 2 px 或没有有效预测的位置计为错误；EPE 只在有效预测点上计算。因此三项指标必须结合阅读：SGBM 覆盖更多点，但有效点 EPE 高于 BM。

证据：[测试报告](results/eth3d_v2/test_report.json)、[参数候选表](results/eth3d_v2/validation_candidates.json)、[数据哈希与场景名单](results/eth3d_v2/manifest.json)、[已导出点数](results/eth3d_v2/point_counts.json)。

## 安装与复现

使用 Python 3.11 或更新版本，从仓库根目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

从 [ETH3D 官方页面](https://www.eth3d.net/datasets)下载 `two_view_training.7z`、`two_view_training_gt.7z`，放到 `data/raw/`；将两包解压到 `data/eth3d/`，使每个场景目录内有 `im0.png`、`im1.png`、`calib.txt`、`disp0GT.pfm`、`mask0nocc.png`。

```powershell
$env:PYTHONPATH='src'
python -m stereolab.benchmark --out results/local_run
```

程序会选参数并评估 14 组场景，生成数值报告、三张对比图和三份彩色点云。仓库保留数值报告；原始图像、派生结果图和点云由用户下载数据后自行生成。

## 几何与局限

使用 Middlebury 标定格式：`Z = baseline * f / (d + doffs)`，本项目深度和三维坐标以毫米为单位。弱纹理、遮挡和重复纹理仍可能造成孔洞或离群点；公开数据上的结果不能直接当作实际测量精度。

- [中文学习说明](docs/learning-guide.md)。
- [评估入口](src/stereolab/benchmark.py)。
- [来源与归属](THIRD_PARTY_NOTICES.md)。
