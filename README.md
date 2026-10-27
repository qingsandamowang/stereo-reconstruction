# 双目视差估计与三维重建

**Stereo Reconstruction · BM / SGBM / corrected left-right consistency / SGBM + WLS**

从已校正双目图片和标定文件生成视差、毫米深度、彩色 PLY 和可离线旋转/缩放的 HTML 点云。提供无需下载数据的合成演示，以及 ETH3D 27 场景的验证选参和留出评测。使用 OpenCV 传统算法；代码由 Codex Agent 协助开发和核验。

## 五分钟运行

Python 3.11+，从仓库根目录运行。WLS 需要 contrib 包，请使用新环境，避免同时安装多个提供 `cv2` 的包：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH='src'
python -m pytest -q
python -m stereolab.demo --out outputs/synthetic
```

双击 `outputs/synthetic/viewer.html` 即可在本地浏览器查看，无需服务器、联网或 CDN。拖拽旋转，滚轮/滑块缩放，方向键旋转，R 或按钮复位。默认嵌入最多 20,000 点；页面显示实际点数、总点数和毫米单位。点云使用左相机坐标：X 向右、Y 向下、Z 向前。

合成演示用固定种子生成纹理平面，水平平移 12 px，`f=180 px`、`baseline=60 mm`，预期深度 `900 mm`，本机实测中位深度 `900 mm`。这验证数据与几何流程，不能代表真实场景精度。[数值报告](results/synthetic_demo/report.json)

## 使用自己的校正双目图片

输入必须已完成畸变校正和水平极线校正，当前匹配搜索正视差。程序不执行相机标定或自动校正。彩色图片用于点云着色，灰度图片用于匹配。

```powershell
python -m stereolab.reconstruct --left LEFT.png --right RIGHT.png --calibration calib.txt --out outputs/my_pair --method sgbm_wls --num-disp 64 --block-size 7
```

`calib.txt` 使用 ETH3D/Middlebury 格式，基线必须以毫米填写，例如：

```text
cam0=[500 0 320; 0 510 240; 0 0 1]
cam1=[500 0 325; 0 510 240; 0 0 1]
doffs=5
baseline=60
width=640
height=480
ndisp=128
```

支持不同的 `fx/fy`、图像范围外的主点和非零 `doffs`；不要裁切/缩放图片后继续沿用原标定。`cam1` 可省略，提供时检查与水平校正模型、主点差一致。焦距/基线必须正且有限，尺寸必须与图片一致。`ndisp` 不要求为 16 的倍数；默认向上取整并限制在图像可搜索范围内。大搜索范围会使左侧无效区域变宽，示例显式使用验证选出的 `64`；自己的数据应按实际视差范围设定。

输出文件：

| 文件 | 含义 |
|---|---|
| `disparity.npy` / `depth_mm.npy` | float32 数值，视差 px / 深度 mm；无效值为 NaN |
| `disparity.png` / `depth.png` | 独立按第 99 百分位缩放的伪彩预览，黑色为无效；不能直接读取物理数值 |
| `cloud.ply` | 彩色点云，坐标 mm；`--stride` 控制采样间隔 |
| `viewer.html` | 单文件离线 Canvas 查看器；`--max-points` 控制嵌入点数，不改变 PLY |
| `report.json` | 参数、输入哈希、点数、时间、单位和精度评估状态 |

`--method bm/sgbm/sgbm_wls` 选择算法；`--consistency` 可加左右一致性。WLS 默认 `--wls-lambda 8000 --wls-sigma 0.8` 来自本项目验证集，并不保证适合所有数据。默认不裁剪远点；可用 `--max-depth-mm 20000` 只裁剪点云，NPY 深度保留原值。

没有真值时报告不生成 `accuracy`。如有像素单位 PFM 真值，可加 `--ground-truth GT.pfm`，可选 `--mask mask0nocc.png`（255 才计入评估）。`prediction_fraction_of_image` 是整图输出比例，与真值掩码内的 `accuracy.coverage` 含义不同。

提供真值时还报告派生深度 MAE/RMSE 和有效深度覆盖率：两张视差按同一标定换算成深度，使用相同非遮挡、有限正 GT 掩码，并排除非有限/非正深度。这是**由真值视差换算的深度误差，不等同硬件实测精度**。

## ETH3D v3 实测结果

27 个公开低分辨率训练场景按场景族划分：`delivery_area/electro/facade` 共 13 个验证场景，`forest/playground/terrace/terrains` 共 14 个留出场景。不是 ETH3D 官方隐藏测试榜。BM/SGBM 各 9 组候选；先冻结验证选出的 SGBM，再验证 WLS 的 6 组 λ/σ，全部候选逐场景记录。保存选参检查点后才加载留出图片与真值。

| 方法 | 覆盖率 ↑ | 有效点 EPE ↓ | Bad1-all ↓ | Bad2-all ↓ | Bad3-all ↓ | 匹配中位时间 |
|---|---:|---:|---:|---:|---:|---:|
| BM | 68.02% | **0.682 px** | 36.21% | 34.50% | 34.20% | **9.2 ms** |
| SGBM | **88.76%** | 1.768 px | 23.29% | 18.49% | 17.33% | 22.6 ms |
| SGBM + 修正左右一致性 | 78.98% | 1.134 px | 28.90% | 25.04% | 24.26% | 49.4 ms |
| SGBM + WLS | 88.70% | 1.167 px | **20.35%** | **16.97%** | **16.11%** | 52.1 ms |

仅评估 `mask0nocc==255` 且真值有限、正的像素；Bad N-all 把误差严格大于 N px **或缺少预测** 计为错误。EPE 只在有效预测点计算，需要结合覆盖率阅读。WLS 保留原 SGBM 的无效掩码，不借助填洞提升覆盖率。

WLS 总体 Bad2-all 比 SGBM 降低 **1.52 个百分点**，有效点 EPE 降低 **34.0%**，代价是约 **2.31 倍**的本次中位耗时；14 场景中 7 个 Bad2 改善、7 个退步。`forest_1s` Bad2 反而增加 3.76 个百分点，EPE 从 0.316 升到 0.705 px。BM 的有效点 EPE 仍最低，不能只把 WLS 称为“全面更好”。详见[逐场景比较、回归与实验限制](docs/experiments-v3.md)。

| 方法 | 有效深度覆盖率 ↑ | 派生深度 MAE ↓ | 派生深度 RMSE ↓ |
|---|---:|---:|---:|
| BM | 68.02% | **359.6 mm** | **1986.4 mm** |
| SGBM | **88.76%** | 683.0 mm | 3348.8 mm |
| SGBM + 修正左右一致性 | 78.98% | 488.3 mm | 2577.6 mm |
| SGBM + WLS | 88.70% | 526.8 mm | 5855.6 mm |

WLS 的深度 MAE 下降，但 **RMSE 从 3348.8 升到 5855.6 mm**。深度对小视差误差很敏感，远处离群点会放大平方误差；本报告没有裁剪远点来美化分数。所有深度分数只在各自有效预测点计算，需同时读覆盖率，不能据此称为测量级系统。

v3 修复了右视图 `minDisparity=-63` 时把无效哨兵 `-64` 当成有效负视差的错误；测试覆盖左 63 / 右无效 -64，以及真实负视差 -63 / -0.25。旧 [eth3d_v2](results/eth3d_v2/test_report.json) 完整保留，其中一致性结果属于历史实现。新证据：[v3 测试报告](results/eth3d_v3/test_report.json)、[全部验证候选](results/eth3d_v3/validation_candidates.json)、[冻结参数](results/eth3d_v3/selected_parameters.json)、[来源/哈希/环境](results/eth3d_v3/manifest.json)。

## 复现实验与真实场景演示

从 [ETH3D 官方数据页面](https://www.eth3d.net/datasets) 下载 `two_view_training.7z` 和 `two_view_training_gt.7z`。解压到 `DATA/eth3d/`，每个场景中有 `im0.png`、`im1.png`、`calib.txt`、`disp0GT.pfm`、`mask0nocc.png`；压缩包可放在同级 `DATA/raw/`。支持仓库外数据，不需要复制到 Git：

```powershell
python -m stereolab.benchmark --data C:/datasets/eth3d --out results/local_run --threads 1
# 如需生成三组本地对比图/PLY，可追加 --visuals
python -m stereolab.reconstruct --left C:/datasets/eth3d/forest_1s/im0.png --right C:/datasets/eth3d/forest_1s/im1.png --calibration C:/datasets/eth3d/forest_1s/calib.txt --out outputs/forest_1s --method sgbm_wls --num-disp 64 --block-size 7 --ground-truth C:/datasets/eth3d/forest_1s/disp0GT.pfm --mask C:/datasets/eth3d/forest_1s/mask0nocc.png
```

原始图像、派生图片、NPY、PLY、嵌入点云的 HTML 均由 `.gitignore` 排除；只提交数值报告。只有解压目录、没有压缩包的源码 ZIP 使用者也能运行：清单会把压缩包哈希标为不可用，并记录场景文件哈希。

实测环境 Python 3.13.9 / OpenCV contrib 5.0.0 / NumPy 2.5.4，固定 OpenCV 单线程；[精确依赖](requirements-repro.txt)。时间只测一次每场景，包含匹配器创建、左右匹配和过滤，不含 IO/导出；不能当作充分预热或生产性能测试。

## 几何、学习与限制

`Z = baseline × fx / (d + doffs)`，`X = (x-cx)Z/fx`，`Y = (y-cy)Z/fy`。非有限视差、非正分母和无效深度不输出为点。弱纹理、遮挡、反光和重复纹理仍会造成孔洞与离群点；标定误差会改变尺度。没有训练新模型、采集自有数据或实现结构光投影硬件。

39 项测试覆盖匹配哨兵、WLS、视差/派生深度指标、选参、PFM、标定、几何、合成 CLI 和查看器事件。可选 Node 的 Canvas 事件桩验证旋转/滚轮/键盘/复位和空点云，不能替代真实浏览器渲染检查。本次环境的浏览器策略禁止打开 `file://`，因此未声称完成浏览器视觉验收；已检查真实输出的视差/深度 PNG。

- [中文学习与逐步练习](docs/learning-guide.md)
- [v3 实验说明](docs/experiments-v3.md)
- [来源与归属](THIRD_PARTY_NOTICES.md)
