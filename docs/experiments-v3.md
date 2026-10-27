# ETH3D v3：选参、结果与退步

日期：2026-10-27。此报告对应 `results/eth3d_v3/`，使用公开训练数据的内部场景族划分，不是官方隐藏测试结果。历史 `results/eth3d_v2/` 未修改。

## 预定方案与数据边界

验证族 `delivery_area/electro/facade` 共 13 场景，留出族 `forest/playground/terrace/terrains` 共 14 场景，与 v2 完全相同。先在验证集运行全部候选，以**逐场景 Bad2-all 的均值**选择；测试表中的总体指标按各自有效像素数加权，二者不可混淆。并列时按数值较小参数顺序选择。

| 方法 | 验证候选 | 最终参数 |
|---|---|---|
| BM | num_disp 64/96/128 × block 9/15/21，共 9 组 | 64 / 21 |
| SGBM | num_disp 64/96/128 × block 3/5/7，共 9 组 | 64 / 7 |
| SGBM + 一致性 | 复用选定 SGBM，固定容差 1 px，无额外搜索 | 64 / 7 / tolerance 1 |
| SGBM + WLS | 冻结上述 SGBM，再搜索 λ 8000/20000 × σ 0.8/1.2/1.6，共 6 组 | 64 / 7 / λ 8000 / σ 0.8 |

这不是对 SGBM/WLS 所有参数的联合全局搜索。滤波参数范围参考 [OpenCV 官方 WLS 文档](https://docs.opencv.org/4.x/d9/d51/classcv_1_1ximgproc_1_1DisparityWLSFilter.html)，接线方式参考[官方视差后处理教程](https://docs.opencv.org/4.x/d3/d14/tutorial_ximgproc_disparity_filtering.html)。先计算未过滤的左视差，再创建 WLS，避免工厂函数修改匹配器设置而改变基线掩码；右视差由 contrib 的 `createRightMatcher` 计算。

`validation_candidates.json` 包含 24 组候选的逐场景精度、覆盖率和耗时。`selected_parameters.json` 在加载测试图像/真值前写出；测试结果没有用于改候选或挑选场景。源文件可由 `manifest.json` 的 SHA-256 核对，环境与时间协议同样记录在清单中。外部 `--data` 的压缩包按其父目录下 `raw/` 查找，不再绑定仓库工作目录；没有压缩包也可运行。

## 指标与无效值

仅使用双目可见 `mask0nocc==255`、有限且正的真值视差。预测必须有限且正；原始 OpenCV 定点值除以 16 后转为 px。覆盖率 = 有效预测真值像素数 / 有效真值像素数。EPE = 共同有效像素上的平均绝对误差。Bad N-valid = 共同有效点中误差严格大于 N px 的比例；Bad N-all =（误差严格大于 N px 的点 + 缺少预测的点）/ 有效真值点数，N 为 1/2/3。

WLS 可以传播视差，为了本实验不靠填洞改变可评估范围，输出必须同时满足原 SGBM 正视差掩码和过滤结果为正。WLS 的覆盖率仍轻微下降，原因是少量原有效点的过滤结果非正。没有用 GT 修补或剔除错误点，也没有对测试结果做后验阈值调整。

一致性修复：右 matcher 的最小值为 `-num_disp+1`，无效哨兵为 `-num_disp`。以前只判断“负数”，会在左视差 63、右哨兵 -64、容差 1 时误接受。v3 显式排除哨兵，同时保留 -63 和 -0.25 等有效负视差。v2 一致性 EPE 1.155 px / Bad2-all 25.04%，v3 为 1.134 px / 25.04%；四舍五入接近不代表问题不存在，回归测试直接复现错误。

补充派生深度指标时没有重新调匹配参数：仅重放已冻结的四种配置，并逐场景断言新视差 coverage/EPE/Bad1/2/3 与原报告完全相等。保留原匹配耗时，将新增深度分数合并到原报告；`manifest.json` 记录了这一过程和冻结参数文件哈希。常规运行新版 benchmark 会同时产生视差与深度指标。

## 14 留出场景的总体指标

| 方法 | Coverage | EPE px | Bad1-all | Bad2-all | Bad3-all | 匹配中位 ms |
|---|---:|---:|---:|---:|---:|---:|
| BM | 68.02% | 0.682 | 36.21% | 34.50% | 34.20% | 9.2 |
| SGBM | 88.76% | 1.768 | 23.29% | 18.49% | 17.33% | 22.6 |
| 修正 SGBM 一致性 | 78.98% | 1.134 | 28.90% | 25.04% | 24.26% | 49.4 |
| SGBM + WLS | 88.70% | 1.167 | 20.35% | 16.97% | 16.11% | 52.1 |

WLS 相对 SGBM：Bad2-all 降 1.52 个百分点（相对降 8.24%），EPE 相对降 34.0%，覆盖率降 0.062 个百分点，本次中位时间约 2.31 倍。EPE 的有效像素集合略有不同，所以不能把 EPE 降幅解释为同一批点全部改善。BM 有效点误差更低但覆盖率只有 68%，一致性拒绝更多点后 Bad2-all 反而更高。

## 派生深度：MAE 改善，RMSE 明显退步

使用同一标定，将 GT 和预测视差转换为 `Z=baseline×fx/(d+doffs)`，复用上述非遮挡、正 GT 掩码，再要求 GT 深度有限且正；预测还必须具有有效正视差和有限正深度。记录 `depth_gt_rejected_pixels`，防止标定分母无效时悄悄改变分母。MAE/RMSE 在共同有效深度点上计算，另报有效深度覆盖率；没有给缺失点虚构深度，也没有用任意有限深度补洞。聚合 RMSE 为按有效点数加权的平方误差均值再开根号，不是场景 RMSE 的简单平均。

| 方法 | 深度覆盖率 | MAE mm | RMSE mm |
|---|---:|---:|---:|
| BM | 68.02% | 359.6 | 1986.4 |
| SGBM | 88.76% | 683.0 | 3348.8 |
| 修正 SGBM 一致性 | 78.98% | 488.3 | 2577.6 |
| SGBM + WLS | 88.70% | 526.8 | 5855.6 |

**这是由真值视差换算的深度误差，不等同硬件实测精度。** WLS 的深度 MAE 改善，但 RMSE 变差约 74.9%，尤其 `playground_2l` 的 WLS 深度 RMSE 为 17,841.2 mm，`forest_1s` 为 8,083.0 mm。误差随 `1/(d+doffs)` 非线性变化，少量很小的视差可产生很远的点。评估没有设置最大深度裁剪；不能用视差 EPE 改善来代替深度可靠性的判断。各方法有效集合不同，所有深度指标也必须与覆盖率一起看。

## 逐场景变化：WLS 相对 SGBM

负的 Bad2 变化表示改善，单位为百分点；EPE 单位 px。7 个场景 Bad2 改善，7 个退步，不能从总体改善推断每幅图都受益。

| 场景 | SGBM EPE | WLS EPE | Bad2-all 变化 |
|---|---:|---:|---:|
| forest_1s | 0.316 | 0.705 | +3.756 |
| forest_2s | 0.308 | 0.351 | +0.888 |
| playground_1l | 0.350 | 0.439 | +0.658 |
| playground_1s | 0.247 | 0.326 | +0.184 |
| playground_2l | 3.864 | 1.660 | -2.353 |
| playground_2s | 0.300 | 0.277 | +0.158 |
| playground_3l | 0.588 | 0.620 | +1.597 |
| playground_3s | 0.355 | 0.523 | +1.783 |
| terrace_1s | 2.918 | 0.787 | -2.786 |
| terrace_2s | 0.436 | 0.327 | -0.350 |
| terrains_1l | 2.289 | 1.805 | -0.973 |
| terrains_1s | 1.615 | 0.959 | -4.600 |
| terrains_2l | 3.249 | 2.045 | -5.956 |
| terrains_2s | 4.330 | 3.020 | -5.199 |

推测：边缘引导的平滑在部分高误差区域有效，在原匹配已很准的细节区域可能引入偏差。当前证据是逐场景指标与预览，尚未做材质/距离/遮挡边缘的分区统计，不能把这个机制解释当作已验证结论。

## 本地验证记录与限制

实际运行命令（PowerShell，项目根目录；环境在仓库外）：

```powershell
$py='python'
$data='C:/datasets/eth3d'
$env:PYTHONPATH='src'
& $py -m pytest -q
& $py -m stereolab.benchmark --data $data --out results/eth3d_v3 --threads 1
& $py -m stereolab.demo --out results/synthetic_demo
& $py -m stereolab.reconstruct --left "$data/forest_1s/im0.png" --right "$data/forest_1s/im1.png" --calibration "$data/forest_1s/calib.txt" --out results/local_demo/forest_1s --method sgbm_wls --num-disp 64 --block-size 7 --ground-truth "$data/forest_1s/disp0GT.pfm" --mask "$data/forest_1s/mask0nocc.png" --max-points 20000
```

合成平面实测 900.00 mm；真实 `forest_1s` 导出 259,147 点，HTML 均匀抽样到 20,000 点，GT 掩码内覆盖率 92.75%、EPE 0.705 px、Bad2-all 12.54%。数值来自真实运行，图像、NPY、PLY 与带数据的 HTML 留在本机。检查了视差/深度 PNG 的黑色无效区和场景轮廓，没有把伪彩当作精度证明。

证据：[合成数值报告](../results/synthetic_demo/report.json)、[真实场景数值报告](../results/local_demo/forest_1s/report.json)。独立审查重新运行39项测试及14×4组匹配，并以独立公式累计深度误差，核对报告一致。

HTML 没有外部脚本、网络请求或图片依赖，测试执行了 Canvas 事件桩中的拖动/滚轮/方向键/复位及空点云路径；浏览器 `file://` 打开被当前工具安全策略拒绝，因此本次没有真实浏览器视觉验收。使用者可双击本地 HTML 验证显示；事件桩不是完整浏览器兼容性测试。

耗时固定 OpenCV 单线程，每个场景一次调用，没有预热/重复/置信区间，受本机并发工作与调度影响；对 CPU 和 IO 环境不可泛化。只测试 27 个低分辨率场景，未验证真实相机噪声、畸变校正误差、移动物体或测量级标定。HTML 为正交投影，均匀抽样可漏掉细小结构，离群点会挤压视图尺度，尚无选点测距和离群点交互剔除功能。
