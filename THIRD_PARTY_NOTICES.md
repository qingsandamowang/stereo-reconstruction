# 来源与归属

- 双目图片、真值及相机参数来自 [ETH3D](https://www.eth3d.net/datasets)，格式见[官方文档](https://www.eth3d.net/documentation)。下载及使用数据请遵守来源许可和引用要求。
- 标定参数与深度换算采用 [Middlebury version 3 格式](https://vision.middlebury.edu/stereo/data/scenes2014/)。
- BM/SGBM 与 contrib WLS 使用 OpenCV 提供的实现（[官方 WLS 教程](https://docs.opencv.org/4.x/d3/d14/tutorial_ximgproc_disparity_filtering.html)）；本项目负责参数选择、评估、无效值处理、格式解析、点云输出及离线查看流程。
- 代码由 Codex Agent 协助实现。公开包不再分发原始图片、真值、派生图片或点云；保存数值报告用于核对实验。
