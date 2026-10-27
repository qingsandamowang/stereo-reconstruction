# Portfolio Upgrade Implementation Plan

> **For agentic workers:** 按任务使用 executing-plans 或 subagent-driven-development 执行；每个独立项目都有单独计划，必须先完成安全测试再集成。

**Goal:** 完成修复后的多算法评测、自定义输入和离线点云查看。
**Architecture:** matching.py管理视差算法与无效值，benchmark.py管理验证选择，新增reconstruct.py管理用户输入、viewer.py生成离线点云HTML。
**Tech Stack:** OpenCV contrib、NumPy、Python、标准HTML/Canvas、pytest。

### Task 1: 无效视差修复
Files: src/stereolab/matching.py; tests/test_matching.py。
- [x] 用左63/右无效-64的例子写失败测试，不得将无效哨兵判为对应点。
```python
assert not left_right_consistency(left_disparity, invalid_right_disparity).any()
```
- [x] 在构造右图视差时明确排除OpenCV无效哨兵，保留有效负值，跑匹配测试。

### Task 2: WLS对比
Files: matching.py; benchmark.py; metrics.py; requirements.txt; tests/test_metrics.py。
- [x] 使用独立contrib环境，测试缺少ximgproc时可解释报错。
- [x] 增加WLS配置与验证选择，固定13/14划分；新增报告而不覆盖旧结果。
- [x] 实跑27场景，输出覆盖率、EPE、Bad1/2/3、耗时和配置来源；保留所有方法实际表现。

### Task 3: 用户入口与查看器
Files: src/stereolab/reconstruct.py; viewer.py; tests/test_reconstruct.py。
- [x] 写合成平移图的端到端测试、尺寸/标定错误测试、无真值不报精度测试。
```text
python -m stereolab.reconstruct --left LEFT --right RIGHT --calibration CALIB --out OUTPUT
```
- [x] 导出视差/深度/PLY/JSON与离线HTML交互点云，默认点数限制避免页面卡死。
- [x] 实跑一个官方场景并目视检查报告；跑全部测试；更新文档与学习说明。
- [x] 经独立审查修正后提交；按用户当前指令暂不推送。

实施记录：独立 contrib 环境在仓库外；v3保留完整24组验证候选与冻结参数，v2未修改；合成平面深度中位数900 mm；真实forest_1s点云259,147点，HTML上限20,000点。已检查真实视差/深度PNG；工具安全策略禁止打开file://，未绕过，HTML真实浏览器视觉验收列为限制。

补充：按父任务确认加入由GT视差及标定推导的深度MAE/RMSE与有效深度覆盖率；先测公式/无效分母/聚合RMSE。仅重放冻结参数，断言原视差指标完全不变后追加深度报告。WLS深度RMSE退步已在README与实验说明中明确保留，不将派生误差称为硬件实测精度。

独立审查：39项测试全部通过；重新推理14留出场景×4方法，视差与深度指标均与报告一致；另用独立公式累计深度绝对误差与平方误差核对MAE/RMSE。没有阻塞问题。提交仅含源码、测试、文档、数值报告与依赖；数据及派生二进制/HTML留在本机。
