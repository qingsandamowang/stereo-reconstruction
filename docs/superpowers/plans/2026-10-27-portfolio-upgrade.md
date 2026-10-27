# Portfolio Upgrade Implementation Plan

> **For agentic workers:** 按任务使用 executing-plans 或 subagent-driven-development 执行；每个独立项目都有单独计划，必须先完成安全测试再集成。

**Goal:** 完成修复后的多算法评测、自定义输入和离线点云查看。
**Architecture:** matching.py管理视差算法与无效值，benchmark.py管理验证选择，新增reconstruct.py管理用户输入、viewer.py生成离线点云HTML。
**Tech Stack:** OpenCV contrib、NumPy、Python、标准HTML/Canvas、pytest。

### Task 1: 无效视差修复
Files: src/stereolab/matching.py; tests/test_matching.py。
- [ ] 用左63/右无效-64的例子写失败测试，不得将无效哨兵判为对应点。
```python
assert not left_right_consistency(left_disparity, invalid_right_disparity).any()
```
- [ ] 在构造右图视差时明确排除OpenCV无效哨兵，保留有效负值，跑匹配测试。

### Task 2: WLS对比
Files: matching.py; benchmark.py; metrics.py; requirements.txt; tests/test_metrics.py。
- [ ] 使用独立contrib环境，测试缺少ximgproc时可解释报错。
- [ ] 增加WLS配置与验证选择，固定13/14划分；新增报告而不覆盖旧结果。
- [ ] 实跑27场景，输出覆盖率、EPE、Bad1/2/3、耗时和配置来源；保留所有方法实际表现。

### Task 3: 用户入口与查看器
Files: src/stereolab/reconstruct.py; viewer.py; tests/test_reconstruct.py。
- [ ] 写合成平移图的端到端测试、尺寸/标定错误测试、无真值不报精度测试。
```text
python -m stereolab.reconstruct --left LEFT --right RIGHT --calibration CALIB --out OUTPUT
```
- [ ] 导出视差/深度/PLY/JSON与离线HTML交互点云，默认点数限制避免页面卡死。
- [ ] 实跑一个官方场景并目视检查报告；跑全部测试；更新文档与学习说明。
- [ ] 经独立审查修正后提交并推送。
