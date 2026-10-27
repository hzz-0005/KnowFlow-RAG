# 合成账本示例

这些输入完全来自软件测试构造，不是真实问答、人工标注或质量成绩。

在根目录运行 `uv run python -m knowflow.app.evaluate_unified --manifest evaluation/unified_example/manifest.json --out evaluation/results/example-01`；输出目录必须尚不存在。CLI不调用模型/数据库。主语义指标没有标注，保持unknown；0退出只证明核算完成。

历史M1真实资料不随公开包再分发；相关契约反例在backend/tests/test_evaluation_replay.py中运行。
