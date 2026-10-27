# 本地回归测试

浏览器自然口吻探索与固定回归分开，见[多轮浏览器测试规则](conversational-browser-testing.md)。实际脚本版本、阶段故障、连续追问依赖及原始失败都必须保留；执行完成数不代表答案正确数。

运行 PostgreSQL 后，在项目根目录执行：

```powershell
.venv/Scripts/python.exe scripts/run_tests.py -q
# 仅诊断相关用例：
.venv/Scripts/python.exe scripts/run_tests.py -q backend/tests/test_ingestion_diagnostics.py backend/tests/test_diagnostics_api.py
```

脚本使用配置中的数据库连接建立管理连接，新建随机命名的 `knowflow_tests_<uuid>` 数据库，仅在该库升级至最新迁移并运行 pytest。最后删除本次创建的库；它不向应用库写测试行，不停止 Docker 或线上 worker。测试失败返回原退出码；建库权限不足会直接失败，不回退到应用库。默认角色需要 CREATE DATABASE 权限，测试所需扩展权限与项目迁移一致。

直接执行 `pytest` 仍会使用当前 DATABASE_URL，线上 API 的持久 worker 可能先领取测试创建的任务；普通 ASGI 客户端不启动 lifespan 不能隔离另一个进程的 worker。不要用“重跑直到通过”处理这一竞态。离线测试仍使用已有 mock，诊断接口本身没有供应商调用；脚本不为付费端到端实验提供沙箱。

可设置 `KNOWFLOW_TEST_EVIDENCE_DIR` 为输出目录，逐条保存迁移与 pytest 的 argv/cwd/起止时间/退出码、stdout/stderr 及 SHA-256，另存临时库创建/删除状态。连接凭据只通过子进程环境传递，不写入参数或证据。

本轮保留了共享库的一次失败输出，未将其涂改为通过；最终通过结果来自隔离库，见 诊断交付记录（本地研究记录，未随公开包分发）。


## v0.1.0公开包的可复现范围

后端通过`uv sync --locked --extra dev`安装。公开包不带私人的历史results、真实年报/论文及检查截图；M2门槛测试改用即时生成的自写合成PDF/数值/标注夹具，仍检查单位、符号、范围、来源指纹、保留集和假审核标记。真实档案的两项核对，仅在公开包SOURCE_MANIFEST存在而原档案未分发时明确skip；本地原档案保留且仍可执行。这不是给失败代码永久豁免。

`evaluation/unified_example/manifest.json`可用于离线账本演示；退出0仅表示核算完成，不是质量gate。示例数据全部合成，主语义指标保持unknown。

`compose.smoke.yml`是独立的软件协议stub栈，不读取真实.env，也不调用供应商；`scripts/smoke_test.py`核对账号隔离、文件处理、引用、持久聊天和有来源的图谱接口。它不验证模型、指代理解、关系语义或检索质量。当前付费模型的效果仍需另外固定协议和真实标注。

用户已验收当前前端，本轮发布整理不再做页面测试；前端自动检查与干净构建继续验证发布可复现性。GitHub Actions的远端运行结果须待上传后检查，不能用本机通过冒称远端CI绿。
