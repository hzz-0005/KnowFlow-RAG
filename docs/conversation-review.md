# 多轮对话复核包

在某轮“查看详情”底部点击“导出截至本轮的复核包”。这是本地下载，包含该会话第1轮至所选轮的原话、回答、保存的任务条件/更正、记忆身份线索与引用节选，不含后续轮次。只读GET，不调用模型、不重检索、不改变会话。默认不导出模型审核结果、模型身份、相关性判分，人工复核表全部未填写。

这是一份开发者已经看过的材料，不自动成为盲测或独立金标。只保存回答当时的引用节选，没有原文全文或完整候选池；证据完整性、正确拒答、Recall@k等仍可能未知。不会用现在的文档替过去补证据，也不会使用最新任务/记忆反写旧轮目标。旧恢复轮次的原目标只允许按既有规则从同用户同库任务源原话重建，并记录goal_basis与源轮次ID。

## 复核内容

| 字段 | 人工要核查什么 |
|---|---|
| referent_resolution | “它/这笔/刚才那个”是否指向正确对象，歧义是否保留 |
| scope_and_corrections | 当前否定、日期、范围和更正是否优先于旧条件 |
| topic_boundary | 切换话题时是否错误继承旧对象或约束 |
| task_completeness | 所需动作/必要内容是否完成，不能用执行结束代替 |
| source_support | 保存的引用节选是否支持相关断言，不能把来源存在当蕴含 |
| abstention_appropriateness | 证据缺口是否说清；仅凭没找回不能断言文档没有答案 |

模板每维度取unreviewed/satisfied/violated/unknown/not_applicable。未审核保持unknown观测，不填0。未知和不适用也须写理由；不适用只是标注人的声明，不由程序自动证明。

先从原话和来源建立判据，再看回答。系统所选requirement_sources保留为当时轨迹，**不是人工可查看历史的边界**：request_ref.source_id可用`current`、保存的`task_goal`/`task_update_*`/`history_*`，也可用`turn:<快照中任意已有轮次ID>`引用完整历史原话，从而指出系统漏选历史的错误。后续轮次和记忆线索不能作要求来源。

判satisfied/violated需原请求与最终回答的非空连续节选；source_support还需citation_refs，格式`{citation_id:"<目标轮ID>:c1",quote:"引用中的原话"}`。无充分节选则用unknown并解释。字面存在只检查可追溯性，程序不能证明蕴含、要求覆盖或判断正确。

标注填写declared_method（human/model/model_assisted_human）、reviewer_id和带时区reviewed_at。这些仅为来源声明；即使全部填写，独立复核仍为not_established，task_success为空，没有自动成功率。独立标签、M1外审、M2协议和代表性样本仍需要真实复核证据。

## 离线校验

在仓库根目录运行，输出必须是新目录：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m knowflow.app.review_conversation_bundle `
  --bundle <下载的复核包.json> --out <新的输出目录>
```

得到summary.md、report.json、annotations.json和audit.json。复制/编辑annotations.json后，用`--annotations <标注文件>`写入另一个新输出目录；原下载包和旧结果保持原样。该CLI不导入数据库或供应商客户端，不读取配置/密钥，不联网。

- 退出0：数据格式、哈希和字面引用校验完成，不是质量通过。
- 退出2：输入/格式/哈希/节选错误，或输出目录已存在；已有输出不覆盖。
- 加`--require-annotations`时，仍有unreviewed维度返回3。它只是填写完整性闸门，不是独立复核或语义通过闸门；明确unknown可保留。

SHA256绑定每轮、整体快照、标注和输入/输出原文件；相同快照重复导出保持snapshot_sha256，导出时间单列。哈希检测内容变化，不是签名，不证明生产来源或复核身份。JSON显式UTF-8、拒绝重复键及NaN/Infinity。最多100轮及2MiB canonical JSON，超出不默默截断；文件读取另有16MiB上限。

接口：`GET /api/v1/conversations/{cid}/turns/{tid}/review-export`。需会话与知识库所有者权限；跨用户/跨会话目标404，所选轮仍在执行409，超过上限413；历史缺轮明确拒绝。导出失败不会创建不完整复核包，响应Cache-Control:no-store。

本包不修改M1/M2冻结契约或保留集，不运行质量基线，不把已用开发案例计成新盲测。交付记录见复核包验收（本地研究记录，未随公开包分发）。
