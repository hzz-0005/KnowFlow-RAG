# 有界问题提炼与静态依赖规划

2026-10-25，当前`quoted-constraints-static-plan-v2-source-ids`，是C2的有限检索规划，不是通用任务代理或任务记忆。v1首次页面回退已保留；v2明确源ID命名空间和安全失败计数，不放宽原始输入引用校验。

混合/混合重排/图文混合查询在本轮用户原话>=240字符，或出现先…再/first…then顺序时，以一次规划替代原一次改写；普通短问题继续旧中英改写。`TASK_PLANNING_ENABLED=false`可关闭新分支；`QUERY_REWRITE_ENABLED=false`时也不启用。vector/keyword/graph单策略暂不启用。触发只是运行分流，可能多触发或漏触发，不是问题难度分类准确率。

规划输入包含本轮完整原话、必要的历史用户问题/补充、解析后的问题线索；历史助手回答不进入。目标、步骤与条件是模型输出，只能作为检索线索，不能替代原始请求。用户自己输入的JSON不被当作服务器intent_context。

结构与边界：

- 最多8个条件，必须在明确用户输入中唯一逐字定位；保存source_id及0起始、右端不含的Python字符偏移，不是PDF坐标。global条件关联所有步骤，local条件至少关联一步。只能证明出处与关联形式，不能证明模型提取了所有必要条件或正确判断global/local。
- 最多3个有序步骤；依赖只能指向更早步骤，拒绝未知/重复/自身/前向引用。最多2个无依赖lookup编译为独立检索线索；拼上所关联条件原文，原问题始终进入同一retriever。原MAX_QUERY_PARTS=4、k、RRF、展开和上下文额度不扩大。
- 步骤数字必须来自其声明的用户输入或关联条件；若数字仅属于未关联的其他局部条件，整组拒绝。精确局部标注缺失/过宽时仍不能证明数字角色与对象对应，coverage_status固定unverified，不把成员集合校验当语义保真。
- 带依赖lookup标waiting_on_evidence，不把未知上游答案猜成查询参数；calculate/synthesize仅记计划，不单独执行。没有变量替换、工具执行、程序计算、逐步答案或任务完成率。完整原查询仍可召回这些主题的相关材料，但不代表依赖步骤已被独立执行。

静态线索共享同一个候选池与排序，不保证每一步都召回了必要证据。retrieval_status=shared_retriever_returned只说明调用返回，shared_candidate_count是总候选数，不是每步覆盖。尚无逐子问题候选贡献表或独立证据覆盖指标。

复杂分支不为每个子问题再生成中英双份查询，以免突破现有查询数量预算；它与旧双语改写属于不同策略，跨语言效果仍需独立对照，不能直接宣称召回更好。

格式/依赖/quote/数字校验任一失败，整组不用，沿原问题检索；不会再调一个改写器补救。只采用stop响应，超限、取消、缺用量、封闭错误原因沿旧保护。使用query_rewrite阶段的完整提示预算/2048输出预留和原超时，在线进入整轮预算task_planning请求；无需新增供应商key。不截断原始条件来凑规划容量。

规划成功不会改写回答任务：相关性、生成和核验仍接收完整用户原话及引用的历史条件；目标/步骤不会作为文档事实传入。模式、提示指纹、触发规则、数值/条件契约加入query_rewrite_contract，新旧合同不同不能硬算提升。旧轮次不回填任务计划。详情在“问题提炼与拆解”显示原话条件、步骤依赖、共享检索和未执行状态。

参考：[LlamaIndex Query Transform](https://developers.llamaindex.ai/python/examples/query_transformations/query_transform_cookbook/)、[SubQuestionQueryEngine](https://developers.llamaindex.ai/python/examples/query_engine/sub_question_query_engine/)展示分解与聚合的信息流；[ACL2025问题分解研究](https://aclanthology.org/2025.acl-srw.32/)讨论子问题召回/合并/重排。借鉴成熟结构，不引入其框架，也不把论文收益迁移成本项目效果。

验收证据见工作包（本地研究记录，未随公开包分发）。后续C2仍需复杂对象/隐含条件与迁移样本，C3需要可恢复依赖执行/任务状态，C4长期记忆需要来源/有效期/更正/删除，C5仍需独立多轮语义评测。M1–M7原路线不变。
