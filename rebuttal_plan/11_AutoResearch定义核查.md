# Auto Research术语核查（2026-10-27）

## 结论

代表性工作支持用hypothesis–experiment–evaluation–revision概括自主研究的核心流程，但没有据此建立一个统一、充分必要的正式定义。AutoIR符合该流程层面的模式，可以表述为domain-specific instantiation of Auto Research for restoration-model development。这是定位判断，不是独立的新颖性或优先权证明。

## 原始来源

- The AI Scientist (2024)：提出idea generation、实现和实验迭代、总结、论文与反馈审阅；使用既有代码模板，并将反馈用于后续研究。https://sakana.ai/ai-scientist/
- Coscientist (Nature 2023)：以工具支持实验设计、规划、执行和基于先前实验数据的优化，明确为特定任务下的(semi-)autonomous系统。https://www.nature.com/articles/s41586-023-06792-0
- Agent Laboratory (2025)：人提供研究idea，系统进行文献回顾、实验和报告撰写，并允许人类反馈。https://arxiv.org/abs/2501.04227
- Google AI co-scientist (2025)：明确描述自动反馈下反复生成、评价、修订hypotheses；实验验证环节包含专家协作。https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/

## 与AutoIR的对应

DC的可执行退化假设与模拟验证对应hypothesis；RS的数据生成、配置及训练对应experiment；支持集复原指标与训练动态对应evaluation；CLR修改DC或RS并重新训练对应revision。工具/模块由人预置不自动否定这类研究自动化定位，但自主范围需要明确。

不应将“包含闭环”直接等同于完整通用科学发现，也不应声称已有文献把AutoIR正式划为一个公认sub-branch。使用“we position/instantiate”表达作者的定位；“rethinking”建议具体写作evidence-based revision或reflection。
