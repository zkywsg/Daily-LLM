# 实战项目

把多个架构家族的知识组装成可以真实交付的系统。

## 本阶段项目

### [企业级 RAG 系统](enterprise-rag-system/README.md)
支持百万文档检索的生产级 RAG 系统
- 文档处理 → Embedding → 向量存储 → 检索增强 → 生成 → 评估 → 上线监控
- 对应技术：Phase 03（LoRA）+ Phase 04（对齐）+ Phase 05（RAG + vLLM）

### [自动化微调与部署流水线](finetune-deploy-pipeline/README.md)
从数据准备到上线监控的端到端 LLM 微调工程
- 指令数据构造 → LoRA/QLoRA 微调 → DPO 对齐 → 评估 → vLLM 部署 → 监控
- 对应技术：Phase 04（PEFT + 对齐）+ Phase 05（服务 + MLOps）

→ 按年份查看全部模型：[TIMELINE.md](../TIMELINE.md)

→ 返回[项目主页](../README.md)
