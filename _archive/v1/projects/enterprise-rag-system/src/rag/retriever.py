"""
RAG检索器模块 - 混合检索实现
"""

import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """检索结果文档"""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str  # 'dense', 'sparse', 'rerank'


class HybridRetriever:
    """混合检索器 - 结合稠密检索和稀疏检索"""
    
    def __init__(
        self,
        embedding_model,
        vector_store,
        keyword_store,
        reranker=None,
        top_k_retrieve: int = 20,
        top_k_rerank: int = 5,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.reranker = reranker
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
    
    async def retrieve(
        self, 
        query: str, 
        filters: Optional[Dict] = None
    ) -> List[RetrievedDocument]:
        """
        执行混合检索
        
        Args:
            query: 查询文本
            filters: 元数据过滤条件
            
        Returns:
            检索到的文档列表
        """
        # 并行执行稠密检索和稀疏检索
        dense_task = self._dense_retrieve(query, filters)
        sparse_task = self._sparse_retrieve(query, filters)
        
        dense_results, sparse_results = await asyncio.gather(
            dense_task, sparse_task
        )
        
        logger.info(f"Dense retrieved: {len(dense_results)}, "
                   f"Sparse retrieved: {len(sparse_results)}")
        
        # 融合结果
        fused_results = self._fuse_results(dense_results, sparse_results)
        
        # 重排序
        if self.reranker and len(fused_results) > 0:
            reranked_results = await self._rerank(query, fused_results)
            return reranked_results[:self.top_k_rerank]
        
        return fused_results[:self.top_k_rerank]
    
    async def _dense_retrieve(
        self, 
        query: str, 
        filters: Optional[Dict] = None
    ) -> List[RetrievedDocument]:
        """稠密向量检索"""
        # 生成查询向量
        query_embedding = await self.embedding_model.embed_query(query)
        
        # 从向量数据库检索
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k_retrieve,
            filters=filters
        )
        
        return [
            RetrievedDocument(
                id=r['id'],
                content=r['content'],
                metadata=r['metadata'],
                score=r['score'],
                source='dense'
            )
            for r in results
        ]
    
    async def _sparse_retrieve(
        self, 
        query: str, 
        filters: Optional[Dict] = None
    ) -> List[RetrievedDocument]:
        """稀疏关键词检索 (BM25)"""
        results = await self.keyword_store.search(
            query=query,
            top_k=self.top_k_retrieve,
            filters=filters
        )
        
        return [
            RetrievedDocument(
                id=r['id'],
                content=r['content'],
                metadata=r['metadata'],
                score=r['score'],
                source='sparse'
            )
            for r in results
        ]
    
    def _fuse_results(
        self, 
        dense_results: List[RetrievedDocument], 
        sparse_results: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """
        融合稠密和稀疏检索结果 (RRF - Reciprocal Rank Fusion)
        """
        # 创建ID到文档的映射
        doc_map: Dict[str, RetrievedDocument] = {}
        
        # RRF参数
        k = 60
        
        # 计算RRF分数
        rrf_scores: Dict[str, float] = {}
        
        # 处理稠密检索结果
        for rank, doc in enumerate(dense_results):
            doc_map[doc.id] = doc
            rrf_scores[doc.id] = rrf_scores.get(doc.id, 0) + \
                                self.dense_weight / (k + rank + 1)
        
        # 处理稀疏检索结果
        for rank, doc in enumerate(sparse_results):
            doc_map[doc.id] = doc
            rrf_scores[doc.id] = rrf_scores.get(doc.id, 0) + \
                                self.sparse_weight / (k + rank + 1)
        
        # 按RRF分数排序
        sorted_ids = sorted(
            rrf_scores.keys(), 
            key=lambda x: rrf_scores[x], 
            reverse=True
        )
        
        # 重建文档列表
        fused_results = []
        for doc_id in sorted_ids[:self.top_k_retrieve]:
            doc = doc_map[doc_id]
            doc.score = rrf_scores[doc_id]
            fused_results.append(doc)
        
        return fused_results
    
    async def _rerank(
        self, 
        query: str, 
        documents: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """使用重排序模型优化结果顺序"""
        if not self.reranker:
            return documents
        
        # 准备重排序输入
        pairs = [(query, doc.content) for doc in documents]
        
        # 调用重排序模型
        scores = await self.reranker.rerank(pairs)
        
        # 更新分数并排序
        for doc, score in zip(documents, scores):
            doc.score = score
            doc.source = 'rerank'
        
        return sorted(documents, key=lambda x: x.score, reverse=True)


class QueryRewriter:
    """查询改写器 - 优化查询以提高检索效果"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def rewrite(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """
        改写查询以改善检索效果
        
        策略：
        1. 消除歧义
        2. 扩展同义词
        3. 处理指代消解
        """
        if not conversation_history:
            return query
        
        # 使用LLM进行查询改写
        prompt = self._build_rewrite_prompt(query, conversation_history)
        
        try:
            rewritten = await self.llm.generate(prompt, temperature=0.3)
            return rewritten.strip()
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}, using original query")
            return query
    
    def _build_rewrite_prompt(
        self, 
        query: str, 
        history: List[Dict]
    ) -> str:
        """构建改写提示"""
        history_text = "\n".join([
            f"User: {h.get('user', '')}\nAssistant: {h.get('assistant', '')}"
            for h in history[-3:]  # 只使用最近3轮
        ])
        
        return f"""基于以下对话历史，理解用户最新查询的意图，并改写为一个清晰、完整的搜索查询。

对话历史：
{history_text}

用户最新查询：{query}

请直接输出改写后的查询，不要添加任何解释："""
    
    async def expand(self, query: str) -> List[str]:
        """
        查询扩展 - 生成多个相关查询
        """
        prompt = f"""为以下查询生成3个语义相似但表述不同的搜索查询：

原查询：{query}

要求：
1. 保持核心意图不变
2. 使用不同的关键词
3. 每个查询占一行

相关查询："""
        
        try:
            response = await self.llm.generate(prompt, temperature=0.7)
            expanded = [q.strip() for q in response.strip().split('\n') if q.strip()]
            return [query] + expanded[:3]
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return [query]


class RAGPipeline:
    """完整RAG流水线"""
    
    def __init__(
        self,
        retriever: HybridRetriever,
        query_rewriter: QueryRewriter,
        llm_client,
        prompt_template: Optional[str] = None
    ):
        self.retriever = retriever
        self.query_rewriter = query_rewriter
        self.llm = llm_client
        self.prompt_template = prompt_template or self._default_template()
    
    async def query(
        self, 
        user_query: str,
        conversation_history: Optional[List[Dict]] = None,
        filters: Optional[Dict] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        执行完整RAG查询
        
        Args:
            user_query: 用户查询
            conversation_history: 对话历史
            filters: 检索过滤条件
            stream: 是否流式输出
            
        Returns:
            包含回答和引用的字典
        """
        # 1. 查询改写
        rewritten_query = await self.query_rewriter.rewrite(
            user_query, conversation_history
        )
        logger.info(f"Query rewritten: '{user_query}' -> '{rewritten_query}'")
        
        # 2. 文档检索
        documents = await self.retriever.retrieve(rewritten_query, filters)
        logger.info(f"Retrieved {len(documents)} documents")
        
        # 3. 构建Prompt
        context = self._build_context(documents)
        prompt = self._build_prompt(user_query, context, conversation_history)
        
        # 4. 生成回答
        if stream:
            return {
                "stream": self.llm.generate_stream(prompt),
                "documents": documents,
                "rewritten_query": rewritten_query
            }
        
        answer = await self.llm.generate(prompt)
        
        return {
            "answer": answer,
            "documents": documents,
            "rewritten_query": rewritten_query
        }
    
    def _build_context(self, documents: List[RetrievedDocument]) -> str:
        """构建上下文文本"""
        contexts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            contexts.append(
                f"[Document {i}] Source: {source}\n{doc.content}\n"
            )
        return "\n---\n".join(contexts)
    
    def _build_prompt(
        self, 
        query: str, 
        context: str,
        history: Optional[List[Dict]] = None
    ) -> str:
        """构建生成提示"""
        history_text = ""
        if history:
            history_text = "\n".join([
                f"User: {h.get('user', '')}\nAssistant: {h.get('assistant', '')}"
                for h in history[-2:]
            ])
        
        return f"""基于以下参考文档回答用户问题。如果文档中没有相关信息，请明确说明。

参考文档：
{context}

对话历史：
{history_text}

用户问题：{query}

请提供准确、简洁的回答，并在回答末尾列出引用来源："""
    
    def _default_template(self) -> str:
        """默认提示模板"""
        return """基于以下参考文档回答用户问题。

参考文档：
{context}

用户问题：{query}

请提供准确、简洁的回答："""
