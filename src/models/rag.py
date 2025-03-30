# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Optional
import os
from time import time
from glob import glob
from typing import List, Dict, Literal
import numpy as np
import pdb

from llama_index.core import VectorStoreIndex, Document, SimpleKeywordTableIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import QueryBundle, QueryType, NodeWithScore
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever, KeywordTableSimpleRetriever
from llama_index.core.storage.index_store.simple_index_store import SimpleIndexStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.vector_stores.types import VectorStoreQueryMode

Settings.llm = None

class LlamaIndexRetriever(object):
    def __init__(self, chunk_list: List[Dict[str, str]] = [], 
                       storage_path: str = './data/textbooks/rag_storage',
                       emb_model_path: str = "local:../../models/rag_embedding/bge-m3",
                       chunk_size: int = 1024, similarity_top_k: int = 3, hybrid_search: bool = False, 
                       reranker_path: Optional[str] = None, rerank_top_n: int = 3,
                       **kwargs):
        os.makedirs(storage_path, exist_ok=True)
        if len(os.listdir(storage_path)) == 0:
            assert len(chunk_list) > 0
            documents = [Document(text=chunk['data'], doc_id=chunk['idx']) for chunk in chunk_list]
            node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=0)
            nodes = node_parser.get_nodes_from_documents(documents)
            self.index = VectorStoreIndex(nodes, embed_model=emb_model_path, show_progress=True, **kwargs)
            self.index.storage_context.persist(storage_path)
        else:
            print('Loading LlamaIndex Storage ...')
            t0 = time()
            storage_context = StorageContext.from_defaults(persist_dir=storage_path)
            self.index = load_index_from_storage(storage_context, embed_model=emb_model_path)
            print(f'Done in {time() - t0:.1} seconds.')

        self.retriever: VectorIndexRetriever = self.index.as_retriever(similarity_top_k=similarity_top_k)
        if hybrid_search:
            nodes = list(self.index.storage_context.docstore.docs.values())
            self.keyword_index = SimpleKeywordTableIndex(nodes, show_progress=True)
            self.keyword_retriever: KeywordTableSimpleRetriever = \
                self.keyword_index.as_retriever(num_chunks_per_query=similarity_top_k)
        else:
            self.keyword_retriever = None

        if reranker_path:
            self.rerank = SentenceTransformerRerank(top_n=rerank_top_n, model=reranker_path)
        else:
            self.rerank = None

    def retrieve(self, query, return_text=True):
        nodes: List[NodeWithScore] = self.retriever.retrieve(query)
        
        if self.keyword_retriever:  # hybrid search
            kw_nodes: List[NodeWithScore] = self.keyword_retriever.retrieve(query)
            node_dict = {node.node.node_id: node for node in (nodes + kw_nodes)}
            nodes = list(node_dict.values())  # union

        if self.rerank:
            nodes = self.rerank.postprocess_nodes(nodes, query_bundle=QueryBundle(query))
            
        if return_text:
            nodes = [node.get_text() for node in nodes]
            
        return nodes

    def set_topk(self, topk):
        if self.rerank:
            self.rerank.top_n = topk
        else:
            self.retriever.similarity_top_k = topk
            if self.keyword_retriever:
                self.keyword_retriever.num_chunks_per_query = topk


if __name__ == '__main__':
    chunk_size = 512
    prefix = '_sp'
    llama_index = LlamaIndexRetriever(
        data_dir=f'./data/textbooks/rag_doc{prefix}_{chunk_size}',
        storage_path=f'./data/textbooks/rag_storage{prefix}_{chunk_size}',
        chunk_size=chunk_size
        # reranker_path='../../models/rag_embedding/bge-reranker-v2-m3'
    )
    # query = "A 79-year-old man presents to the office due to shortness of breath with moderate exertion and a slightly productive cough. He has a medical history of 25 years of heavy smoking. His vitals include: heart rate 89/min, respiratory rate 27/min, and blood pressure 120/90 mm Hg. The physical exam shows increased resonance to percussion, decreased breath sounds, and crackles at the lung base. Chest radiography shows signs of pulmonary hyperinflation. Spirometry shows a forced expiratory volume in the first second (FEV1) of 48%, a forced vital capacity (FVC) of 85%, and an FEV1/FVC ratio of 56%. According to these results, what is the most likely diagnosis?"
    # query = "经调查证实出现医院感染流行时，医院应报告当地卫生行政部门的时间是（　　）。"
    # query = "Parmi les techniques voltampérométriques, on trouve:"
    # query = 'Противокашлевое действие бутамирата цитрата обусловлено главным образом воздействием на кашлевой центр в мозге?'
    query = 'El complejo proteico responsable de la ramificación de los filamentos de actina es:'
    res = llama_index.retrieve(query, return_text=False)
    print(len(res))
    chunk = res[0].get_text()
    print(len(chunk.split()))
    # print(chunk)
    print(res[0].metadata['file_name'])  # file_idx