# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
from time import sleep
from datetime import datetime as dt
from typing import Optional, Any, List, Dict, Union
from src.configuration import configuration as cfg
from src.utility.gold.basic_sqlalchemy_interface import BasicSQLAlchemyInterface
from src.utility.bronze import sqlalchemy_utility
from src.utility.bronze.hashing_utility import hash_text_with_sha256
from src.model.backend_control.data_model import populate_data_instrastructure
from src.model.backend_control.llm_pool import ThreadedLLMPool
from src.utility.silver import embedding_utility
from src.utility.bronze.hashing_utility import hash_text_with_sha256
from src.utility.silver.file_system_utility import safely_create_path
from src.model.knowledgebase_control.chromadb_knowledgebase import ChromaKnowledgeBase, KnowledgeBase


class BackendController(BasicSQLAlchemyInterface):
    """
    Controller class for handling backend interface requests.
    """

    def __init__(self, working_directory: str = None, database_uri: str = None) -> None:
        """
        Initiation method.
        :param working_directory: Working directory.
            Defaults to folder 'processes' folder under standard backend data path.
        :param database_uri: Database URI.
            Defaults to 'backend.db' file under default data path.
        """
        # Main instance variables
        self._logger = cfg.LOGGER
        self.working_directory = cfg.PATHS.BACKEND_PATH if working_directory is None else working_directory
        if not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)
        self.database_uri = f"sqlite:///{os.path.join(cfg.PATHS.DATA_PATH, 'backend.db')}" if database_uri is None else database_uri

        # Database infrastructure
        super().__init__(self.working_directory, self.database_uri,
                         populate_data_instrastructure, "backend.", self._logger)
        self.base = None
        self.engine = None
        self.model = None
        self.schema = None
        self.session_factory = None
        self.primary_keys = None

        self.llm_pool = None
        self._cache = None
        
    """
    Setup and shutdown methods
    """
    def setup(self) -> None:
        """
        Method for running setup process.
        """
        self._setup_database()

        # Knowledgebase infrastructure
        self.knowledgebase_directory = os.path.join(
            self.working_directory, "knowledgebases")
        self.document_directory = os.path.join(
            self.working_directory, "library")
        safely_create_path(self.knowledgebase_directory)
        safely_create_path(self.document_directory)
        self.default_embedding_function = embedding_utility.LocalHuggingFaceEmbeddings(
            cfg.PATHS.INSTRUCT_XL_PATH
        )
        self.kbs: Dict[str, KnowledgeBase] = {}
        self.documents = {}
        for kb in self.get_objects_by_type("knowledgebase"):
            self.register_knowledgebase(kb.id, kb.handler, kb.persinstant_directory,
                                        kb.meta_data, kb.embedding_instance_id, kb.implementation)

        # LLM infrastructure
        self.llm_pool = ThreadedLLMPool()

        # Cache
        self._cache = {
            "active": {}
        }



    def shutdown(self) -> None:
        """
        Method for running shutdown process.
        """
        self.llm_pool.stop_all()
        while any(self.llm_pool.is_running(instance_id) for instance_id in self._cache):
            sleep(2.0)

    """
    LLM handling methods
    """

    def load_instance(self, instance_id: Union[str, int]) -> Optional[str]:
        """
        Method for loading a configured language model instance.
        :param instance_id: Instance ID.
        :return: Instance ID if process as successful.
        """
        instance_id = str(instance_id)
        if instance_id in self._cache:
            if not self.llm_pool.is_running(instance_id):
                self.llm_pool.start(instance_id)
                self._cache[instance_id]["restarted"] += 1
        else:
            self._cache[instance_id] = {
                "started": None,
                "restarted": 0,
                "accessed": 0,
                "inactive": 0
            }
            instance = self.get_object("instance", instance_id)
            llm_config = {
                "model_path": instance.model.path,
                "model_config": {
                    "type": instance.type,
                    "loader": instance.loader,
                    "loader_kwargs": instance.loader_kwargs,
                    "model_version": instance.model_version,
                    "gateway": instance.gateway
                }
            }

            self.llm_pool.prepare_llm(llm_config, instance_id)
            self.llm_pool.start(instance_id)
            self._cache[instance_id]["started"] = dt.now()
        return instance_id

    def unload_instance(self, instance_id: Union[str, int]) -> Optional[str]:
        """
        Method for unloading a configured language model instance.
        :param instance_id: Instance ID.
        :return: Instance ID if process as successful.
        """
        instance_id = str(instance_id)
        if instance_id in self._cache:
            if self.llm_pool.is_running(instance_id):
                self.llm_pool.stop(instance_id)
            return instance_id
        else:
            return None

    def forward_generate(self, instance_id: Union[str, int], prompt: str) -> Optional[str]:
        """
        Method for forwarding a generate request to an instance.
        :param instance_id: Instance ID.
        :param prompt: Prompt.
        :return: Instance ID.
        """
        instance_id = str(instance_id)
        self.load_instance(instance_id)
        return self.llm_pool.generate(instance_id, prompt)

    """
    Knowledgebase handling methods
    """

    def embed_via_instance(self, instance_id: Union[str, int], documents: List[str]) -> List[Any]:
        """
        Wrapper method for embedding via instance.
        :param instance_id: LLM instance ID.
        :param documents: List of documents to embed.
        :return: List of embeddings.
        """
        embeddings = []
        for document in documents:
            embeddings.append(self.forward_generate(instance_id, document))
        return embeddings

    def register_knowledgebase(self, kb_id: Union[str, int], handler: str, persistant_directory: str,  metadata: dict = None, embedding_instance_id: Union[str, int] = None, implementation: str = None) -> str:
        """
        Method for registering knowledgebase.
        :param kb_id: Knowledgebase ID.
        :param handler: Knowledgebase handler.
        :param persistant_directory: Knowledgebase persistant directory.
        :param metadata: Knowledgebase metadata.
        :param embedding_instance: Embedding instance ID.
        :param implementation: Knowledgebase implementation.
        :return: Knowledgebase ID.
        """
        kb_id = str(kb_id)
        embedding_instance_id = str(embedding_instance_id)

        self.load_instance(embedding_instance_id)

        handler_kwargs = {
            "peristant_directory": persistant_directory,
            "metadata": metadata,
            "base_embedding_function": None if embedding_instance_id is None else lambda x: self.embed_via_instance(embedding_instance_id, x),
            "implementation": implementation
        }

        self.kbs[kb_id] = {"chromadb": ChromaKnowledgeBase}[handler](
            **handler_kwargs
        )
        return kb_id

    def create_default_knowledgebase(self, uuid: str) -> int:
        """
        Method for creating default knowledgebase.
        :param uuid: UUID to locate knowledgebase under.
        :return: Knowledgebase ID.
        """
        kb_id = self.post_object("knowledgebase",
                                 persistant_directory=os.path.join(self.knowledgebase_directory, uuid))
        kb = self.get_object_by_id("knowledgebase", kb_id)
        self.register_knowledgebase(
            kb.id, kb.handler, kb.persistant_directory, kb.meta_data, kb.embedding_instance_id, kb.implementation
        )

    def delete_documents(self, kb: str, document_ids: List[Any], collection: str = "base") -> None:
        """
        Method for deleting a document from the knowledgebase.
        :param kb: Target knowledgebase.
        :param document_ids: Document IDs.
        :param collection: Collection to remove document from.
        """
        for document_id in document_ids:
            self.kbs[kb].delete_document(document_id, collection)

    def wipe_knowledgebase(self, target_kb: str) -> None:
        """
        Method for wiping a knowledgebase.
        :param target_kb: Target knowledgebase.
        """
        self.kbs[target_kb].wipe_knowledgebase()

    def migrate_knowledgebase(self, source_kb: str, target_kb: str) -> None:
        """
        Method for migrating knowledgebase.
        :param source_kb: Source knowledgebase.
        :param target_kb: Target knowledgebase.
        """
        pass

    def embed_documents(self, kb: str, documents: List[str], metadatas: List[dict] = None, ids: List[str] = None, hashes: List[str] = None, collection: str = "base", compute_metadata: bool = False) -> None:
        """
        Method for embedding documents.
        :param kb: Target knowledgebase.
        :param documents: Documents to embed.
        :param metadatas: Metadata entries for documents.
            Defaults to None.
        :param ids: Custom IDs to add. 
            Defaults to the hash of the document contents.
        :param hashes: Content hashes.
            Defaults to None in which case hashes are computet.
        :param collection: Target collection.
            Defaults to "base".
        :param compute_metadata: Flag for declaring, whether to compute metadata.
            Defaults to False.
        """
        hashes = [hash_text_with_sha256(document.page_content)
                  for document in documents] if hashes is None else hashes
        for doc_index, hash in enumerate(hashes):
            if hash not in self.documents:
                path = os.path.join(self.document_directory, f"{hash}.bin")
                open(os.path.join(self.document_directory, f"{hash}.bin"), "wb").write(
                    documents[doc_index].encode("utf-8"))
                self.documents[hash] = {
                } if metadatas is None else metadatas[doc_index]
                self.documents[hash]["controller_library_path"] = path

        if metadatas is None:
            metadatas = [self.documents[hash] for hash in hashes]

        self.kbs[kb].embed_documents(
            collection=collection, documents=documents, metadatas=metadatas, ids=hashes if ids is None else ids)

    """
    Custom methods
    """

    def embed_document(self, kb_id: Union[int, str], document_content: str, document_metadata: dict = None) -> int:
        """
        Method for embedding document.
        :param kb_id: Knowledgebase ID.
        :param document_content: Document content.
        :param document_metadata: Document metadata.
            Defaults to None
        :return: Document ID.
        """
        kb = self.get_object_by_id("knowledgebase", kb_id)
        doc_id = self.post_object("document", {
            "content": document_content,
            "knowledgebase_id": kb.id
        })
        document_metadata = {} if document_metadata is None else document_metadata
        document_metadata.update({
            "hash": hash_text_with_sha256(document_content),
            "kb_id": kb.id,
        })

        self.kb_controller.embed_documents(
            str(kb.id), documents=[document_content],
            metadatas=[document_metadata],
            ids=[str(doc_id)],
            hashes=[document_metadata["hash"]]
        )
        return doc_id

    def delete_document_embeddings(self, document_id: int) -> int:
        """
        Method for deleting document embeddings.
        :param document_id: Document ID.
        :return: Document ID.
        """
        doc = self.get_object_by_id("document", document_id)
        self.kb_controller.delete_documents(
            str(doc.knowledgebase_id),
            [str(doc.id)]
        )

    def forward_document_qa(self, llm_id: Union[int, str], kb_id: Union[int, str], query: str, include_sources: bool = True) -> dict:
        """
        Method for posting query.
        :param llm_id: LLM ID.
        :param kb_id: Knowledgebase ID.
        :param query: Query.
        :param include_sources: Flag declaring, whether to include sources.
        :return: Response.
        """
        docs = self.kbs[kb_id].get_retriever(
        ).get_relevant_documents(query=query)

        document_list = "'''" + "\n\n '''".join(
            [doc.page_content for doc in docs]) + "'''"
        generation_prompt = f"Answer the question '''{query}''' with the following information: \n\n {document_list}"

        response = self.forward_generate(llm_id, generation_prompt)

        return response, [doc.metadata for doc in docs] if include_sources else []
