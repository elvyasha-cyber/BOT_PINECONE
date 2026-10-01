"""Управление чтением и записью в векторную базу Pinecone."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import DenseVectorQuery, Pinecone, ServerlessSpec, TextQuery

load_dotenv()

EmbedFn = Callable[[str], Sequence[float]]

# Граница косинусного сходства для долговременной памяти.
# Ниже порога — информация новая, записываем.
# Больше или равно порогу — дубликат или вариация.
COSINE_SIMILARITY_THRESHOLD = 0.5

# Что делать при высоком сходстве: "update" — обновить существующий слот, "skip" — не писать.
DUPLICATE_MEMORY_POLICY = "update"


class PineconeManager:
    """Клиент Pinecone: запись векторов и документов, поиск по вектору и по тексту."""

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str | None = None,
        cloud: str | None = None,
        region: str | None = None,
        namespace: str = "default",
        dimension: int = 1536,
        metric: str = "cosine",
        pinecone_base_url: str | None = None,
        proxy_url: str | None = None,
        openai_api_key: str | None = None,
        openai_model: str | None = None,
        openai_base_url: str | None = None,
    ) -> None:
        self.api_key = (api_key or os.getenv("PINECONE_API_KEY", "")).strip()
        if not self.api_key:
            raise RuntimeError("Не задан PINECONE_API_KEY.")

        self.index_name = (
            index_name or os.getenv("PINECONE_INDEX_NAME", "bot-memory")
        ).strip() or "bot-memory"
        self.cloud = (cloud or os.getenv("PINECONE_CLOUD", "aws")).strip() or "aws"
        self.region = (region or os.getenv("PINECONE_REGION", "us-east-1")).strip() or "us-east-1"
        self.namespace = namespace or "default"
        self.dimension = dimension
        self.metric = metric
        self.pinecone_base_url = (
            pinecone_base_url or os.getenv("PINECONE_BASE_URL") or os.getenv("PINECONE_HOST") or ""
        ).strip()
        self.proxy_url = (proxy_url or os.getenv("PINECONE_PROXY_URL") or "").strip()

        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.pinecone_base_url:
            client_kwargs["host"] = self.pinecone_base_url
        if self.proxy_url:
            client_kwargs["proxy_url"] = self.proxy_url
        self._pc = Pinecone(**client_kwargs)
        self._index = None

        self.openai_api_key = (openai_api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        self.openai_model = (
            openai_model or os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small"
        ).strip()
        self.openai_base_url = (openai_base_url or os.getenv("OPENAI_BASE_URL") or "").strip()
        self.openai_client = None
        if self.openai_api_key:
            if not self.openai_base_url:
                raise RuntimeError(
                    "Не задан OPENAI_BASE_URL. Укажите адрес прокси к моделям в .env."
                )
            self.openai_client = OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url,
            )

        self.similarity_threshold = COSINE_SIMILARITY_THRESHOLD
        self.duplicate_policy = DUPLICATE_MEMORY_POLICY

    @property
    def index(self):
        if self._index is None:
            if not self._pc.has_index(self.index_name):
                raise RuntimeError(
                    f"Индекс {self.index_name!r} не найден. Сначала вызовите ensure_index()."
                )
            self._index = self._pc.Index(self.index_name)
        return self._index

    def _ns(self, namespace: str | None) -> str:
        return namespace if namespace not in (None, "") else self.namespace

    def ensure_index(self) -> None:
        """Создаёт dense-индекс, если его ещё нет, и подключается к нему."""
        if not self._pc.has_index(self.index_name):
            self._pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
                timeout=None,
            )
        self._index = self._pc.Index(self.index_name)

    # --- запись векторов ---

    def upsert_vector(
        self,
        values: Sequence[float],
        *,
        vector_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        namespace: str | None = None,
        check_similarity: bool = True,
        threshold: float | None = None,
        on_duplicate: str | None = None,
    ) -> dict[str, Any]:
        """Запись вектора в Pinecone с проверкой косинусного сходства.

        Returns:
            action: inserted / updated / skipped
            similarity_score: сходство с ближайшим вектором, если найден
            existing_id: id найденного слота, если найден
            id: id слота, который в итоге относится к этой записи
        """
        vector_id = vector_id or str(uuid.uuid4())
        payload = dict(metadata or {})
        result: dict[str, Any] = {
            "action": "inserted",
            "similarity_score": None,
            "existing_id": None,
            "id": vector_id,
            "duplicate": False,
        }

        if check_similarity:
            similar = self._check_similarity(values, namespace=namespace, threshold=threshold)
            if similar:
                existing_id = similar["id"]
                result["similarity_score"] = similar["score"]
                result["existing_id"] = existing_id
                result["id"] = existing_id
                result["duplicate"] = True
                policy = (on_duplicate or self.duplicate_policy or DUPLICATE_MEMORY_POLICY).strip().lower()
                if policy == "skip":
                    result["action"] = "skipped"
                    return result
                result["action"] = "updated"
                self.index.upsert(
                    vectors=[
                        {
                            "id": existing_id,
                            "values": list(values),
                            "metadata": payload,
                        }
                    ],
                    namespace=self._ns(namespace),
                )
                return result

        self.index.upsert(
            vectors=[
                {
                    "id": vector_id,
                    "values": list(values),
                    "metadata": payload,
                }
            ],
            namespace=self._ns(namespace),
        )
        return result

    def _check_similarity(
        self,
        vector: Sequence[float],
        *,
        namespace: str | None = None,
        threshold: float | None = None,
    ) -> dict[str, Any] | None:
        """Возвращает ближайший вектор, если сходство >= порога, иначе None."""
        matches = self.query_by_vector(vector, top_k=1, namespace=namespace)
        if not matches:
            return None
        best = matches[0]
        score = best.get("score")
        if score is None:
            return None
        cutoff = self.similarity_threshold if threshold is None else threshold
        if float(score) < cutoff:
            return None
        return best

    def upsert_text(
        self,
        text: str,
        *,
        vector_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        namespace: str | None = None,
        check_similarity: bool = True,
    ) -> dict[str, Any]:
        """Эмбеддит текст и пишет вектор через upsert_vector."""
        return self.remember(
            text,
            vector_id=vector_id,
            metadata=metadata,
            namespace=namespace,
            check_similarity=check_similarity,
        )

    def remember(
        self,
        text: str,
        *,
        vector_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        namespace: str | None = None,
        threshold: float | None = None,
        on_duplicate: str | None = None,
        check_similarity: bool = True,
        force: bool = False,
    ) -> dict[str, Any]:
        """Пишет текст в долговременную память через upsert_vector."""
        text = text.strip()
        if not text:
            raise ValueError("Пустой текст нельзя записать в память.")
        payload = dict(metadata or {})
        payload.setdefault("text", text[:8000])
        return self.upsert_vector(
            self.embed(text),
            vector_id=vector_id,
            metadata=payload,
            namespace=namespace,
            check_similarity=False if force else check_similarity,
            threshold=threshold,
            on_duplicate=on_duplicate,
        )

    def upsert_vectors(
        self,
        vectors: Sequence[Mapping[str, Any] | tuple],
        *,
        namespace: str | None = None,
    ) -> int:
        """Записывает набор векторов: dict `{id, values, metadata?}` или tuple `(id, values, metadata?)`."""
        response = self.index.upsert(vectors=list(vectors), namespace=self._ns(namespace))
        return int(getattr(response, "upserted_count", 0) or 0)

    # --- запись документов ---

    def upsert_document(
        self,
        doc_id: str,
        fields: Mapping[str, Any],
        *,
        namespace: str | None = None,
    ) -> None:
        """Записывает один документ в schema-based индекс (`_id` + поля)."""
        document = {"_id": doc_id, **dict(fields)}
        self.upsert_documents([document], namespace=namespace)

    def upsert_documents(
        self,
        documents: Sequence[Mapping[str, Any]],
        *,
        namespace: str | None = None,
    ) -> int:
        """Записывает документы. Каждый должен содержать `_id`."""
        response = self.index.documents.upsert(
            namespace=self._ns(namespace),
            documents=list(documents),
        )
        return int(getattr(response, "upserted_count", 0) or 0)

    def upsert_records(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        namespace: str | None = None,
    ) -> int:
        """Записывает записи в индекс с интегрированными эмбеддингами (текст эмбеддится на стороне Pinecone)."""
        response = self.index.upsert_records(
            records=[dict(item) for item in records],
            namespace=self._ns(namespace),
        )
        return int(getattr(response, "record_count", 0) or 0)

    # --- чтение по вектору ---

    def query_by_vector(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 5,
        namespace: str | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
        include_metadata: bool = True,
        include_values: bool = False,
    ) -> list[dict[str, Any]]:
        """Ближайшие соседи по готовому вектору (классический dense-индекс)."""
        result = self.index.query(
            vector=list(vector),
            top_k=top_k,
            namespace=self._ns(namespace),
            filter=dict(metadata_filter) if metadata_filter else None,
            include_metadata=include_metadata,
            include_values=include_values,
        )
        matches: list[dict[str, Any]] = []
        for match in result.matches or []:
            item: dict[str, Any] = {
                "id": match.id,
                "score": match.score,
                "metadata": dict(match.metadata or {}),
            }
            if include_values:
                item["values"] = list(match.values or [])
            matches.append(item)
        return matches

    def search_documents_by_vector(
        self,
        vector: Sequence[float],
        *,
        field: str = "embedding",
        top_k: int = 5,
        namespace: str | None = None,
        include_fields: Sequence[str] | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Поиск документов по вектору (schema-based индекс)."""
        result = self.index.documents.search(
            namespace=self._ns(namespace),
            top_k=top_k,
            score_by=[DenseVectorQuery(field=field, values=list(vector))],
            include_fields=list(include_fields) if include_fields is not None else ["*"],
            filter=dict(metadata_filter) if metadata_filter else None,
        )
        return self._document_matches(result)

    # --- чтение по тексту ---

    def embed(self, text: str) -> list[float]:
        """Превращает текст в вектор через OpenAI (с учётом OPENAI_BASE_URL)."""
        text = text.strip()
        if not text:
            raise ValueError("Пустой текст нельзя эмбеддить.")
        client = self._require_openai()
        response = client.embeddings.create(model=self.openai_model, input=text)
        return response.data[0].embedding

    def _require_openai(self) -> OpenAI:
        if self.openai_client is None:
            raise RuntimeError(
                "Не задан OPENAI_API_KEY. Укажите ключ в параметрах или в файле .env."
            )
        return self.openai_client

    def query_by_text(
        self,
        text: str,
        embed_fn: EmbedFn | None = None,
        *,
        top_k: int = 5,
        namespace: str | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
        include_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        """Поиск по тексту: эмбеддинг через OpenAI внутри класса, затем query по вектору."""
        text = text.strip()
        if not text:
            return []
        vector = embed_fn(text) if embed_fn else self.embed(text)
        return self.query_by_vector(
            vector,
            top_k=top_k,
            namespace=namespace,
            metadata_filter=metadata_filter,
            include_metadata=include_metadata,
        )

    def search_by_text(
        self,
        text: str,
        *,
        top_k: int = 5,
        namespace: str | None = None,
        fields: Sequence[str] | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Поиск по тексту в индексе с интегрированными эмбеддингами (эмбеддинг на стороне Pinecone)."""
        text = text.strip()
        if not text:
            return []
        result = self.index.search(
            namespace=self._ns(namespace),
            top_k=top_k,
            inputs={"text": text},
            fields=list(fields) if fields is not None else None,
            filter=dict(metadata_filter) if metadata_filter else None,
        )
        hits = getattr(getattr(result, "result", None), "hits", None) or []
        matches: list[dict[str, Any]] = []
        for hit in hits:
            matches.append(
                {
                    "id": hit.id,
                    "score": hit.score,
                    "fields": dict(getattr(hit, "fields", None) or {}),
                }
            )
        return matches

    def search_documents_by_text(
        self,
        text: str,
        *,
        fields: Sequence[str],
        top_k: int = 5,
        namespace: str | None = None,
        include_fields: Sequence[str] | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Поиск документов по тексту (BM25 по указанным текстовым полям)."""
        text = text.strip()
        if not text:
            return []
        result = self.index.documents.search(
            namespace=self._ns(namespace),
            top_k=top_k,
            score_by=[TextQuery(query=text, fields=list(fields))],
            include_fields=list(include_fields) if include_fields is not None else ["*"],
            filter=dict(metadata_filter) if metadata_filter else None,
        )
        return self._document_matches(result)

    # --- точечное чтение и удаление ---

    def fetch(
        self,
        ids: Sequence[str],
        *,
        namespace: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Читает векторы по id, без ранжирования."""
        result = self.index.fetch(ids=list(ids), namespace=self._ns(namespace))
        found: dict[str, dict[str, Any]] = {}
        for vector_id, vector in (result.vectors or {}).items():
            found[vector_id] = {
                "id": vector_id,
                "values": list(getattr(vector, "values", None) or []),
                "metadata": dict(getattr(vector, "metadata", None) or {}),
            }
        return found

    def delete(
        self,
        *,
        ids: Sequence[str] | None = None,
        delete_all: bool = False,
        metadata_filter: Mapping[str, Any] | None = None,
        namespace: str | None = None,
    ) -> None:
        """Удаляет векторы по id, по фильтру или все записи в namespace."""
        kwargs: dict[str, Any] = {"namespace": self._ns(namespace)}
        if delete_all:
            kwargs["delete_all"] = True
        elif ids:
            kwargs["ids"] = list(ids)
        elif metadata_filter:
            kwargs["filter"] = dict(metadata_filter)
        else:
            raise ValueError("Укажите ids, metadata_filter или delete_all=True.")
        self.index.delete(**kwargs)

    @staticmethod
    def _document_matches(result: Any) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for doc in getattr(result, "matches", None) or []:
            item = {"id": getattr(doc, "id", None), "score": getattr(doc, "score", None)}
            extra = getattr(doc, "__dict__", {}) or {}
            for key, value in extra.items():
                if key not in {"id", "score"} and not key.startswith("_"):
                    item[key] = value
            matches.append(item)
        return matches


def _manual_test() -> None:
    """Проверка подключения к Pinecone и поиска — без Telegram."""
    manager = PineconeManager()
    manager.ensure_index()
    namespace = "manual_test"
    print(f"Индекс: {manager.index_name}")
    print("Записываю тестовый факт...")
    first = manager.upsert_text(
        "Пользователя зовут Анна, живёт в Москве.",
        metadata={"role": "user"},
        namespace=namespace,
    )
    print(f"  action={first['action']} id={first['id']}")

    print("Пишу похожую фразу (должен обновиться слот)...")
    second = manager.upsert_text(
        "Меня зовут Анна, я из Москвы.",
        metadata={"role": "user"},
        namespace=namespace,
    )
    print(
        f"  action={second['action']} "
        f"similarity={second.get('similarity_score')} "
        f"existing_id={second.get('existing_id')}"
    )

    print("Ищу по запросу: как зовут пользователя?")
    hits = manager.query_by_text("Как зовут пользователя?", top_k=3, namespace=namespace)
    if not hits:
        print("  совпадений нет")
    for hit in hits:
        text = (hit.get("metadata") or {}).get("text", "")
        print(f"  score={hit.get('score'):.3f}  {text}")

    manager.delete(delete_all=True, namespace=namespace)
    print("Тестовый namespace очищен. Менеджер работает.")


if __name__ == "__main__":
    _manual_test()
