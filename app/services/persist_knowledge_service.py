# migrate_zilliz_collection.py
import json, os
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
import dotenv

dotenv.load_dotenv()
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")

LOCAL_HOST = "standalone"
LOCAL_PORT = 19530

def export_collection(name, batch_size=1000):
    connections.connect(alias="src", uri=MILVUS_URI, token=MILVUS_TOKEN)
    src = Collection(name, using="src")
    src.load()
    total = src.num_entities
    print(f"Exporting {total} entities from '{name}'...")
    offset = 0
    results = []
    while offset < total:
        cnt = min(batch_size, total - offset)
        batch = src.query(expr="", offset=offset, limit=cnt, output_fields=["*"])
        results.extend(batch)
        offset += len(batch)
        print(f"Exported {offset}/{total}")
    return results

def import_to_local(name, records, embedding_dim):
    # Connect to local Milvus (could be Docker container)
    milvus_host = LOCAL_HOST
    milvus_port = LOCAL_PORT
    connections.connect(alias="default", host=milvus_host, port=milvus_port, timeout=30)
    if utility.has_collection(name, using="default"):
        print(f"Dropping existing local collection '{name}'")
        Collection(name, using="default").drop()
    fields = [
        FieldSchema("id", DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema("document", DataType.VARCHAR, max_length=65535),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=embedding_dim),
        FieldSchema("file_name", DataType.VARCHAR, max_length=500),
        FieldSchema("file_path", DataType.VARCHAR, max_length=1000),
        FieldSchema("file_type", DataType.VARCHAR, max_length=50),
        FieldSchema("source_link", DataType.VARCHAR, max_length=2000),
        FieldSchema("chunk_index", DataType.INT64),
        FieldSchema("language", DataType.VARCHAR, max_length=50),
        FieldSchema("has_code", DataType.BOOL),
        FieldSchema("repo_name", DataType.VARCHAR, max_length=200),
        FieldSchema("content_quality_score", DataType.FLOAT),
        FieldSchema("semantic_density_score", DataType.FLOAT),
        FieldSchema("information_value_score", DataType.FLOAT),
    ]
    schema = CollectionSchema(fields, "Enhanced repository content with semantic chunking and image metadata")
    default = Collection(name, schema=schema, using="default")
    default.create_index("embedding", index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
    default.load()
    print(f"Inserting {len(records)} records into local '{name}'...")
    # Build columnar data for Milvus
    entities = [
        [rec.get("id") for rec in records],
        [rec.get("document") for rec in records],
        [rec.get("embedding") for rec in records],
        [rec.get("file_name") for rec in records],
        [rec.get("file_path") for rec in records],
        [rec.get("file_type") for rec in records],
        [rec.get("source_link") for rec in records],
        [rec.get("chunk_index") for rec in records],
        [rec.get("language") for rec in records],
        [rec.get("has_code") for rec in records],
        [rec.get("repo_name") for rec in records],
        [rec.get("content_quality_score") for rec in records],
        [rec.get("semantic_density_score") for rec in records],
        [rec.get("information_value_score") for rec in records],
    ]
    default.insert(entities)
    print("Import complete.")

if __name__ == "__main__":
    COLLECTION = "beaglemind_col"
    exported = export_collection(COLLECTION)
    if exported:
        emb = exported[0].get("embedding")
        if not isinstance(emb, list):
            raise ValueError("Embedding field not found or invalid")
        embedding_dim = len(emb)
        import_to_local(COLLECTION, exported, embedding_dim)
        print("Migration finished successfully.")
    else:
        print("No records found to migrate.")
