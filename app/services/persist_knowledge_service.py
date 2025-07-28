# migrate_zilliz_collection.py
import subprocess, time, json
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from config import MILVUS_URI, MILVUS_TOKEN

CONTAINER_NAME = "milvus"
IMAGE_NAME = "milvusdb/milvus:latest"
LOCAL_HOST = "localhost"
LOCAL_PORT = "19530"

def ensure_milvus_docker():
    res = subprocess.run(["docker", "inspect", CONTAINER_NAME],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if res.returncode == 0:
        state = subprocess.run(["docker", "inspect", "-f", "{{.State.Status}}", CONTAINER_NAME],
                               capture_output=True, text=True).stdout.strip()
        if state != "running":
            print("Starting existing Milvus container...")
            subprocess.run(["docker", "start", CONTAINER_NAME], check=True)
        else:
            print("Milvus container is already running.")
    else:
        print("Pulling and running Milvus container...")
        subprocess.run(["docker", "pull", IMAGE_NAME], check=True)
        subprocess.run([
            "docker", "run", "-d",
            "--name", CONTAINER_NAME,
            "-p", f"{LOCAL_PORT}:19530",
            IMAGE_NAME
        ], check=True)
    print("Waiting for local Milvus to be ready...")
    time.sleep(10)

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
    connections.connect(alias="dest", host=LOCAL_HOST, port=LOCAL_PORT)
    if utility.has_collection(name, using="dest"):
        print(f"Dropping existing local collection '{name}'")
        Collection(name, using="dest").drop()
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
    schema = CollectionSchema(fields, description="Migrated collection")
    dest = Collection(name, schema=schema, using="dest")
    dest.create_index("embedding", index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
    dest.load()
    print(f"Inserting {len(records)} records into local '{name}'...")
    rows = []
    for rec in records:
        rows.append([
            rec.get("id"),
            rec.get("document"),
            rec.get("embedding"),
            rec.get("file_name"),
            rec.get("file_path"),
            rec.get("file_type"),
            rec.get("source_link"),
            rec.get("chunk_index"),
            rec.get("language"),
            rec.get("has_code"),
            rec.get("repo_name"),
            rec.get("content_quality_score"),
            rec.get("semantic_density_score"),
            rec.get("information_value_score"),
        ])
    dest.insert(rows)
    print("Import complete.")

if __name__ == "__main__":
    COLLECTION = "beaglemind_col"
    exported = export_collection(COLLECTION)
    if exported:
        emb = exported[0].get("embedding")
        if not isinstance(emb, list):
            raise ValueError("Embedding field not found or invalid")
        embedding_dim = len(emb)
        ensure_milvus_docker()
        import_to_local(COLLECTION, exported, embedding_dim)
        print("Migration finished successfully.")
    else:
        print("No records found to migrate.")
