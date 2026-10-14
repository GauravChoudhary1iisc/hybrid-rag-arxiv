"""Step 1a: embed the 28,711 fixed chunks with each model and index into NEW Qdrant
collections exp_<key>. Existing collections are never touched."""
import subprocess
import time

import numpy as np
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from common import MODELS, BATCH_SIZE, OUT, load_chunks, seed_all, save_json

UPLOAD_BATCH = 256
PROTECTED = {"rag_fixed", "rag_semantic"}

seed_all()
chunks = load_chunks()
q = QdrantClient(host="localhost", port=6333, timeout=120)
gpu = torch.cuda.get_device_properties(0)
results = {"gpu": gpu.name, "gpu_vram_gb": round(gpu.total_memory / 1024**3, 2),
           "torch": torch.__version__, "batch_size": BATCH_SIZE, "models": {}}
print(f"GPU: {gpu.name}  VRAM {results['gpu_vram_gb']} GB  | {len(chunks):,} chunks")

for key, cfg in MODELS.items():
    coll = f"exp_{key}"
    assert coll not in PROTECTED
    print(f"\n== {key}: {cfg['name']} -> {coll}")
    model = SentenceTransformer(cfg["name"], device="cuda")
    dim = model.get_sentence_embedding_dimension()
    print(f"   dim={dim} max_seq_length={model.max_seq_length}")
    texts = [cfg["passage_prefix"] + c["text"] for c in chunks]

    torch.cuda.synchronize()
    t0 = time.time()
    emb = model.encode(texts, batch_size=BATCH_SIZE, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False)
    torch.cuda.synchronize()
    enc_s = time.time() - t0
    emb = emb.astype(np.float32)
    print(f"   encoded {emb.shape} in {enc_s:.1f}s = {len(texts)/enc_s:.1f} chunks/s; "
          f"norm mean {np.linalg.norm(emb, axis=1).mean():.4f}")
    np.save(OUT / f"emb_{key}.npy", emb)

    if coll in [c.name for c in q.get_collections().collections]:
        print(f"   {coll} exists from an earlier attempt of this experiment -> recreating")
        q.delete_collection(coll)
    q.create_collection(coll, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
    t0 = time.time()
    for s in range(0, len(chunks), UPLOAD_BATCH):
        e = min(s + UPLOAD_BATCH, len(chunks))
        q.upsert(coll, points=[PointStruct(
            id=chunks[i]["chunk_id"], vector=emb[i].tolist(),
            payload={k: chunks[i][k] for k in ("arxiv_id", "title", "chunk_index", "text", "char_count")})
            for i in range(s, e)])
    up_s = time.time() - t0
    for _ in range(120):  # wait for indexing to finish
        ci = q.get_collection(coll)
        if str(ci.status).lower().endswith("green"):
            break
        time.sleep(5)
    disk = subprocess.run(["du", "-sb", f"../qdrant_storage/collections/{coll}"],
                          capture_output=True, text=True)
    disk_bytes = int(disk.stdout.split()[0]) if disk.returncode == 0 else None
    print(f"   upload {up_s:.1f}s; points={ci.points_count} status={ci.status} "
          f"disk={disk_bytes if disk_bytes is not None else 'unreadable: ' + disk.stderr.strip()}")
    results["models"][key] = {
        "model": cfg["name"], "collection": coll, "dim": dim,
        "max_seq_length": model.max_seq_length,
        "passage_prefix": cfg["passage_prefix"], "query_prefix": cfg["query_prefix"],
        "encode_seconds": round(enc_s, 2), "chunks_per_second": round(len(texts) / enc_s, 1),
        "upload_seconds": round(up_s, 2), "points_count": ci.points_count,
        "raw_vector_mb": round(emb.nbytes / 1024**2, 1),
        "qdrant_disk_mb": round(disk_bytes / 1024**2, 1) if disk_bytes else None,
    }
    del model
    torch.cuda.empty_cache()

save_json(results, OUT / "02_embed_index.json")
print("\nsaved out/02_embed_index.json")
