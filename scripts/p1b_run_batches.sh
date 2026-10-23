#!/usr/bin/env bash
set -e

# Run Master Folder ingestion in 1,000-file batches until no unindexed files remain.
# This is resilient to crashes because each invocation uses --resume, while the
# larger batch size avoids reloading the embedding model every 100 files.

export RAG_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
export RAG_VECTOR_NAMESPACE=v2
export SAFETY_WORLD_WEIGHTS=data/models/safety_world_v2.onnx

batch_num=0
while true; do
    out="manifests/p1b_master_folder_batch_$(printf '%03d' $batch_num).json"
    echo "=== Starting batch $batch_num -> $out ==="
    .venv/Scripts/python scripts/p1b_ingest_local_folder.py \
        --folder "Master Folder" \
        --output "$out" \
        --resume \
        --limit 1000
    
    # Stop when the batch processed 0 files (all done).
    count=$(.venv/Scripts/python -c "import json; print(len(json.load(open('$out')).get('results', [])))")
    echo "=== Batch $batch_num processed $count files ==="
    if [ "$count" -eq 0 ]; then
        echo "=== No more files to ingest ==="
        break
    fi
    batch_num=$((batch_num + 1))
done
