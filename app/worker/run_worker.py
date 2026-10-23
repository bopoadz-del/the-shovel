from arq.worker import run_worker

from app.worker.ingest_worker import WorkerSettings

if __name__ == "__main__":
    run_worker(WorkerSettings)
