"""Ingest worker entry.

The production startup guard is ``app.main._validate_startup_env`` — the
same function the web process runs — so a worker cannot boot into a
configuration the web process would refuse. That includes the document
byte-store check.
"""

from arq.worker import run_worker

from app.worker.ingest_worker import WorkerSettings


def main() -> None:
    from app.main import _validate_startup_env

    _validate_startup_env()
    run_worker(WorkerSettings)


if __name__ == "__main__":
    main()
