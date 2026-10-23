"""An OOM kill leaves no traceback. The watchdog must name the code that is
allocating while there is still memory left to log it (live 21 Sep 2026: RSS
773 MB -> 2 GiB OOM kill in ~40 s after the final LLM call, no log line)."""
import threading

from app.core import rss_watchdog as rw


def _dog(samples, threshold=1200, step=300):
    logs = []
    it = iter(samples)
    dog = rw.RssWatchdog(threshold, step, read_rss=lambda: next(it), log=logs.append)
    return dog, logs


def test_crossing_the_threshold_logs_the_allocating_threads_stack():
    ready, release = threading.Event(), threading.Event()

    def _synthetic_allocator_under_test():
        ready.set()
        release.wait(5)

    t = threading.Thread(target=_synthetic_allocator_under_test, name="worker-x")
    t.start()
    ready.wait(5)
    try:
        dog, logs = _dog([800, 1250])
        assert dog.check() is False and logs == []
        assert dog.check() is True
    finally:
        release.set()
        t.join()
    assert len(logs) == 1
    assert "rss=1250MB" in logs[0]
    assert "worker-x" in logs[0] and "_synthetic_allocator_under_test" in logs[0]


def test_it_logs_once_per_step_not_every_sample():
    dog, logs = _dog([1250, 1300, 1400, 1510, 1600, 1850])
    fired = [dog.check() for _ in range(6)]
    # 1200 mark at 1250; next 1500 at 1510; next 1800 at 1850.
    assert fired == [True, False, False, True, False, True]
    assert len(logs) == 3


def test_it_rearms_after_memory_falls_back():
    dog, logs = _dog([1250, 700, 1250])
    assert [dog.check() for _ in range(3)] == [True, False, True]


def test_stacks_carry_locations_never_local_values():
    secret = "SYNTHETIC-SECRET-VALUE-42"
    ready, release = threading.Event(), threading.Event()

    def holder(value):
        ready.set()
        release.wait(5)
        return value

    t = threading.Thread(target=holder, args=(secret,))
    t.start()
    ready.wait(5)
    try:
        text = rw.format_thread_stacks()
    finally:
        release.set()
        t.join()
    assert "in holder" in text
    assert secret not in text


def test_unreadable_rss_never_fires():
    dog, logs = _dog([None, None])
    assert dog.check() is False and logs == []


def test_disabled_by_zero(monkeypatch):
    monkeypatch.setenv("RSS_WATCHDOG_MB", "0")
    monkeypatch.setattr(rw, "_started", None)
    assert rw.start() is None
