from __future__ import annotations

import json
import logging
import queue
import threading
import time
import urllib.request
from collections.abc import Callable
from typing import Any

from .events import EventRecord, EventRepository, WebhookOptions


LOGGER = logging.getLogger(__name__)
WebhookSender = Callable[[str, dict[str, Any], float], None]


def send_webhook(
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> None:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        if not 200 <= int(response.status) < 300:
            raise RuntimeError(f"Webhook返回HTTP {response.status}")


class WebhookDispatcher:
    """Bounded delivery queue that never blocks the video metadata probe."""

    def __init__(
        self,
        options: WebhookOptions,
        *,
        sender: WebhookSender = send_webhook,
        max_pending: int = 100,
        repository: EventRepository | None = None,
    ) -> None:
        options.validate()
        self._options = options
        self._sender = sender
        self._repository = repository
        self._queue: queue.Queue[EventRecord | None] = queue.Queue(
            maxsize=max_pending
        )
        self._thread: threading.Thread | None = None
        if options.url is not None or repository is not None:
            self._thread = threading.Thread(
                target=self._run,
                name="event-webhook",
                daemon=True,
            )
            self._thread.start()

    def enqueue(self, event: EventRecord) -> bool:
        if self._thread is None:
            return False
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            LOGGER.error("Webhook队列已满，丢弃事件: %s", event.event_id)
            return False
        return True

    def shutdown(self, timeout_seconds: float = 2.0) -> None:
        thread = self._thread
        if thread is None:
            return
        timeout = max(timeout_seconds, 0.0)
        deadline = time.monotonic() + timeout
        try:
            self._queue.put(None, timeout=timeout)
        except queue.Full:
            LOGGER.warning("事件交付队列关闭超时")
        thread.join(timeout=max(deadline - time.monotonic(), 0.0))
        self._thread = None

    def _run(self) -> None:
        while True:
            event = self._queue.get()
            if event is None:
                return
            try:
                if self._repository is not None:
                    self._repository.append(event)
                if self._options.url is not None:
                    self._sender(
                        self._options.url,
                        event.to_dict(),
                        self._options.timeout_seconds,
                    )
            except Exception:
                LOGGER.exception("事件异步交付失败: event=%s", event.event_id)
