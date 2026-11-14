"""Deterministic CPU event scheduling; times are modeled, not GPU measurements."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
from typing import Callable, Iterable


@dataclass(eq=False)
class Event:
    """A scheduled operation on a modeled serial stream."""

    id: int
    name: str
    stream: str
    start_ms: float
    end_ms: float
    metadata: dict
    completed: bool = False
    resource: str | None = None
    _timeline: object = field(default=None, repr=False)
    _action: Callable[[], None] | None = field(default=None, repr=False)
    _dependencies: tuple[Event, ...] = field(default=(), repr=False)
    _error: BaseException | None = field(default=None, repr=False)

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


class Timeline:
    """Serialize each stream and optional shared resource; overlap other work.

    ``submit`` schedules an interval immediately. Its action executes exactly once
    at interval completion during ``run_until``. Explicit dependency Events must
    come from this Timeline. A resource models an exclusive link or engine, not
    a fluid bandwidth-sharing model: two operations naming that resource queue.
    """

    def __init__(self) -> None:
        self.now_ms = 0.0
        self._events: list[Event] = []
        self._pending: list[tuple[float, int, Event]] = []
        self._streams: dict[str, Event] = {}
        self._resources: dict[str, Event] = {}
        self._failure: BaseException | None = None

    @property
    def total_ms(self) -> float:
        return max((event.end_ms for event in self._events), default=0.0)

    def submit(
        self,
        name: str,
        duration_ms: float,
        stream: str,
        wait_for: Iterable[Event] = (),
        action: Callable[[], None] | None = None,
        metadata: dict | None = None,
        resource: str | None = None,
    ) -> Event:
        if self._failure is not None:
            raise RuntimeError("timeline previously failed") from self._failure
        duration = float(duration_ms)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("duration_ms must be finite and nonnegative")
        if not isinstance(name, str) or not name:
            raise ValueError("event name must be a nonempty string")
        if not isinstance(stream, str) or not stream:
            raise ValueError("stream must be a nonempty string")
        if resource is not None and (not isinstance(resource, str) or not resource):
            raise ValueError("resource must be None or a nonempty string")
        if action is not None and not callable(action):
            raise TypeError("action must be callable")
        dependencies = list(wait_for)
        for event in dependencies:
            self._validate_event(event)
        if stream in self._streams:
            dependencies.append(self._streams[stream])
        if resource is not None and resource in self._resources:
            dependencies.append(self._resources[resource])
        dependencies = list(dict.fromkeys(dependencies))
        start = max(self.now_ms, *(event.end_ms for event in dependencies)) if dependencies else self.now_ms
        end = start + duration
        if not math.isfinite(end):
            raise ValueError("scheduled end time must be finite")
        event = Event(
            id=len(self._events), name=name, stream=stream,
            start_ms=start, end_ms=end, metadata=dict(metadata or {}),
            resource=resource, _timeline=self, _action=action,
            _dependencies=tuple(dependencies),
        )
        self._events.append(event)
        self._streams[stream] = event
        if resource is not None:
            self._resources[resource] = event
        heapq.heappush(self._pending, (end, event.id, event))
        return event

    def _validate_event(self, event: Event) -> None:
        if not isinstance(event, Event) or event._timeline is not self:
            raise ValueError("event belongs to another timeline or was not scheduled")
        if event.id < 0 or event.id >= len(self._events) or self._events[event.id] is not event:
            raise ValueError("event was not scheduled by this timeline")

    def run_until(self, event: Event | None = None) -> None:
        """Complete queued operations through ``event`` (or drain the queue).

        Operations with the same completion time execute in submission order,
        which also puts predecessors before zero-duration dependent operations.
        After a callback failure the timeline is unusable, avoiding silent
        continuation with partially materialized tensors.
        """
        if event is not None:
            self._validate_event(event)
        if self._failure is not None:
            raise RuntimeError("timeline previously failed") from self._failure
        if event is not None and event.completed:
            return
        while self._pending:
            _, _, current = heapq.heappop(self._pending)
            self.now_ms = current.end_ms
            try:
                if any(not dependency.completed for dependency in current._dependencies):
                    raise RuntimeError("event executed before its dependencies completed")
                if current._action is not None:
                    current._action()
                current.completed = True
            except BaseException as exc:
                current._error = exc
                self._failure = exc
                raise
            if current is event:
                return

    def trace(self) -> list[dict]:
        """Return JSON-compatible event records in submission order."""
        return [
            {
                "id": event.id,
                "name": event.name,
                "stream": event.stream,
                "resource": event.resource,
                "start_ms": event.start_ms,
                "end_ms": event.end_ms,
                "duration_ms": event.duration_ms,
                "completed": event.completed,
                "dependencies": [dependency.id for dependency in event._dependencies],
                "metadata": dict(event.metadata),
                "error": str(event._error) if event._error is not None else None,
            }
            for event in self._events
        ]
