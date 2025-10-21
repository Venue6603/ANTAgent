from typing import Dict, List, Set
from AntAgent.models import TimelineSpec, Timeline, Task

def _toposort(tasks: List[Task]) -> List[str]:
    graph: Dict[str, Set[str]] = {t.name: set(t.depends_on) for t in tasks}
    ordered: List[str] = []
    while graph:
        ready = [n for n, deps in graph.items() if not deps]
        if not ready:  # cycle fallback
            ordered.extend(list(graph.keys()))
            break
        ordered.extend(ready)
        for r in ready:
            graph.pop(r, None)
        for deps in graph.values():
            deps.difference_update(ready)
    return ordered

def build_timeline(spec: TimelineSpec) -> Timeline:
    order = _toposort(spec.tasks)
    # naive critical path == topological order here
    return Timeline(tasks=spec.tasks, critical_path=order)
