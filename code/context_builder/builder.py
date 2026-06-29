# code/context_builder/builder.py
from context_builder.compressor import Compressor
from context_builder.layers import (
    ContextBundle,
    EvidenceItem,
    EvidenceLayer,
    MemoryItem,
    MemoryLayer,
    SystemLayer,
    TaskLayer,
    TraceLayer,
    TraceStep,
)


class ContextBuilder:
    """
    链式 API，组装 5 层 Context：

        bundle = (
            ContextBuilder(compressor)
            .set_system(role, instructions, tool_names)
            .set_task(description, goal, constraints)
            .add_memory(key, value)
            .add_evidence(source, content)
            .add_trace_step(step, tool, args, result)
            .build()
        )
    """

    def __init__(self, compressor: Compressor | None = None):
        self._compressor = compressor or Compressor()
        self._system: SystemLayer | None = None
        self._task: TaskLayer | None = None
        self._memory = MemoryLayer()
        self._evidence = EvidenceLayer()
        self._trace = TraceLayer()

    def set_system(
        self,
        role: str,
        instructions: list[str],
        tool_names: list[str] | None = None,
    ) -> "ContextBuilder":
        self._system = SystemLayer(
            role=role,
            instructions=instructions,
            tool_names=tool_names or [],
        )
        return self

    def set_task(
        self,
        description: str,
        goal: str,
        constraints: list[str] | None = None,
    ) -> "ContextBuilder":
        self._task = TaskLayer(
            description=description,
            goal=goal,
            constraints=constraints or [],
        )
        return self

    def add_memory(self, key: str, value: str) -> "ContextBuilder":
        self._memory.items.append(MemoryItem(key=key, value=value))
        return self

    def add_evidence(self, source: str, content: str) -> "ContextBuilder":
        summary, ref_id, char_count = self._compressor.compress(content, source)
        self._evidence.items.append(
            EvidenceItem(
                source=source,
                summary=summary,
                ref_id=ref_id,
                char_count=char_count,
            )
        )
        return self

    def add_trace_step(
        self, step: int, tool: str, args: dict, result: str
    ) -> "ContextBuilder":
        summary, ref_id, char_count = self._compressor.compress(result)
        self._trace.steps.append(
            TraceStep(
                step=step,
                tool=tool,
                args=args,
                result_summary=summary,
                result_ref=ref_id,
                result_char_count=char_count,
            )
        )
        return self

    def build(self) -> ContextBundle:
        if self._system is None:
            raise ValueError("system 层是必填的，请先调用 set_system()")
        if self._task is None:
            raise ValueError("task 层是必填的，请先调用 set_task()")
        return ContextBundle(
            system=self._system,
            task=self._task,
            memory=self._memory,
            evidence=self._evidence,
            trace=self._trace,
        )
