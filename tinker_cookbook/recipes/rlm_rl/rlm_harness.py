from __future__ import annotations

from tinker_cookbook.completers import MessageCompleter
from tinker_cookbook.recipes.rlm_rl.harness.repl import REPLHarness
from tinker_cookbook.recipes.rlm_rl.rlm.prompts import system_messages, turn_prompt
from tinker_cookbook.recipes.rlm_rl.rlm.tools import RLMTools, SubCallBudget
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.rl.types import Transition


class RLMHarness(REPLHarness):
    def __init__(
        self,
        *,
        context: str,
        root_prompt: str | None,
        max_iterations: int = 20,
        depth: int = 0,
        max_depth: int = 1,
        child_max_iterations: int = 8,
        max_sub_calls: int = 25,
        # Matches the per-prompt capacity the system prompt advertises to the model.
        max_subcall_chars: int = 100_000,
        budget: SubCallBudget | None = None,
        sub_completer: MessageCompleter | None = None,
        nudge_hint: bool = True,
        on_policy_llm_query: bool = False,
        mem_limit_bytes: int | None = None,
        compute_timeout_s: float | None = None,
    ):
        self.root_prompt = root_prompt
        self.depth = depth
        self.max_depth = max_depth
        self.child_max_iterations = child_max_iterations
        self.nudge_hint = nudge_hint
        self.on_policy_llm_query = on_policy_llm_query
        # True for a sub-agent that never gets to run code: a single-turn `llm_query`
        # standing in for an agent at the depth limit. It is graded on its answer alone,
        # since the process a REPL agent is judged on is not available to it.
        self.no_repl = False
        self.transitions: list[Transition] = []
        self._max_sub_calls = max_sub_calls
        self._max_subcall_chars = max_subcall_chars
        self._sub_completer = sub_completer
        self._budget = budget
        self.repl_tools = RLMTools(
            sub_completer=self._sub_completer,
            budget=self._budget if self._budget is not None else SubCallBudget(self._max_sub_calls),
            depth=self.depth,
            max_depth=self.max_depth,
            make_child=self._make_child,
            max_prompt_chars=self._max_subcall_chars,
            on_policy_llm_query=self.on_policy_llm_query,
        )
        repl_kwargs: dict[str, int | float] = {}
        if mem_limit_bytes is not None:
            repl_kwargs["mem_limit_bytes"] = mem_limit_bytes
        if compute_timeout_s is not None:
            repl_kwargs["compute_timeout_s"] = compute_timeout_s
        super().__init__(
            context=context,
            max_iterations=max_iterations,
            async_bindings=self.repl_tools.bindings,
            **repl_kwargs,
        )

    @property
    def tools(self) -> RLMTools:
        return self.repl_tools

    @property
    def children(self) -> list[RLMHarness]:
        """Sub-agents this agent spawned, in the order they completed."""
        return self.repl_tools.child_agents

    def _make_child(self, prompt: str, context: str | None = None) -> RLMHarness:
        if context is None:
            child_context, child_prompt = prompt, None
        else:
            child_context, child_prompt = context, prompt
        return RLMHarness(
            context=child_context,
            root_prompt=child_prompt,
            max_iterations=self.child_max_iterations,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            child_max_iterations=self.child_max_iterations,
            max_sub_calls=self.tools.budget.limit,
            max_subcall_chars=self._max_subcall_chars,
            budget=self.tools.budget,
            sub_completer=self._sub_completer,
            nudge_hint=self.nudge_hint,
            on_policy_llm_query=self.on_policy_llm_query,
            mem_limit_bytes=self.mem_limit_bytes,
            compute_timeout_s=self.compute_timeout_s,
        )

    def _initial_messages(self) -> list[Message]:
        return system_messages(
            context_chars=len(self.context),
            context_type=type(self.context).__name__,
            root_prompt=self.root_prompt,
            max_sub_calls=self.tools.budget.limit,
            nudge_hint=self.nudge_hint,
        )

    def _turn_message(self, iteration: int) -> Message:
        return turn_prompt(iteration, self.max_iterations)

    def _metrics(self) -> dict[str, float]:
        return {**super()._metrics(), **self.tools.metrics()}
