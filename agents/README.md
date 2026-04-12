# kairo-agents

Create agents, wire inter-agent communication, delegate to subagents, and
observe reasoning via verbose mode.

```python
from kairo_agents import Agent, MessageBus

bus = MessageBus(verbose=True)
planner = Agent("planner", role="task planner",
                instruction="Break goals into steps.", verbose=True, bus=bus)
writer = Agent("writer", role="writer",
               instruction="Draft prose.", verbose=True)
planner.add_subagent(writer)

planner.think("Plan a short article.")
planner.delegate("writer", "Draft an opening line.")
planner.send("researcher", "Need 3 citations.")
```

Plug in an LLM by passing `think_fn=(role, instruction, input) -> str`.

Install:

```bash
pip install -e ./core ./agents
```
