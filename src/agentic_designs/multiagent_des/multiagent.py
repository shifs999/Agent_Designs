from textwrap import dedent

from agentic_designs.multiagent_des.crew_sys import Crew
from agentic_designs.planning_des.planning_agent_ReAct_des import ReactAgent
from agentic_designs.tool_des.tool import Tool

class Agent:
    """
    This class implements an agent with dependencies, context handling, and task execution capabilities.

    Attributes:
        name (str): The name of the agent.
        agent_desc (str): The description of the agent's role.
        task_description (str): A description of the task assigned to the agent.
        task_expected_output (str): The expected format or content of the task output.
        react_agent (ReactAgent): An instance of ReactAgent used for generating responses.
        context (str): Accumulated context information from other agents.
    """

    def __init__(
        self,
        name: str,
        agent_desc: str,
        task_description: str,
        task_expected_output: str = "",
        tools: list[Tool] | None = None,
        llm: str = "llama-3.3-70b-versatile",
    ):
        self.name = name
        self.agent_desc = agent_desc
        self.task_description = task_description
        self.task_expected_output = task_expected_output
        self.react_agent = ReactAgent(
            model=llm, system_prompt=self.agent_desc, tools=tools or []
        )

        self.dependencies: list[Agent] = []  # Agents that this agent depends on
        self.dependents: list[Agent] = []  # Agents that depend on this agent

        self.context = ""

        # Automatically register this agent to the active Crew context if one exists
        Crew.register_agent(self)

    def __repr__(self):
        return f"{self.name}"

    def __rshift__(self, other):
        # Defines the '>>' operator to indicate agent dependency.
        self.add_dependent(other)
        return other  # Allow chaining

    def __lshift__(self, other):
        # Defines the '<<' operator to indicate agent dependency in reverse.
        self.add_dependency(other)
        return other  # Allow chaining

    def __rrshift__(self, other):
        # Defines the '<<' operator to indicate agent dependency.
        self.add_dependency(other)
        return self  # Allow chaining

    def __rlshift__(self, other):
        # Defines the '<<' operator when evaluated from right to left. This operator is used to indicate agent dependency in the normal order.
        self.add_dependent(other)
        return self  # Allow chaining

    def add_dependency(self, other):
        # Adds a dependency to this agent.
        if isinstance(other, Agent):
            self.dependencies.append(other)
            other.dependents.append(self)
        elif isinstance(other, list) and all(isinstance(item, Agent) for item in other):
            for item in other:
                self.dependencies.append(item)
                item.dependents.append(self)
        else:
            raise TypeError("The dependency must be an instance or list of Agent.")

    def add_dependent(self, other):
        # Adds a dependent to this agent.
        if isinstance(other, Agent):
            other.dependencies.append(self)
            self.dependents.append(other)
        elif isinstance(other, list) and all(isinstance(item, Agent) for item in other):
            for item in other:
                item.dependencies.append(self)
                self.dependents.append(item)
        else:
            raise TypeError("The dependent must be an instance or list of Agent.")

    def receive_context(self, input_data):
        
        self.context += f"{self.name} received the context as: \n\n{input_data}"

    def create_prompt(self):
       
        prompt = dedent(
            f"""
        You are an AI agent. You are part of a team of agents working together to complete a task.
        I'm going to give you the task description enclosed in <task_description></task_description> tags. I'll also give
        you the available context from the other agents in <context></context> tags. If the context
        is not available, the <context></context> tags will be empty. You'll also receive the task
        expected output enclosed in <task_expected_output></task_expected_output> tags. With all this information
        you need to create the best possible response, always respecting the format as describe in
        <task_expected_output></task_expected_output> tags. If expected output is not available, just create
        a meaningful response to complete the task.

        <task_description>
        {self.task_description}
        </task_description>

        <task_expected_output>
        {self.task_expected_output}
        </task_expected_output>

        <context>
        {self.context}
        </context>

        Your response:
        """
        ).strip()

        return prompt

    def run(self):
        
        msg = self.create_prompt()
        output = self.react_agent.run(user_msg=msg)

        for dependent in self.dependents:
            dependent.receive_context(output)
        return output
    