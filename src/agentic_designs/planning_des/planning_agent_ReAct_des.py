import json
import re

from colorama import Fore
from dotenv import load_dotenv
from groq import Groq

from agentic_designs.tool_des.tool import Tool
from agentic_designs.tool_des.tool import validate_arguments
from agentic_designs.utils.completions import build_prompt_structure
from agentic_designs.utils.completions import ChatHistory
from agentic_designs.utils.completions import completions_create
from agentic_designs.utils.completions import update_chat_history
from agentic_designs.utils.extraction import extract_tag_content

load_dotenv()

BASE_SYSTEM_PROMPT = ""


REACT_SYSTEM_PROMPT = """
You are a function-calling AI model. You operate by following a loop: Thought → Action → Observation.
Your task is to solve user queries by calling available mathematical tools as needed — step by step.
You are provided with function signatures within <tools></tools> XML tags.

You should never guess or invent function arguments. Instead, follow the 'types' strictly as valid Python dictionaries.

For each function call, return a JSON object inside <tool_call></tool_call> tags like this:

<tool_call>
{"name": <function-name>, "arguments": <args-dict>, "id": <monotonically-increasing-id>}
</tool_call>

Here are the available tools:

<tools>
%s
</tools>

Example session:

<question>What is the sum of 2345 and 6789, and then multiply the result by 7?</question>
<thought>I need to first add 2345 and 6789, then multiply the result by 7.</thought>
<tool_call>{"name": "sum_two_elements", "arguments": {"a": 2345, "b": 6789}, "id": 0}</tool_call>

You will then be called again with this:

<observation>{0: 9134}</observation>

Then:

<thought>I will now multiply the result 9134 by 7.</thought>
<tool_call>{"name": "multiply_two_elements", "arguments": {"a": 9134, "b": 7}, "id": 1}</tool_call>

You will be called again with this:

<observation>{1: 63938}</observation>

Then:

<response>The sum of 2345 and 6789 is 9134. Multiplying that by 7 gives 63938.</response>

Additional constraints:
- Only call functions that are explicitly defined in the <tools> section.
- If the question doesn’t require any function, reply with your answer enclosed in <response></response> tags.
"""



class ReactAgent:
    """
    A class that represents an agent using the ReAct logic that interacts with tools to process
    user inputs, make decisions, and execute tool calls. The agent can run interactive sessions,
    collect tool signatures, and process multiple tool calls in a given round of interaction.
    """

    def __init__(
        self,
        tools: Tool | list[Tool],
        model: str = "llama-3.3-70b-versatile",
        system_prompt: str = BASE_SYSTEM_PROMPT,
    ) -> None:
        self.client = Groq()
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools if isinstance(tools, list) else [tools]
        self.tools_dict = {tool.name: tool for tool in self.tools}

    def add_tool_signatures(self) -> str:
        # Collects the function signatures of all available tools.
        return "".join([tool.fn_signature for tool in self.tools])

    def process_tool_calls(self, tool_calls_content: list) -> dict:
        # Processes each tool call, validates arguments, executes the tools, and collects results.
        observations = {}
        for tool_call_str in tool_calls_content:
            tool_call = json.loads(tool_call_str)
            tool_name = tool_call["name"]
            tool = self.tools_dict[tool_name]

            print(Fore.GREEN + f"\nUsing Tool: {tool_name}")

            # Validate and execute the tool call
            validated_tool_call = validate_arguments(
                tool_call, json.loads(tool.fn_signature)
            )
            print(Fore.GREEN + f"\nTool call dict: \n{validated_tool_call}")

            result = tool.run(**validated_tool_call["arguments"])
            print(Fore.GREEN + f"\nTool result: \n{result}")

            # Store the result using the tool call ID
            observations[validated_tool_call["id"]] = result

        return observations

    def run(
        self,
        user_msg: str,
        max_rounds: int = 10,
    ) -> str:
        """
        Executes a user interaction session, where the agent processes user input, generates responses,
        handles tool calls, and updates chat history until a final response is ready or the maximum
        number of rounds is reached.
        """
        user_prompt = build_prompt_structure(
            prompt=user_msg, role="user", tag="question"
        )
        if self.tools:
            self.system_prompt += (
                "\n" + REACT_SYSTEM_PROMPT % self.add_tool_signatures()
            )

        chat_history = ChatHistory(
            [
                build_prompt_structure(
                    prompt=self.system_prompt,
                    role="system",
                ),
                user_prompt,
            ]
        )

        if self.tools:
            # Run the ReAct loop for max_rounds
            for _ in range(max_rounds):

                completion = completions_create(self.client, chat_history, self.model)

                response = extract_tag_content(str(completion), "response")
                if response.found:
                    return response.content[0]

                thought = extract_tag_content(str(completion), "thought")
                tool_calls = extract_tag_content(str(completion), "tool_call")

                update_chat_history(chat_history, completion, "assistant")

                print(Fore.MAGENTA + f"\nThought: {thought.content[0]}")

                if tool_calls.found:
                    observations = self.process_tool_calls(tool_calls.content)
                    print(Fore.BLUE + f"\nObservations: {observations}")
                    update_chat_history(chat_history, f"{observations}", "user")

        return completions_create(self.client, chat_history, self.model)
    