from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
import functools
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode
from typing import Literal
import operator
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

load_dotenv()


def create_agent_1(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful AI assistant, collaborating with other AI assistants.
                Other AI assistants will ask you for help. 
                However, there are some rules and regulations you must follow while giving your answer. 
                The rules are as follows:
                1. Try to find the answer quickly without wasting too much time.
                2. Using tools will help you find the answer.
                3. You have access to the following tools: {tool_names}.
                4. make sure that you provide the response as quickly and accurately as possible.
                {system_message}
                """,

            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)


def create_agent_2(llm, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful AI assistant, collaborating with other AI assistants. 
                When the human asks a question, follow these rules to provide your answer :
                The rules are as follows:
                1. Try to answer the question on your own first. If you find the answer, then prefix your response 
                with "FINAL ANSWER". It is very important to follow this rule.
                2. if the human is trying to greet you or get along with you, prefix you response with FINAL ANSWER.
                3. If you do not know the answer, ask the other AI assistants for help. Do not prefix your 
                response with "FINAL ANSWER" when asking other AI assistants for help.They will provide you with the 
                necessary information.
                4. Forward any questions about current events to other assistants.They will provide you with 
                necessary information.
                5. Do not ask other AI assistants for help more than 2 times. Just prefix your response 
                with FINAL ANSWER.
                5. Once you think you have the final answer, prefix your response with "FINAL ANSWER."
                {system_message}
                """
                ,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)

    return prompt | llm


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


# Helper function to create a node for a given agent
def agent_node(state, agent, name):
    result = agent.invoke(state)

    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name,
    }


# Either agent can decide to end


def router_1(state):
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        # The previous agent is invoking a tool
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return "__end__"
    return "continue"


def router_2(state):
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return "__end__"
    return "continue"


def create_graph():
    llm = ChatGroq(
        temperature=0,
        model="llama3-70b-8192",
    )

    # Normal agent and node
    normal_agent = create_agent_2(
        llm,
        system_message=" ",
    )
    normal_node = functools.partial(agent_node, agent=normal_agent, name="Normal")

    # Research agent and node
    research_agent = create_agent_1(
        llm,
        [TavilySearchResults(max_results=5)],
        system_message="You should provide accurate data for use.",
    )
    research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

    # Tool Node
    tools = [TavilySearchResults(max_results=2)]
    tool_node = ToolNode(tools)

    # Workflow Node
    workflow = StateGraph(AgentState)
    workflow.add_node("Researcher", research_node)
    workflow.add_node("Normal", normal_node)

    workflow.add_node("call_tool", tool_node)

    # Workflow Edges
    workflow.add_conditional_edges(
        "Normal",
        router_2,
        {"continue": "Researcher", "__end__": END},
    )
    workflow.add_conditional_edges(
        "Researcher",
        router_1,
        {"continue": "Normal", "call_tool": "call_tool", "__end__": END},
    )

    workflow.add_conditional_edges(
        "call_tool",
        # Each agent node updates the 'sender' field
        # the tool calling node does not, meaning
        # this edge will route back to the original agent
        # who invoked the tool
        lambda x: x["sender"],
        {
            "Researcher": "Researcher",
        },
    )
    workflow.set_entry_point("Normal")
    graph = workflow.compile()
    return graph


def create_graph_stream(query):
    graph = create_graph()
    events = graph.stream(
        {
            "messages": [
                HumanMessage(
                    content=query
                )
            ],
        },
        # Maximum number of steps to take in the graph
        {"recursion_limit": 150},
    )
    return events
