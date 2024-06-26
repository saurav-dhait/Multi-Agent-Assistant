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

def create_agent(llm, tools, system_message: str):
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
                The human will ask a question, and you have to answer it. 
                However, there are some rules and regulations you must follow while giving your answer. 
                The rules are as follows:
                1. Try to find the answer to question on your own.
                2. if the question looks like a greeting prefix your response with FINAL ANSWER so the 
                other assistants know when to stop.
                3. If you find the answer, prefix your response with FINAL ANSWER so the other assistants know when to stop.
                4. If you cannot find the answer on your own, that's Ok, The other assistants will provide you with 
                necessary information. Use that information and find the answer and then prefix your
                response with FINAL ANSWER so that the other assistants know when to stop.
                5. Do not prefix your response with FINAL ANSWER if you are asking other assistants for help.
                6. If you are unable to find the answer even after getting information from other AI assistants, then
                prefix your response with FINAL ANSWER so that the other assistants know when to stop.
                
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


def router(state) -> Literal["call_tool", "__end__", "continue"]:
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
    research_agent = create_agent(
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
        router,
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
