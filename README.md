# Multi-Agent-Assistant
This code defines a multi-agent system designed to efficiently answer questions by leveraging both direct responses and tool-based research.

## Project Structure

- `app.py`: This is the main script that contains the streamlit app.
- `assitant_graph.py`: This file contains langgraph code to create a multi-agent system.

### Agent Creation Functions

- `create_agent_1`: Creates an agent with access to tools, guided by a prompt template to provide quick and accurate answers using the tools available.
- `create_agent_2`: Creates an agent without tool access, designed to independently answer questions or collaborate with other agents if needed.
### State Management

- `AgentState`: A typed dictionary to manage the state of messages and the sender information.
### Helper Functions

- `agent_node`: Processes the output of an agent and appends it to the global state, handling both tool messages and regular AI messages.
### Routers

- `router_1` and `router_2`: Determine the next step in the workflow based on the current state, checking for tool invocation and final answers to decide the flow.
### Graph Creation

- `create_graph`: Defines the workflow graph for the multi-agent system.
- `Normal Agent`: An agent that answers questions independently.
- `Research Agent`: An agent with access to the TavilySearchResults tool for research-based answers.
- `Tool Node`: Manages tool invocations.
- `Workflow Node`: Manages the state and transitions between agents based on their responses and actions.
- `Workflow edges`: They are edges which are set to route between the agents and tools based on the conditions defined in the routers.
### Streaming Workflow Execution
- `create_graph_stream`: Initializes the workflow graph and starts a streaming execution based on the given query, returning events that represent the steps taken in the workflow.

## Requirements

Ensure you have the following Python packages installed:

- streamlit
- langchain
- langgraph


You can install the required packages using the following command:

```sh
pip install -r requirements.txt
``` 

## Langgraph Graph structure

![image](https://github.com/saurav-dhait/Multi-Agent-Assistant/blob/main/img/graph.png)

## Running the code
- To run the project, execute the `main.py` script:

```sh
streamlit run app.py
```

## Acknowledgements
This project is inspired by various tutorials and resources available for Multi-Agent Systems.