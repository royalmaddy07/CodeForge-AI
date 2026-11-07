from langgraph.graph import MessagesState
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langgraph.checkpoint.memory import InMemorySaver
from chatbot import llm
from prompts import programming_prompt

def chatbot(state: MessagesState):
    # here state -> variable name and : MessagesState -> TypeHint, ye bhi hota hai python me
    # the state is a dictionary which contains -> "messages" : [....] -> we extract that and give it to 

    # 1. format prompt using conversation history
    formattedPrompt = programming_prompt.invoke({
        "history" : state["messages"]
    })

    # 2. call our llm/gemini returns us response = AIMessage object
    response = llm.invoke(formattedPrompt)

    # we update that to our state and return it to our graph
    return {
        "messages" : [response]
    }

graph_builder = StateGraph(MessagesState)
graph_builder.add_node("chatbot", chatbot)
# we told langgraph that there is a node whose name is chatbot & when you execute it, call the chatbot() function
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

checkpointer = InMemorySaver()
# while the python process is running the memory will be saved and when the process ends, the memory is erased
# Later we'll move to a persistent checkpointer backed by something like a database.

graph = graph_builder.compile(
    checkpointer=checkpointer # our graph now has persistence enabled
)# this creates/compiles our graph object

# thread_id: similar concept like session_id, each thread_id belongs to a user  

# important point to notice : response = chain.invoke() gave us the AIMessage or parsed string
# Now, result = graph.invoke() returns the state
# LangGraph is about running a workflow and updating the state
