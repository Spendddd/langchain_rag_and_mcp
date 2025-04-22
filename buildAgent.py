# define tools for the agent  
from langchain.agents import Tool  
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  
from langchain.tools.render import render_text_description_and_args  
from langchain.schema.runnable import RunnablePassthrough  
from langchain.agents.output_parsers import JSONAgentOutputParser  
from langchain.agents.format_scratchpad import format_log_to_str  
from langchain_openai import OpenAI  
import os
from langchain.agents import AgentExecutor
from defineTool import vector_search_tool_func, web_search_tool_func, mcp_tool_func  
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import PromptTemplate
from langchain.agents.output_parsers import JSONAgentOutputParser
from dotenv import load_dotenv

load_dotenv()

# llm = OpenAI(temperature=0, model="gpt-3.5-turbo", api_key=os.environ["OPENAI_API_KEY"])
model = ChatDeepSeek(model="deepseek-chat", api_key=os.getenv["DS_API_KEY"])  
vector_search_tool = vector_search_tool_func()  # 向量检索工具

tools = [  
    vector_search_tool,
    web_search_tool_func,
    mcp_tool_func
    # Tool(
    #     name="WebSearch",  # 工具名称
    #     func=,
    #     description="用于检索网页的工具"
    #     ),
    # Tool(
    #     name="MCPWebSearchTool",  # 工具名称
    #     func=mcp_tool_func,
    #     description="用于通过mcp协议检索网页的工具"
    #     )  # 目前不支持异步调用
]

# define system prompt
prompt_template = """
尽你所能用中文回答一下问题，你可以使用这些工具：{tools}。
使用以下格式进行响应，直到输出最终结果（Final Answer）为止：
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
Question: {input}
Thought:{agent_scratchpad}
"""

system_prompt = """请尽可能以有帮助和准确的方式回应人类。你可以使用以下工具：{tools}。
始终首先尝试使用“VectorStoreSearch”工具在向量数据库中搜索相关信息。如果向量数据库中不包含所需信息，
再使用“WebSearch”工具或“MCPWebSearchTool”工具。
使用JSON对象通过提供action key（工具名称）和action_input key（工具输入）来指定工具。 
有效的“action”值包括：“Final Answer”或者{tool_names}。每个JSON对象只提供一个action。"""

# human prompt  
human_prompt = """用户问题：{input}
当前进度：{agent_scratchpad}
注意：始终以JSON对象进行相应。"""


# create prompt template
# prompt = PromptTemplate.from_template(prompt_template)  
prompt = ChatPromptTemplate.from_messages(  
    [  
        ("system", system_prompt),  
        ("human", human_prompt),  
    ]  
)
prompt = prompt.partial(  
    tools=render_text_description_and_args(list(tools)),  
    tool_names=", ".join([t.name for t in tools]),  
)
# 创建react Agent
agent = create_react_agent(model, tools, prompt)
output_parser = JSONAgentOutputParser()

# 创建代理执行器
# planner = load_chat_planner(model)
# executor = load_agent_executor(model, tools, verbose=True)
# 初始化Plan-and-Execute Agent
# agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
# agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, handle_parsing_errors=True, verbose=False)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, output_parser=output_parser, return_intermediate_steps=True, handle_parsing_errors=True, verbose=True)

print(agent.invoke({"input": "介绍一下deepseek关键技术", "intermediate_steps": []}))