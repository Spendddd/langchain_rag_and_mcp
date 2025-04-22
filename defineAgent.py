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
from defineTool import vector_search_tool_func, web_search_tool_func, mcp_tool_func, create_tools
from mcp_configs import mcp_configs     
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import PromptTemplate
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain_mcp_tools import convert_mcp_to_langchain_tools
import asyncio
import logging
import os
import sys
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# todo：换成deepseek的api key

# llm = OpenAI(temperature=0, model="gpt-3.5-turbo", api_key=os.environ["OPENAI_API_KEY"])
# Standard library imports

# Third-party imports
try:
    from dotenv import load_dotenv
    from langchain.chat_models import init_chat_model
    from langchain.schema import HumanMessage
    from langchain_ollama import ChatOllama
    from langgraph.prebuilt import create_react_agent
except ImportError as e:
    print(f'\nError: Required package not found: {e}')
    print('Please ensure all required packages are installed\n')
    sys.exit(1)

# Local application imports

# A very simple logger
def init_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,  # logging.DEBUG,
        format='\x1b[90m[%(levelname)s]\x1b[0m %(message)s'
    )
    return logging.getLogger()

async def run(query) -> None:
    # Be sure to set ANTHROPIC_API_KEY and/or OPENAI_API_KEY as needed
    load_dotenv()

    # Check the api key early to avoid showing a confusing long trace
    # if not os.environ.get('ANTHROPIC_API_KEY'):
    #     raise Exception('ANTHROPIC_API_KEY env var needs to be set')
    # if not os.environ.get('OPENAI_API_KEY'):
    #     raise Exception('OPENAI_API_KEY env var needs to be set')

    try:
        # mcp_configs = {
        #     "tavily-mcp": {
        #         "command": "npx",
        #         "args": ["-y", "tavily-mcp@0.1.4"],
        #         "env": {
        #             "TAVILY_API_KEY": os.environ["TAVILY_API_KEY"],
        #         },
        #         "disabled": False,
        #         "autoApprove": []
        #     }
        # }

        mcptools, cleanup = await convert_mcp_to_langchain_tools(
            mcp_configs,
            init_logger()
        )
        # print(mcptools)
        # tools = [  
        #     vector_search_tool_func,
        #     web_search_tool_func,
        # ]
        tools = create_tools()
        tools.extend(mcptools)
        model = ChatDeepSeek(model="deepseek-chat", api_key=os.environ["DS_API_KEY"])  
        
        system_prompt = """请尽可能以有帮助和准确的方式回应人类。你可以使用以下工具：{tools}。
        始终首先尝试使用向量数据库搜索工具在向量数据库中搜索相关信息。如果向量数据库中不包含所需信息，再使用其他工具。
        使用JSON对象通过提供action key（工具名称）和action_input key（工具输入）来指定工具。 
        有效的“action”值包括：“Final Answer”或者{tool_names}。每个JSON对象只提供一个action。"""

        # human prompt  
        human_prompt = """用户问题：{messages}。
        注意：始终以JSON对象进行响应。"""

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

        agent = create_react_agent(
            model,
            tools,
            prompt=prompt
        )
        # agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, return_intermediate_steps=True, handle_parsing_errors=True, verbose=True)

        print('\x1b[33m')  # color to yellow
        print(query)
        print('\x1b[0m')   # reset the color
        # messages = [HumanMessage(content=query)]
        # result = await agent.ainvoke({"messages": messages})
        result = await agent.invoke({"messages": query})
        # the last message should be an AIMessage
        response = result['messages'][-1].content
        print('\x1b[36m')  # color to cyan
        print(result)
        print(response)
        print('\x1b[0m')   # reset the color
    finally:
        if cleanup is not None:
            await cleanup()
def main(query) -> None:
    asyncio.run(run(query))

if __name__ == '__main__':
    main("介绍一下DeepSeek")