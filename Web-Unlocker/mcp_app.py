import os
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
from langchain_groq import ChatGroq
import streamlit as st
from langchain_openai import ChatOpenAI

MCP_TOKEN = os.environ.get("MCP_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

mcp = MultiServerMCPClient(
    {
        "scrape-do": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "-y",
                "--registry=https://registry.npmjs.org",
                "scrape-do-mcp"
            ],
            "env": {
                "SCRAPE_DO_TOKEN": MCP_TOKEN
            }
        }
    }
)


async def mcp_tools():
    tools = await mcp.get_tools()

    for idx, tool in enumerate(tools):
        print(idx, tool.name)
        print(tool.args)
        print("********************")

    return tools


async def handle_prompt(tools, url, user_prompt):
    scraper = tools[0]
    result = await scraper.ainvoke({
        "url": url
    })

    # user_prompt = "what is this website?"
    # Fixed the string syntax error in full_prompt
    full_prompt = (
        "context (scraped data):\n\n"
        + str(result)
        + "\n\nquestion:\n\n"
        + user_prompt
    )

    # Replaced OllamaLLM with ChatGroq
    # llm = ChatGroq(
    #    model="gemma2-9b-it",
    #    api_key=GROQ_API_KEY,
    # )
    llm = ChatOpenAI(
        model="openrouter/free",  # Or "deepseek/deepseek-r1:free"
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    llm_response = llm.invoke(full_prompt)

    # Printed the text content of the returned AIMessage
    print(llm_response.content)
    return llm_response.content
    # print(result)
    # return result


tools = asyncio.run(mcp_tools())


st.title("LLM web search")
url = st.text_input("Enter URL")
user_prompt = st.text_area("Enter Question:")
submit = st.button("submit")

if submit:
    with st.spinner("Processing..."):
        llm_response = asyncio.run(handle_prompt(tools, url, user_prompt))
        st.write(llm_response)
