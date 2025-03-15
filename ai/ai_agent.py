from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool
import os
import json

load_dotenv()

# Provide response schema 
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


# Importing AI Model
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    api_key=os.getenv("OPENAI_API_KEY")
)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Chat Prompt for AI
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            # Change below later
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Cheating the AI Agent
tools = [search_tool, wiki_tool, save_tool]

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)


# This is the main program functionality
def main():
    while True:
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        query = input("What can i help you research? ")
        raw_response = agent_executor.invoke({"query": query})
        
        
        # print(raw_response)


        try:
            # First parse the JSON string into a Python dictionary
            output_str = raw_response.get("output")
            output_dict = json.loads(output_str)
            
            # Then create the Pydantic model from the dictionary
            structured_response = ResearchResponse(**output_dict)
            
            print(f"Summary: {structured_response.summary}")


        except Exception as e:
            print("Error parsing response", e)
            print("Raw Response - ", raw_response)
            
            # Add additional debugging
            if "output" in raw_response:
                print("\nOutput type:", type(raw_response["output"]))
                print("Output content:", raw_response["output"])

        
        continue_input = input("Do you wanna continue? (y/ n)")
        
        if continue_input == "y":
            continue
        else:
            break


if __name__ == "__main__":
    main()