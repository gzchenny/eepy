from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool
import os
import json
import stt_tts

"""
TDL
- connect AI with fatigue level
- make AI cleaner 
"""
load_dotenv()

# Define the response schema using Pydantic BaseModel
class ResearchResponse(BaseModel):
    topic: str
    summary: str

# Function to initialise AI Agent
def initialise_agent():
     # Ensure the API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    # Initialize the language model
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        api_key=api_key
    )
    # Define a parser to convert the AI's output into the Pydantic model
    parser = PydanticOutputParser(pydantic_object=ResearchResponse)

    # Chat Prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are sitting in the passenger seat of a car. 
                The driver will talk to you so just make light conversation with them. 
                Make sure they're driving safe. \n{format_instructions}
                """,
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    # Define the tools the AI agent can use
    tools = [search_tool, wiki_tool]

    # Create the AI agent with the specified LLM, prompt, and tools
    agent = create_tool_calling_agent(
        llm=llm,
        prompt=prompt,
        tools=tools
    )
    return agent, tools

# Function to process the AI response
def process_response(raw_response):
    try:
        # Parse the JSON string into a Python dictionary
        output_str = raw_response.get("output")
        output_dict = json.loads(output_str)

        # Create the Pydantic model from the dictionary
        structured_response = ResearchResponse(**output_dict)

        # Return the summary
        return structured_response.summary
    except Exception as e:
        print("Error parsing response:", e)
        print("Raw Response:", raw_response)
        return None
    
# Main program functionality
def main():
   # Initialize the agent and tools
    agent, tools = initialise_agent()
    
    # Voice activation
    active = False
    wake_word = "hey"
    detected_text = stt_tts.record_audio()
    if detected_text and wake_word in detected_text.lower():
        active = True
        stt_tts.output_audio("Hey!")

    # main loop 
    while active:
        # Create an executor to run the agent with the tools
        agent_executor = AgentExecutor(agent=agent, tools=tools)

        # Use speech-to-text to get the user's query
        query = stt_tts.record_audio()
        if not query or query == "bye":
            stt_tts.output_audio("Goodbye!")
            break

        # Invoke the AI agent with the user's query
        raw_response = agent_executor.invoke({"query": query})

        # Process the response
        summary = process_response(raw_response)
        if summary:
            print(f"Summary: {summary}")

            # Convert the summary to speech using text-to-speech
            stt_tts.output_audio(summary)

    active = False

if __name__ == "__main__":
    main()