from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from scripts.ai.tools import search_tool, wiki_tool
import os
import json
import scripts.ai.stt_tts

"""
TDL
- connect AI with fatigue level
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
        model="gpt-4o-mini",
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
                "You are a helpful assistant sitting in the passenger seat of a car. 
                Your job is to make light conversation with the driver. 
                Keep your sentences short and sweet." \n{format_instructions}
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

# Runs the main AI functionality after activation
def run_ai(agent, tools):
    # Create an executor to run the agent with the tools
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    chat_history = []
    while True:
        query = scripts.ai.stt_tts.record_audio()

        # Record user query and chat history
        if query:
            chat_history.append({"role": "user", "content": query})
            raw_response = agent_executor.invoke({"query": query, "chat_history": chat_history})
            chat_history.append({"role": "assistant", "content": raw_response.get("output")})

        # Check if the user wants to exit
        deactivate_words = ["bye", "goodbye", "later"]
        if any(word in query.lower() for word in deactivate_words):
            scripts.ai.stt_tts.output_audio("Goodbye!")
            break
        else:
            # display the query 
            print(query)

        # Process the response
        summary = process_response(raw_response)
        if summary:
            print(f"Summary: {summary}")

            # Convert the summary to speech using text-to-speech
            scripts.ai.stt_tts.output_audio(summary)

def main():
    # Initialize the agent and tools
    agent, tools = initialise_agent()

    # Word for activation
    activate_words = ["hey", "hello", "hi", "yo", "hey jit"]

    while True:    
        #if avg_fatigue < x:
           # save_lives()

        print("Listening for activation...")
        detected_text = scripts.ai.stt_tts.record_audio()
        print(f"Detected text: {detected_text}")  

        # Check if the wake word is detected
        if detected_text and any(activate_word in detected_text.lower() for activate_word in activate_words):
            scripts.ai.stt_tts.output_audio("Hey!")
            # Activate the AI
            run_ai(agent, tools)  
        else:
            print("Activate word not detected. Please try again.")

#def save_lives():
    # wake up, talk to jit by saying hey
    #jkljlk

if __name__ == "__main__":
    main()