from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool
import os
import stt_tts

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

    # Initialise the language model
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=api_key
    )
    # Define a parser to convert the AI's output into the Pydantic model
    parser = PydanticOutputParser(pydantic_object=ResearchResponse)

    # Chat Prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI assistant. Keep responses concise.\n{format_instructions}"),
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
        # Parse the output string from the raw response
        output_str = raw_response.get("output")
        return ResearchResponse.parse_raw(output_str).summary
    
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(f"Raw Response: {raw_response}")
        return None

# Runs the main AI functionality after activation
def run_ai(agent, tools):
    # Ensure agent and tools are initialized
    if not agent or not tools:
        print("Error: Agent initialization failed. Exiting AI loop.")
        return
    
    
    # Create an executor to run the agent with the tools
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    chat_history = []

    # Words to deactivate the AI
    deactivate_words = ["bye", "goodbye", "later"]

    while True:
        query = stt_tts.record_audio()

        # Record user query and chat history
        if query:
            chat_history.append({"role": "user", "content": query})
            raw_response = agent_executor.invoke({"query": query, "chat_history": chat_history})
            chat_history.append({"role": "assistant", "content": raw_response.get("output")})
        elif not query:
            print("Error: Failed to record audio. Please try again.")
            continue

        # Check if the user wants to exit
        if any(word in query.lower() for word in deactivate_words):
            stt_tts.output_audio("Goodbye!")
            break

        # Process the response
        summary = process_response(raw_response)
        if summary:
            print(f"User Query: {query}")
            print(f"AI Response: {summary}")
            # Convert the summary to speech using text-to-speech
            stt_tts.output_audio(summary)

# Function to listen for the activation word
def listen_for_activation():
     # Word for activation
    activate_words = ["hey", "hello", "hi", "yo", "hey jit"]

    # Keep listening for the activation word
    while True:
        print("Listening for activation...")
        detected_text = stt_tts.record_audio()
        print(f"Detected text: {detected_text}")

        # activation word detected
        if any(word in detected_text.lower() for word in activate_words):
            stt_tts.output_audio("Hey!")
            return True
        
        print("Activation word not detected. Waiting...")

# Function to link the AI with fatigue level
"""
def fatigue_level(agent, tools, avg_fatigue):
    # run AI if fatigue level is high
    if avg_fatigue > 0.5:
        stt_tts.output_audio("You seem tired. Do you want to chat?")
        run_ai(agent, tools)
"""

# Main function
def main():
    # Initialise the agent and tools
    agent, tools = initialise_agent()
    
    # Keep listening for activation and run the AI
    while True:
        if listen_for_activation():
            run_ai(agent, tools)

if __name__ == "__main__":
    main()