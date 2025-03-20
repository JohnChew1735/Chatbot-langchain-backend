from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from langchain_ollama import OllamaLLM  # Correct import
from langchain.agents import initialize_agent, AgentType
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from sqlalchemy import create_engine, Column, Integer, Text, TIMESTAMP, func
from sqlalchemy.orm import sessionmaker, declarative_base
import math


# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change this for security)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Database Configuration
DATABASE_URL = "mysql+pymysql://test:test@localhost/backend"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define Chat History Table
class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_query = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    timestamp = Column(TIMESTAMP, server_default=func.now())

# Create the table in the database
Base.metadata.create_all(bind=engine)

# Load Ollama model
llm = OllamaLLM(model="mistral")


#Open weather tool
OPENWEATHERMAP_API_KEY = "dedda7f59f0cd310c6626ae34cd8242c"
weather_api = OpenWeatherMapAPIWrapper(openweathermap_api_key=OPENWEATHERMAP_API_KEY)


def get_weather(city: str):
    """Fetch weather information for a given city."""
    try:
        return weather_api.run(city)
    except Exception as e:
        return f"Error fetching weather: {str(e)}"


weather_tool = Tool(
    name="Weather Info",
    func=get_weather,
    description="Use this tool when the user asks about the weather. Input should be a city name."
)

#Tavily search tool
TAVILY_API_KEY = "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"
tavily_search = TavilySearchAPIWrapper(tavily_api_key=TAVILY_API_KEY)

search_tool = Tool(
    name="Tavily Search",
    func=lambda query: tavily_search.run(query, num_results=2),
    description="Performs a web search and returns the top results. Input should be a search query string."
)


#Calculate tool
def calculate(expression: str) -> str:
    """Safely evaluates a mathematical expression using the math module."""
    try:
        # Create a safe dictionary of allowed functions
        safe_dict = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        safe_dict.update({"abs": abs, "round": round})  # Allow abs and round
        
        # Evaluate the expression safely
        return str(eval(expression, {"__builtins__": {}}, safe_dict))
    except Exception as e:
        return f"Invalid math expression: {e}"

calculate_tool = Tool(
    name="calculate",
    func=calculate,
    description="Evaluates mathematical expressions. Input should be a valid math expression as a string."
)

#Wikipedia tool    
wiki_api = WikipediaAPIWrapper()

def safe_wikipedia_lookup(query: str):
    """Fetch Wikipedia information with error handling."""
    try:
        return wiki_api.run(query)
    except Exception as e:
        return f"Error fetching Wikipedia data: {str(e)}"

wiki_tool = Tool(
    name="Wikipedia Lookup",
    func=safe_wikipedia_lookup,
    description="Use this tool to get information from Wikipedia. Input should be a topic name."
)

#agent executor
agent_executor = initialize_agent(
    tools=[weather_tool, search_tool, calculate_tool, wiki_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # This allows the agent to decide which tool to use
    verbose=True,
    handle_parsing_errors = True
)

# Define request schema
class ChatRequest(BaseModel):
    query: str  # User's input query

@app.post("/chat")
def chat(request: ChatRequest):
    session = SessionLocal()
    try:
        # Get AI response
        response = agent_executor.invoke({"input": request.query})
        bot_reply = response["output"]

        # Store in the database
        chat_entry = ChatHistory(user_query=request.query, bot_response=bot_reply)
        session.add(chat_entry)
        session.commit()

        return {"response": bot_reply}

    except Exception as e:
        session.rollback()
        return {"error": str(e)}

    finally:
        session.close()

# Retrieve Chat History API
@app.get("/get_chat_history")
def get_chat_history():
    session = SessionLocal()
    try:
        chat_records = session.query(ChatHistory).all()
        if chat_records:
            return {"message": "Chat history found", "chatHistory": [
                {"id": chat.id, "user_query": chat.user_query, "bot_response": chat.bot_response, "timestamp": chat.timestamp}
                for chat in chat_records
            ]}
        else:
            raise HTTPException(status_code=404, detail="Chat history not found")

    except Exception as e:
        return {"error": str(e)}

    finally:
        session.close()


# Run the API
if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
