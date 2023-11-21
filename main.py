from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI
from langchain.utilities import SQLDatabase
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

POSTGRES_LOCAL_URI = os.getenv("POSTGRES_LOCAL_URI")

db = SQLDatabase.from_uri(POSTGRES_LOCAL_URI)

agent_executor_chat_model = create_sql_agent(
    llm=ChatOpenAI(model='gpt-3.5-turbo-1106', temperature=0),
    toolkit=SQLDatabaseToolkit(db=db, llm=ChatOpenAI(model='gpt-3.5-turbo-1106', temperature=0)),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

agent_executor = create_sql_agent(
    llm=OpenAI(temperature=0),
    toolkit=SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0)),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

# output = agent_executor.run("List all users which can be defaulter in the future. Also include user_id in the output.")
output = agent_executor.run("Recommend all products which would be great for a user with 'user_id' = 9562. Also Include 'lender_id' in the output. Explain your recommendation.")
