import uvicorn
from fastapi import FastAPI, Request
from src.graphs.graph_builder import GraphBuilder
from src.llms.groq_llm import GroqLLM

import os
from dotenv import load_dotenv
load_dotenv()


app = FastAPI()

os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

# Creating APIs

@app.post("/blogs")
async def create_blogs(request: Request):
    data=await request.json()
    topic=data.get("topic", "") 

    # Get the LLM object
    groq_llm=GroqLLM()
    llm=groq_llm.get_llm()

    # Get the Graph
    graph_builder=GraphBuilder(llm)

    if topic:
        graph=graph_builder.setup_graph(usecase="topic")
        state=graph.invoke({"topic": topic})

    return {"data": state}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

    # run in cmd -> python app.py
    # open postman desktop
    # POST: http://127.0.0.1:8000/blogs 
    # then Body -> raw ->
    # {
    # "topic": "Agentic AI"
    # }
    # click Send