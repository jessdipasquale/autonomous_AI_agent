"""
FastAPI wrapper for the Market Entry Agent.
Exposes the agent as an HTTP API for n8n Cloud.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
from dotenv import load_dotenv
load_dotenv()

from rag.rag import create_pinecone_index, load_and_embed_documents
from agent.agent import run

app = FastAPI()

class AgentRequest(BaseModel):
    company: str
    industry: str
    target_market: str

@app.on_event("startup")
async def startup_event():
    create_pinecone_index()
    load_and_embed_documents()

@app.post("/run-agent")
async def run_agent(request: AgentRequest):
    try:
        final_state = run(
            company=request.company,
            industry=request.industry,
            target_market=request.target_market
        )
        report = final_state.get("final_report_md", "No report generated.")
        return {"status": "success", "report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))