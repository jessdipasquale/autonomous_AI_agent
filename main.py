"""
Main entrypoint for the Market Entry Agent.
Executed by n8n via Execute Command node.

Usage:
    python main.py "GT's Living Foods" "Longevity / Health Food" "Germany"
"""

import sys
import json
import os
import contextlib
from dotenv import load_dotenv
load_dotenv()

from rag.rag import create_pinecone_index, load_and_embed_documents
from agent.agent import run


def run_agent(company: str, industry: str, target_market: str):
    final_state = run(company=company, industry=industry, target_market=target_market)
    return final_state.get("final_report_md", "No report generated.")


if __name__ == "__main__":
    company = sys.argv[1] if len(sys.argv) > 1 else "GT's Living Foods"
    industry = sys.argv[2] if len(sys.argv) > 2 else "Longevity / Health Food"
    target_market = sys.argv[3] if len(sys.argv) > 3 else "Germany"

    try:
        # Tutto ciò che stampa durante l'esecuzione va su stderr
        with contextlib.redirect_stdout(sys.stderr):
            create_pinecone_index()
            load_and_embed_documents()
            report = run_agent(company, industry, target_market)

        # Salva il report su file .md
        output_file = f"report_{company.replace(' ', '_')}_{target_market}.md"
        with open(output_file, "w") as f:
            f.write(report)

        # Solo questo arriva a n8n (stdout pulito)
        print(json.dumps({"status": "success", "report": report, "output_file": output_file}))

    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)