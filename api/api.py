from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from loguru import logger
from urllib.request import urlopen
from main_evaluator import QAMainEvaluator
import uvicorn
import httpx
import json
import jwt
import sys
import os
import argparse

app = FastAPI()
PRIVATE_TOKEN = "41ad97aa3c747596a4378cc8ba101fe70beb3f5f70a75407a30e6ddab668310d"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QA Evaluator API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main_evaluator = QAMainEvaluator()

    uvicorn.run(app, host=args.host, port=args.port)
