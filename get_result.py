import argparse
import json
import os
import sys
from datetime import datetime

import jwt
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

app = FastAPI()
PRIVATE_TOKEN = "41ad97aa3c747596a4378cc8ba101fe70beb3f5f70a75407a30e6ddab668310d"


@app.get("/")
def get():
    return {"use POST method"}


@app.post("/")
async def get_result(request: Request):
    print(request)
    token = request.headers.get("Authorization", None)
    if not token:
        return JSONResponse(
            {
                "status": -1,
                "code": 400,
                "message": "Token is missing",
            }
        )

    if token == PRIVATE_TOKEN:
        pass
    else:
        try:
            token = jwt.decode(token, "datamining_vcc", algorithms="HS256")
        except jwt.ExpiredSignatureError:
            return JSONResponse(
                {
                    "status": -1,
                    "message": "Signature expired. Please log in again",
                    "code": 401,
                }
            )
        except jwt.InvalidTokenError:
            return JSONResponse(
                {
                    "status": -1,
                    "message": "Invalid token. Please log in again",
                    "code": 400,
                }
            )
    current_month = datetime.now().strftime("%Y-%m")
    save_path = os.path.join(os.getcwd(), f"tasks_{current_month}")
    request = await request.form()
    task_id = request["task_id"]
    done_json_file_path = os.path.join(save_path, f"{task_id}_done.json")
    running_json_file_path = os.path.join(save_path, f"{task_id}_running.json")
    if not os.path.exists(done_json_file_path):
        if os.path.exists(running_json_file_path):
            with open(running_json_file_path, "r") as f:
                data = json.load(f)
            return JSONResponse(content=data)
        else:
            return JSONResponse(
                {"status": -1, "message": f"Not found task id {task_id}", "code": 400}
            )
    else:
        with open(done_json_file_path, "r") as f:
            data = json.load(f)
        return JSONResponse(content=data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run arguments.")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
    )
    parser.add_argument("--port", type=int, default=9103)
    log_dir = os.getcwd()
    log_level = "INFO"
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
    logger.add(
        sys.stderr,
        level=log_level,
        format=log_format,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    logger.add(
        os.path.join(log_dir, "file.log"),
        level=log_level,
        format=log_format,
        colorize=False,
        backtrace=True,
        diagnose=True,
    )
    args = parser.parse_args()
    print(args)
    uvicorn.run(app, host=args.host, port=args.port)
