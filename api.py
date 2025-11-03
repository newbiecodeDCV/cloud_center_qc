from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from datetime import datetime
from loguru import logger
from src.qa_communicate.core.utils import is_url, is_audio_file, create_task_id, write_json
from urllib.request import urlopen
from src.main_evaluator import QAMainEvaluator
from time import perf_counter
import uvicorn
import jwt
import os
import argparse



app = FastAPI()
PRIVATE_TOKEN = "41ad97aa3c747596a4378cc8ba101fe70beb3f5f70a75407a30e6ddab668310d"


async def get_qa(audio_bytes: bytes,
                 task_id: int,
                 done_json_file_path: str = None,
                 running_json_file_path: str = None):
    rs = {"status": 0, "task_id": task_id, "result": {},
          "message": "Start evaluating", "code": 202}
    write_json(rs, running_json_file_path)
    start = perf_counter()
    result = await main_evaluator.run_evaluate(audio_bytes=audio_bytes,
                                               task_id=task_id)
    end = perf_counter()
    rs = {"status": result['status'],
          "task_id": task_id,
          "inference_time": round(end - start, 2),
          "result": result.get('final_detail_result', {}),
          "segments": result.get('segments', []),
          "message": result['message'],
          "code": result['code']}
    if result['status'] == 1:
        logger.info(f"Task {task_id} is done")
    else:
        logger.error(f"Task {task_id} failed with message: {result['message']}")
    write_json(rs, done_json_file_path)


@app.get("/")
def get():
    return {"use POST method"}


@app.post("/")
async def post(request: Request, bg_task: BackgroundTasks):
    token = request.headers.get("Authorization", None)
    if not token:
        return JSONResponse({
            "status": -1,
            "code": 400,
            "message": "Token is missing",
            "data": {}
        })

    if token == PRIVATE_TOKEN:
        pass
    else:
        try:
            token = jwt.decode(token, "datamining_vcc", algorithms="HS256")
        except jwt.ExpiredSignatureError:
            return JSONResponse({
                "status": -1,
                "message": "Signature expired. Please log in again",
                "data": {},
                "code": 401
            })
        except jwt.InvalidTokenError:
            return JSONResponse({
                "status": -1,
                "message": "Invalid token. Please log in again",
                "data": {},
                "code": 400
            })
    current_month = datetime.now().strftime("%Y-%m")
    save_path = os.path.join(os.getcwd(), f"tasks_{current_month}")
    os.makedirs(save_path, exist_ok=True)
    request = await request.form()
    file = request["file"]
    task_id = request.get("task_id", None)
    try:
        if is_url(file):
            if task_id is None:
                task_id = create_task_id(url=file)
            audio_bytes = urlopen(file).read()
        else:
            bytes = await file.read()
            if is_audio_file(bytes):
                await file.seek(0)
                if task_id is None:
                    task_id = create_task_id(audio_bytes=bytes)
                audio_bytes = bytes
            else:
                return JSONResponse({
                    "status": -1,
                    "message": "Only accept file url and bytes",
                    "code": 400
                    })
    except Exception:
        return JSONResponse({
            "status": -1,
            "message": "Only accept file url and bytes",
            "code": 400
        })
    done_json_file_path = os.path.join(save_path, f'{task_id}_done.json')
    running_json_file_path = os.path.join(save_path, f'{task_id}_running.json')

    if not os.path.exists(done_json_file_path):
        if not os.path.exists(running_json_file_path):
            bg_task.add_task(get_qa, audio_bytes, task_id,
                             done_json_file_path,
                             running_json_file_path)
            return JSONResponse({
                "status": 0,
                "message": f"Task {task_id} is started",
                "task_id": task_id,
                "code": 202
            })
        else:
            return JSONResponse({
                "status": 0,
                "message": f"Task {task_id} is already running",
                "task_id": task_id,
                "code": 202
            })
    else:
        return JSONResponse({
            "status": 1,
            "message": f"Task {task_id} is complete",
            "task_id": task_id,
            "code": 200,
        })


# ==================== SHUTDOWN HANDLER ====================
@app.on_event("shutdown")
async def shutdown_event():
    """Flush Langfuse events trước khi shutdown"""
    logger.info("Shutting down API server...")
    logger.info("Langfuse events flushed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QA Evaluator API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpt_model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--csv_path", type=str, default="src/qa_sales/modules/databases/salescript.tsv")
    parser.add_argument("--eval_prompt_template", type=str, default="src/qa_sales/modules/prompt_templates/evaluate_script.txt")
    parser.add_argument("--preprocess_prompt_template", type=str, default="src/qa_sales/modules/prompt_templates/preprocess.txt")
    parser.add_argument("--classify_prompt_template", type=str, default="src/qa_sales/modules/prompt_templates/classify_utterances.txt")
    parser.add_argument("--db_path", type=str, default="src/qa_sales/modules/databases/salescript_db")
    args = parser.parse_args()
    main_evaluator = QAMainEvaluator(gpt_model=args.gpt_model,
                                     csv_path=args.csv_path,
                                     eval_prompt_template=args.eval_prompt_template,
                                     preprocess_prompt_template=args.preprocess_prompt_template,
                                     classify_prompt_template=args.classify_prompt_template,
                                     db_path=args.db_path)

    uvicorn.run(app, host=args.host, port=args.port)