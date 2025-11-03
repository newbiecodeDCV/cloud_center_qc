from src.qa_communicate.audio_processing.dialogue import call_dialogue_api
from src.qa_sales.modules.database import create_csvdatabase
from src.qa_sales.modules.dialogue_processor import DialogueProcessor
from src.qa_sales.modules.evaluators import ScriptEvaluator
from typing import List, Dict
from logging import getLogger
import pandas as pd
from src.qa_communicate.core.langfuse_config import (
    LANGFUSE_ENABLED,
    log_span
)

logger = getLogger(__name__)


class QASalesEvaluator:
    def __init__(self,
                 model: str = "gpt-4.1-mini",
                 csv_path: str = "data/sale_criteria.tsv",
                 eval_prompt_template: str = "prompt_templates/evaluate_script.txt",
                 preprocess_prompt_template: str = "prompt_templates/preprocess.txt",
                 classify_prompt_template: str = "prompt_templates/classify_utterances.txt",
                 db_path: str = "data/sale_criteria_db"):
        self.sale_script_db = create_csvdatabase(csv_path=csv_path,
                                                 db_path=db_path)

        self.dialogue_processor = DialogueProcessor()
        self.script_evaluator = ScriptEvaluator(model=model,
                                                eval_prompt_template=eval_prompt_template,
                                                classify_prompt_template=classify_prompt_template,
                                                chroma_db=self.sale_script_db,
                                                tsv_path=csv_path)
        self.pre_prompt_template = preprocess_prompt_template
        df = pd.read_csv(csv_path, delimiter="\t")
        self.criteria_name = dict(zip(df['criteria_id'], df['criteria_name']))

    def process_result(self, results: List[Dict]):
        # Process the result from the evaluation
        detail_result = "II. Đánh giá kỹ năng bán hàng theo các tiêu chí:\n"
        final_score = 0.0
        for item in results:
            criteria_id = item['criteria_id']
            criteria_name = self.criteria_name.get(criteria_id, "Unknown")
            status = 'đạt' if item.get('status', 0) == 1 else 'chưa đạt'
            note = item.get('Note', '')
            note = "\n".join("\t" + line for line in note.splitlines())
            score = item.get('score', 0)
            final_score += score
            detail_result += f"+ Tiêu chí: {criteria_name}, đánh giá: {status}, điểm: {score}, nhận xét:\n{note} \n"
        final_score = round(final_score, 2)
        return detail_result, final_score

    async def run_evaluate(self,
                           audio_bytes: bytes,
                           task_id: int,
                           trace=None):
        """
        Run sales evaluation
        
        Args:
            audio_bytes: Audio data
            task_id: Task ID
            trace: Langfuse trace (optional)
        """
        # Span 1: Get dialogue from audio
        if LANGFUSE_ENABLED and trace:
            log_span(
                trace=trace,
                name="extract_dialogue",
                input_data={"audio_size": len(audio_bytes)},
                metadata={"step": "1/3"}
            )
        
        dialogue_result = await call_dialogue_api(audio_bytes=audio_bytes,
                                                  task_id=task_id)

        if dialogue_result['status'] != 1:
            if trace:
                trace.update(
                    output={
                        "status": "error",
                        "error": "Failed to get dialogue from audio"
                    }
                )
            return {'status': -1,
                    'message': 'Failed to get dialogue from audio'}
        
        # Span 2: Process dialogue
        if LANGFUSE_ENABLED and trace:
            log_span(
                trace=trace,
                name="process_dialogue",
                input_data={"dialogue_length": len(dialogue_result['dialogue'])},
                metadata={"step": "2/3"}
            )
        
        processed_result = await self.dialogue_processor(
            dialogue=dialogue_result['dialogue'],
            prompt_template=self.pre_prompt_template
        )

        if processed_result['status'] != 1:
            if trace:
                trace.update(
                    output={
                        "status": "error",
                        "error": "Failed to process dialogue"
                    }
                )
            return {'status': -1,
                    'message': 'Failed to process dialogue'}

        # Span 3: Evaluate
        if LANGFUSE_ENABLED and trace:
            log_span(
                trace=trace,
                name="evaluate_sales_script",
                input_data={"processed_dialogue_length": len(processed_result['dialogue'])},
                metadata={"step": "3/3"}
            )
        
        results = await self.script_evaluator(
            dialogue=processed_result['dialogue'],
            trace=trace  # Pass trace to evaluator
        )

        if results['status'] != 1:
            if trace:
                trace.update(
                    output={
                        "status": "error",
                        "error": "Failed to evaluate dialogue"
                    }
                )
            return {'status': -1,
                    'message': 'Failed to evaluate dialogue'}
        
        detail_result, final_score = self.process_result(results=results['criteria_evals'])
        
        if trace:
            trace.update(
                output={
                    "status": "success",
                    "final_score": final_score,
                    "criteria_count": len(results['criteria_evals'])
                }
            )
        
        return {'status': 1,
                'detail_result': detail_result,
                'final_score': final_score}