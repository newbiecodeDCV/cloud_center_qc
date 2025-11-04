from src.qa_communicate.audio_processing.dialogue import call_dialogue_api
from src.qa_sales.modules.database import create_csvdatabase
from src.qa_sales.modules.dialogue_processor import DialogueProcessor
from src.qa_sales.modules.evaluators import ScriptEvaluator
from src.qa_communicate.core.langfuse_config import log_span, log_generation
from typing import List, Dict
from logging import getLogger
import pandas as pd

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
        """Format evaluation results into readable text."""
        detail_result = "Đánh giá kỹ năng bán hàng theo các tiêu chí:\n"
        final_score = 0.0
        
        for item in results:
            criteria_id = item['criteria_id']
            criteria_name = self.criteria_name.get(criteria_id, "Unknown")
            status = 'đạt' if item.get('status', 0) == 1 else 'chưa đạt'
            note = item.get('Note', '')
            score = item.get('score', 0)
            final_score += score
            
            detail_result += (
                f"\n+ Tiêu chí {criteria_id}: {criteria_name}\n"
                f"  - Đánh giá: {status}\n"
                f"  - Điểm: {score}\n"
                f"  - Nhận xét: {note}\n"
            )
        
        detail_result += f"\nTổng điểm kỹ năng bán hàng: {final_score:.2f}"
        return detail_result, final_score

    async def run_evaluate(self,
                           audio_bytes: bytes,
                           task_id: int,
                           trace=None,
                           parent_span=None):
        """
        Run sales skills evaluation with Langfuse tracing.
        
        Sub-steps:
        1. Get dialogue from API
        2. Preprocess dialogue (identify speakers)
        3. Classify utterances to criteria
        4. Evaluate each criterion
        5. Calculate final score
        """
        try:
            # Sub-step 1: Get dialogue
            if parent_span:
                dialogue_span = parent_span.span(
                    name="get_dialogue",
                    input={"task_id": task_id}
                )
            
            logger.info("Getting dialogue from API...")
            dialogue_result = await call_dialogue_api(
                audio_bytes=audio_bytes,
                task_id=task_id
            )
            
            if parent_span:
                dialogue_span.end(
                    output={
                        "status": dialogue_result.get('status'),
                        "num_segments": dialogue_result.get('dialogue', [])
                    }
                )

            if dialogue_result['status'] != 1:
                return {
                    'status': -1,
                    'detail_result': 'Failed to get dialogue from audio',
                    'final_score': -1
                }
            
            # Sub-step 2: Preprocess dialogue (identify speakers)
            if parent_span:
                preprocess_span = parent_span.span(
                    name="preprocess_dialogue",
                    input={
                        "raw_segments": dialogue_result['dialogue']
                    }
                )
            
            logger.info("Preprocessing dialogue (identifying speakers)...")
            processed_result = await self.dialogue_processor(
                dialogue=dialogue_result['dialogue'],
                prompt_template=self.pre_prompt_template,
                trace=trace,
                parent_span=preprocess_span if parent_span else None
            )
            
            if parent_span:
                preprocess_span.end(
                    output={
                        "status": processed_result.get('status'),
                        "sales_segments": processed_result.get('dialogue', [])
                    }
                )

            if processed_result['status'] != 1:
                return {
                    'status': -1,
                    'detail_result': 'Failed to process dialogue',
                    'final_score': -1
                }

            # Sub-step 3 & 4: Classify and evaluate (done in ScriptEvaluator)
            if parent_span:
                eval_span = parent_span.span(
                    name="evaluate_sales_script",
                    input={
                        "sales_utterances": processed_result['dialogue']
                    }
                )
            
            logger.info("Evaluating sales script...")
            results = await self.script_evaluator(
                dialogue=processed_result['dialogue'],
                trace=trace,
                parent_span=eval_span if parent_span else None
            )
            
            if parent_span:
                eval_span.end(
                    output={
                        "status": results.get('status'),
                        "criteria_evaluated": results.get('criteria_evals', [])
                    }
                )

            if results['status'] != 1:
                return {
                    'status': -1,
                    'detail_result': 'Failed to evaluate dialogue',
                    'final_score': -1
                }
            
            # Sub-step 5: Format results
            detail_result, final_score = self.process_result(
                results=results['criteria_evals']
            )
            
            logger.info(f"✓ Sales evaluation completed. Score: {final_score}")
            
            return {
                'status': 1,
                'detail_result': detail_result,
                'final_score': final_score
            }
            
        except Exception as e:
            logger.error(f"Error in sales evaluation: {e}", exc_info=True)
            return {
                'status': -1,
                'detail_result': str(e),
                'final_score': -1
            }