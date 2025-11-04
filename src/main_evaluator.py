from src.qa_communicate.audio_processing.analysis import extract_features
from src.qa_communicate.evaluation.evaluator import get_qa_evaluation
from src.qa_sales.modules.qa_evaluators import QASalesEvaluator
from src.qa_communicate.core.langfuse_config import (
    create_trace, 
    log_span, 
    log_generation,
    flush_langfuse,
    LANGFUSE_ENABLED
)
from logging import getLogger
from datetime import datetime

logger = getLogger(__name__)


class QAMainEvaluator:
    def __init__(self,
                 gpt_model: str,
                 csv_path: str,
                 eval_prompt_template: str,
                 preprocess_prompt_template: str,
                 classify_prompt_template: str,
                 db_path: str):
        self.qa_evaluator = QASalesEvaluator(
            model=gpt_model,
            csv_path=csv_path,
            eval_prompt_template=eval_prompt_template,
            preprocess_prompt_template=preprocess_prompt_template,
            classify_prompt_template=classify_prompt_template,
            db_path=db_path
        )
        self.gpt_model = gpt_model

    async def evaluate_communication(self,
                                     audio_bytes: bytes,
                                     task_id: int,
                                     trace=None):
        """
        Evaluate communication quality with Langfuse tracing.
        
        Sub-steps:
        1. Extract audio features (dialogue API + acoustic analysis)
        2. LLM evaluation with prompt
        """
        span = None
        try:
            # Create main span for communication evaluation
            if trace:
                span = log_span(
                    trace=trace,
                    name="evaluate_communication",
                    metadata={
                        "task_id": task_id,
                        "evaluation_type": "communication_skills"
                    }
                )
            
            logger.info("Start analyzing audio features...")
            
            # Sub-step 1: Extract features
            if span:
                feature_span = span.span(
                    name="extract_audio_features",
                    input={"audio_size_bytes": len(audio_bytes)}
                )
            
            analysis_result = await extract_features(audio_bytes=audio_bytes)
            
            if span:
                feature_span.end(
                    output={
                        "status": analysis_result.get('status'),
                        "segments": analysis_result.get('segments', []),
                        "metadata": analysis_result.get('metadata')
                    }
                )
            
            logger.info("Finished analyzing audio features.")

            if analysis_result.get('status') != 1:
                if span:
                    span.end(output={"error": "Feature extraction failed"})
                return {
                    'detail_result': analysis_result.get('message'),
                    'score': -1,
                    'status': -1,
                    'segments': []
                }

            # Prepare data for LLM
            data_for_llm = {
                'metadata': analysis_result.get('metadata'),
                'segments': analysis_result.get('segments')
            }
            
            logger.info("Start evaluating communication quality with LLM...")
            
            # Sub-step 2: LLM Evaluation
            evaluation_result = await get_qa_evaluation(
                data_for_llm,
                trace=trace,
                parent_span=span
            )
            
            logger.info("Finished evaluating communication quality with LLM.")

            # Calculate scores
            chao_xung_danh = int(evaluation_result.get('chao_xung_danh', 0))
            ky_nang_noi = int(evaluation_result.get('ky_nang_noi', 0))  
            ky_nang_nghe = int(evaluation_result.get('ky_nang_nghe', 0))  
            thai_do = int(evaluation_result.get('thai_do', 0)) 

            tong_diem = 0.2*(chao_xung_danh + ky_nang_noi) + 0.8 * (ky_nang_nghe + thai_do)
            muc_loi = str(evaluation_result.get('muc_loi', 'Không'))
            ly_do = str(evaluation_result.get('ly_do', 'Không có lý do chi tiết'))

            final_result = {
                "task_id": task_id,
                "status": "completed",
                "chao_xung_danh": chao_xung_danh,
                "ky_nang_noi": ky_nang_noi,
                "ky_nang_nghe": ky_nang_nghe,
                "thai_do": thai_do,
                "tong_diem": tong_diem,
                "muc_loi": muc_loi,
                "ly_do": ly_do,
                "metadata": analysis_result.get('metadata'),
                "segments": analysis_result.get('segments')
            }
            
            ly_do = "\n".join("\t" + line for line in ly_do.splitlines())
            detail_result = "I. Đánh giá kỹ năng giao tiếp theo các tiêu chí:\n"
            detail_result += f"+ Chào xưng danh: {chao_xung_danh}\n"
            detail_result += f"+ Kỹ năng nói: {ky_nang_noi}\n"
            detail_result += f"+ Kỹ năng nghe: {ky_nang_nghe}\n"
            detail_result += f"+ Thái độ: {thai_do}\n"
            detail_result += f"+ Tổng điểm: {tong_diem}\n"
            detail_result += f"+ Lý do: \n{ly_do}\n"

            # End span with success
            if span:
                span.end(
                    output={
                        "status": "success",
                        "score": tong_diem,
                        "scores": {
                            "chao_xung_danh": chao_xung_danh,
                            "ky_nang_noi": ky_nang_noi,
                            "ky_nang_nghe": ky_nang_nghe,
                            "thai_do": thai_do
                        },
                        "muc_loi": muc_loi,
                        "ly_do": ly_do
                    }
                )

            return {
                'detail_result': detail_result,
                'score': tong_diem,
                'status': 1,
                'segments': analysis_result.get('segments')
            }
            
        except Exception as e:
            logger.error(f"Error evaluating communication: {e}")
            if span:
                span.end(output={"error": str(e)})
            return {
                'detail_result': str(e),
                'score': -1,
                'status': -1,
                'segments': []
            }

    async def evaluate_sale_skills(self,
                                   audio_bytes: bytes,
                                   task_id: int,
                                   trace=None):
        """
        Evaluate sales skills with Langfuse tracing.
        
        Sub-steps:
        1. Get dialogue from API
        2. Preprocess dialogue (extract speaker roles)
        3. Classify utterances to criteria
        4. Evaluate each criterion with LLM
        5. Calculate final score
        """
        span = None
        try:
            # Create main span for sales evaluation
            if trace:
                span = log_span(
                    trace=trace,
                    name="evaluate_sales_skills",
                    metadata={
                        "task_id": task_id,
                        "evaluation_type": "sales_skills"
                    }
                )
            
            logger.info("Start evaluating sales skills...")
            
            # Sub-step 1: Get dialogue (tracked inside qa_evaluator)
            if span:
                dialogue_span = span.span(
                    name="get_dialogue_api",
                    input={"audio_size_bytes": len(audio_bytes)}
                )
            
            # Note: dialogue API call happens inside run_evaluate
            # We'll track it by wrapping the call
            
            result = await self.qa_evaluator.run_evaluate(
                audio_bytes=audio_bytes,
                task_id=task_id,
                trace=trace,
                parent_span=span
            )
            
            if span:
                dialogue_span.end(
                    output={"status": result.get('status')}
                )
            
            if result['status'] != 1:
                if span:
                    span.end(output={"error": result.get('detail_result')})
                return {
                    'detail_result': result['detail_result'],
                    'score': -1,
                    'status': -1
                }
            
            # End span with success
            if span:
                span.end(
                    output={
                        "status": "success",
                        "final_score": result['final_score'],
                        "detail_result_preview": result['detail_result']
                    }
                )
            
            return {
                'detail_result': result['detail_result'],
                'score': result['final_score'],
                'status': 1
            }
            
        except Exception as e:
            logger.error(f"Error evaluating sales skills: {e}")
            if span:
                span.end(output={"error": str(e)})
            return {
                'detail_result': str(e),
                'score': -1,
                'status': -1
            }

    async def run_evaluate(self,
                           audio_bytes: bytes,
                           task_id: int):
        """
        Run full evaluation with Langfuse root trace.
        """
        trace = None
        try:
            # Create root trace
            if LANGFUSE_ENABLED:
                trace = create_trace(
                    name="qa_call_evaluation",
                    trace_id=str(task_id),
                    metadata={
                        "task_id": task_id,
                        "audio_size_bytes": len(audio_bytes),
                        "timestamp": datetime.now().isoformat(),
                        "model": self.gpt_model
                    }
                )
                logger.info(f"✓ Created Langfuse trace: {task_id}")
            
            # Step 1: Evaluate communication skills
            communication_result = await self.evaluate_communication(
                audio_bytes, 
                task_id,
                trace=trace
            )
            
            if communication_result['status'] != 1:
                if trace:
                    trace.update(
                        output={
                            "status": "failed",
                            "error": "Communication evaluation failed",
                            "message": communication_result['detail_result']
                        }
                    )
                    flush_langfuse()
                
                return {
                    'status': -1,
                    'message': communication_result['detail_result'],
                    'final_detail_result': None,
                    'code': 500
                }
            
            communication_detail = communication_result['detail_result']
            communication_score = communication_result['score']
            
            # Step 2: Evaluate sales skills
            sales_skills_result = await self.evaluate_sale_skills(
                audio_bytes,
                task_id,
                trace=trace
            )
            
            if sales_skills_result['status'] != 1:
                if trace:
                    trace.update(
                        output={
                            "status": "failed",
                            "error": "Sales evaluation failed",
                            "message": sales_skills_result['detail_result']
                        }
                    )
                    flush_langfuse()
                
                return {
                    'status': -1,
                    'message': sales_skills_result['detail_result'],
                    'final_detail_result': None,
                    'code': 500
                }
            
            sales_skills_detail = sales_skills_result['detail_result']
            sales_skills_score = sales_skills_result['score']
            
            # Calculate final score
            total_score = communication_score + sales_skills_score
            final_detail_result = (
                f"{communication_detail}\n\n"
                f"II. {sales_skills_detail}\n\n"
                f"═══════════════════════════════════════\n"
                f"TỔNG ĐIỂM CUỐI CÙNG: {total_score:.2f}/7.0\n"
                f"═══════════════════════════════════════"
            )
            
            # Update trace with final result
            if trace:
                trace.update(
                    output={
                        "status": "success",
                        "communication_score": communication_score,
                        "sales_score": sales_skills_score,
                        "total_score": total_score,
                        "final_detail_result_preview": final_detail_result
                    }
                )
                flush_langfuse()
                logger.info(f"✓ Langfuse trace completed: {task_id}")
            
            return {
                'status': 1,
                'message': 'Evaluation completed successfully',
                'final_detail_result': final_detail_result,
                'segments': communication_result['segments'],
                'code': 200
            }
            
        except Exception as e:
            logger.error(f"Error in run_evaluate: {e}")
            
            if trace:
                trace.update(
                    output={
                        "status": "error",
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )
                flush_langfuse()
            
            return {
                'status': -1,
                'message': str(e),
                'final_detail_result': None,
                'segments': [],
                'code': 500
            }