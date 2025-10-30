from src.qa_communicate.audio_processing.analysis import extract_features
from src.qa_communicate.evaluation.evaluator import get_qa_evaluation
from src.qa_sales.modules.qa_evaluators import QASalesEvaluator
from logging import getLogger

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

    async def evaluate_communication(self,
                                     audio_bytes: bytes,
                                     task_id: int):
        """
        Evaluate the communication quality of a sales call.
        """
        try:
            logger.info("Start analyzing audio features...")
            analysis_result = await extract_features(audio_bytes)
            logger.info("Finished analyzing audio features.")

            data_for_llm = {
                'metadata': analysis_result.get('metadata'),
                'segments': analysis_result.get('segments')
            }
            logger.info("Start evaluating communication quality with LLM...")
            evaluation_result = await get_qa_evaluation(data_for_llm)
            logger.info("Finished evaluating communication quality with LLM.")

            chao_xung_danh = int(evaluation_result.get('chao_xung_danh', 0))
            ky_nang_noi = int(evaluation_result.get('ky_nang_noi', 0))  
            ky_nang_nghe = int(evaluation_result.get('ky_nang_nghe', 0))  
            thai_do = int(evaluation_result.get('thai_do', 0)) 

            # Tính tổng điểm
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

            detail_result = "I. Đánh giá kỹ năng giao tiếp theo các tiêu chí:\n"
            detail_result += f"+ Chào xưng danh: {chao_xung_danh}\n"
            detail_result += f"+ Kỹ năng nói: {ky_nang_noi}\n"
            detail_result += f"+ Kỹ năng nghe: {ky_nang_nghe}\n"
            detail_result += f"+ Thái độ: {thai_do}\n"
            detail_result += f"+ Tổng điểm: {tong_diem}\n"
            detail_result += f"+ Lý do: {ly_do}\n"

            return {'detail_result': detail_result, 'score': tong_diem, 'status': 1}
        except Exception as e:
            logger.error(f"Error evaluating communication: {e}")
            return {'detail_result': str(e), 'score': -1, 'status': -1}

    async def evaluate_sale_skills(self,
                                   audio_bytes: bytes,
                                   task_id: int):
        """
        Evaluate the sales skills in a sales call.
        """
        try:
            result = await self.qa_evaluator.run_evaluate(audio_bytes=audio_bytes,
                                                          task_id=task_id)
            return {'detail_result': result['detail_result'],
                    'score': result['final_score'],
                    'status': 1}
        except Exception as e:
            logger.error(f"Error evaluating sales skills: {e}")
            return {'detail_result': str(e), 'score': -1, 'status': -1}

    async def run_evaluate(self,
                           audio_bytes: bytes,
                           task_id: int):
        """
        Run the full evaluation process: communication quality and sales skills.
        """
        communication_result = await self.evaluate_communication(audio_bytes, task_id)
        if communication_result['status'] != 1:
            return {'status': -1,
                    'message': communication_result['detail_result'],
                    'final_detail_result': None,
                    'code': 500}
        communication_result, communication_score = communication_result['detail_result'], communication_result['score']
        sales_skills_result = await self.evaluate_sale_skills(audio_bytes, task_id)
        if sales_skills_result['status'] != 1:
            return {'status': -1,
                    'message': sales_skills_result['detail_result'],
                    'final_detail_result': None,
                    'code': 500}
        sales_skills_result, sales_skills_score = sales_skills_result['detail_result'], sales_skills_result['score']
        total_score = communication_score + sales_skills_score
        final_detail_result = f"{communication_result}\n{sales_skills_result}\nTổng điểm cuối cùng: {total_score}"
        return {'status': 1,
                'message': 'Evaluation completed successfully',
                'final_detail_result': final_detail_result,
                'code': 200}
