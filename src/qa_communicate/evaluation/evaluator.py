from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from typing import List, Dict, Any
from dotenv import load_dotenv
from ast import literal_eval
import pandas as pd
import json
import os
import logging

# ======= Logger setup =======
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# ============================

load_dotenv()


class ScriptEvaluator:
    def __init__(self,
                 model: str = "gpt-4.1-mini",
                 eval_prompt_template: str = "prompt_templates/evaluate_script.txt",
                 classify_prompt_template: str = "prompt_templates/classify_utterances.txt",
                 chroma_db: Chroma = None,
                 tsv_path: str = "data/sale_criteria.tsv"):
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("BASE_URL")
        )

        eval_prompt = open(eval_prompt_template).read()
        classify_prompt = open(classify_prompt_template).read()
        self.classify_prompt = PromptTemplate.from_template(classify_prompt)
        self.eval_prompt = PromptTemplate.from_template(eval_prompt)

        df = pd.read_csv(tsv_path, delimiter="\t")
        self.criteria_score = dict(zip(df['criteria_id'], df['criteria_score']))

        self.step_detail = self.from_db_to_text(chroma_db=chroma_db)

    # ======================================================
    # 1️⃣ Classify utterances into criteria
    # ======================================================
    def classify_utterances_to_criteria(self,
                                        utterances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chain = self.classify_prompt | self.llm
        response = chain.invoke({'sale_texts': utterances,
                                 'step_detail': self.step_detail})

        logger.info("=== [LLM classify response content] ===\n%s\n===============================", response.content)

        sale_texts = None
        parse_error = None
        for parse_fn in (json.loads, literal_eval):
            try:
                sale_texts = parse_fn(response.content)
                break
            except Exception as e:
                parse_error = e
                logger.debug("Parser %s failed: %s", parse_fn.__name__, e)

        if sale_texts is None:
            logger.error("❌ Không parse được classify response.\nNội dung:\n%s\nLỗi: %s",
                         response.content, parse_error, exc_info=True)
            raise RuntimeError(f"Failed to parse classify response: {parse_error}")

        return sale_texts

    # ======================================================
    # 2️⃣ Convert database to textual representation
    # ======================================================
    def from_db_to_text(self, chroma_db: Chroma) -> str:
        """
        Convert the Chroma database to a text representation.
        """
        result = ""
        all_metadatas = chroma_db.get(include=["metadatas"])
        for metadata in all_metadatas['metadatas']:
            criteria_id = metadata['criteria_id']
            criteria_name = metadata['criteria_name']
            criteria_actions = metadata['criteria_actions']
            criteria_description = metadata['criteria_description']
            result += (
                f"criteria ID: {criteria_id}\n"
                f"criteria Name: {criteria_name}\n"
                f"criteria Description: {criteria_description}\n"
                f"criteria Actions: {criteria_actions}\n\n"
            )
        return result

    # ======================================================
    # 3️⃣ Scoring function
    # ======================================================
    def score_and_response(self,
                           criteria_evals: List[Dict[str, Any]],
                           criteria_score: Dict[int, float]) -> Dict:
        """
        Score the criteria evaluations based on the criteria scores.
        """
        for criteria_eval in criteria_evals:
            criteria_id = criteria_eval['criteria_id']
            criteria_eval['score'] = criteria_score.get(criteria_id, 0) * int(criteria_eval['status'])
        return criteria_evals

    # ======================================================
    # 4️⃣ Main evaluation pipeline
    # ======================================================
    def __call__(self, dialogue: List[Dict[str, Any]]) -> str:
        chain = self.eval_prompt | self.llm
        sale_texts = self.classify_utterances_to_criteria(utterances=dialogue)
        response = chain.invoke({
            "sale_texts": sale_texts,
            "step_detail": self.step_detail
        })

        criteria_evals = literal_eval(response.content)
        criteria_evals = self.score_and_response(criteria_evals, self.criteria_score)
        return {'status': 1,
                'criteria_evals': criteria_evals,
                'message': 'Success'}
