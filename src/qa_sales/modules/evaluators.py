from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from typing import List, Dict, Any
from dotenv import load_dotenv
from ast import literal_eval
import pandas as pd
import os

load_dotenv()


class ScriptEvaluator:
    def __init__(self,
                 model: str = "gpt-4.1-mini",
                 eval_prompt_template: str = "prompt_templates/evaluate_script.txt",
                 classify_prompt_template: str = "prompt_templates/classify_utterances.txt",
                 chroma_db: Chroma = None,
                 tsv_path: str = "data/sale_criteria.tsv"):
        self.llm = ChatOpenAI(model=model,
                              temperature=0.0,
                              api_key=os.getenv("OPENAI_API_KEY"),
                              base_url=os.getenv("BASE_URL"))
        eval_prompt = open(eval_prompt_template).read()
        classify_prompt = open(classify_prompt_template).read()
        self.classify_prompt = PromptTemplate.from_template(classify_prompt)
        self.eval_prompt = PromptTemplate.from_template(eval_prompt)
        df = pd.read_csv(tsv_path, delimiter="\t")
        self.criteria_score = dict(zip(df['criteria_id'], df['criteria_score']))
        self.step_detail = self.from_db_to_text(chroma_db=chroma_db)

    async def classify_utterances_to_criteria(self,
                                              utterances: List[Dict[str, Any]],) -> List[Dict[str, Any]]:

        chain = self.classify_prompt | self.llm
        response = await chain.ainvoke({'sale_texts': utterances,
                                       'step_detail': self.step_detail})

        sale_texts = literal_eval(response.content)
        return sale_texts

    def from_db_to_text(self,
                        chroma_db: Chroma) -> str:
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
            result += f"criteria ID: {criteria_id}\ncriteria Name: {criteria_name}\ncriteria Description: {criteria_description}\ncriteria Actions: {criteria_actions}\n\n"
        return result

    def score_and_response(self,
                           criteria_evals: List[Dict[str, Any]],
                           criteria_score: Dict[int, float]) -> Dict:
        """
        Score the criteria evaluations based on the criteria scores.
        Args:
            criteria_evals (List[Dict[str, Any]]): List of criteria evaluations.
            criteria_score (Dict[int, float]): Dictionary of criteria scores.
        Returns:
            Dict: Scored criteria evaluations.
        """
        for criteria_eval in criteria_evals:
            criteria_id = criteria_eval['criteria_id']
            criteria_eval['score'] = criteria_score.get(criteria_id, 0) * int(criteria_eval['status'])
        return criteria_evals

    async def __call__(self,
                       dialogue: List[Dict[str, Any]]) -> str:
        # try:
        chain = self.eval_prompt | self.llm
        sale_texts = await self.classify_utterances_to_criteria(utterances=dialogue)
        response = await chain.ainvoke({
            "sale_texts": sale_texts,
            "step_detail": self.step_detail
        })
        criteria_evals = literal_eval(response.content)
        criteria_evals = self.score_and_response(criteria_evals, self.criteria_score)
        return {'status': 1,
                'criteria_evals': criteria_evals,
                'message': 'Success'}
        # except Exception as e:
        #    print(f"Error during script evaluation: {e}")
        #    return {'status': -1,
        #            'message': 'Failed to evaluate script'}

