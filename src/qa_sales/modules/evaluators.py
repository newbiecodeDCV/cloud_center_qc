"""
Script Evaluator - Detailed Sales Script Evaluation with Langfuse Tracing
File: src/qa_sales/modules/evaluators.py
"""

import json
import os
from ast import literal_eval
from typing import List, Dict, Any
from litellm import acompletion


import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from src.qa_communicate.core.langfuse_config import log_generation, LANGFUSE_ENABLED


class DebugHandler(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        logger.info("==== RAW RESPONSE FROM OPENAI ====")
        logger.info(response)
        logger.info("======================")


load_dotenv()


class ScriptEvaluator:
    """
    Evaluate sales script against predefined criteria with Langfuse tracing.

    Pipeline:
    1. Classify utterances into criteria
    2. Evaluate each criterion
    3. Calculate scores
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        eval_prompt_template: str = "prompt_templates/evaluate_script.txt",
        classify_prompt_template: str = "prompt_templates/classify_utterances.txt",
        chroma_db: Chroma = None,
        tsv_path: str = "data/sale_criteria.tsv",
    ):
        """
        Initialize ScriptEvaluator.

        Args:
            model: LLM model name
            eval_prompt_template: Path to evaluation prompt template
            classify_prompt_template: Path to classification prompt template
            chroma_db: Chroma vector database with criteria
            tsv_path: Path to TSV file with criteria scores
        """
        logger.info(f"Initializing ScriptEvaluator with model: {model}")
        # Load prompt templates
        with open(eval_prompt_template, "r", encoding="utf-8") as f:
            eval_prompt = f.read()
        with open(classify_prompt_template, "r", encoding="utf-8") as f:
            classify_prompt = f.read()

        self.classify_prompt = classify_prompt
        self.eval_prompt = eval_prompt

        # Load criteria scores from TSV
        df = pd.read_csv(tsv_path, delimiter="\t")
        self.criteria_score = dict(zip(df["criteria_id"], df["criteria_score"]))

        # Convert Chroma DB to text representation
        self.step_detail = self.from_db_to_text(chroma_db=chroma_db)

        self.model = model

        logger.info(
            f"✓ ScriptEvaluator initialized with {len(self.criteria_score)} criteria"
        )

    async def classify_utterances_to_criteria(
        self, utterances: List[Dict[str, Any]], trace=None, parent_span=None
    ) -> List[Dict[str, Any]]:
        """
        Classify sales utterances into criteria with Langfuse tracing.

        Args:
            utterances: List of sales utterances to classify
            trace: Langfuse trace object (optional)
            parent_span: Parent span for nested tracking (optional)

        Returns:
            List of dictionaries mapping criteria to utterances:
            [
                {
                    "criteria_id": 1,
                    "utterance": ["text1", "text2", ...]
                },
                ...
            ]
        """
        logger.info(f"Classifying {len(utterances)} utterances into criteria...")

        if parent_span:
            classify_span = parent_span.span(
                name="classify_utterances",
                input={
                    "num_utterances": len(utterances),
                    "num_criteria": len(self.criteria_score),
                },
            )

        # Build prompt từ template
        prompt_text = self.classify_prompt.format(
            sale_texts=utterances, step_detail=self.step_detail
        )

        messages = [{"role": "user", "content": prompt_text}]
        response = await acompletion(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("BASE_URL"),
        )
        response_content = response.choices[0].message.content

        logger.info("=== LLM classify response received ===")
        logger.info(response_content)
        logger.info("======================================")

        # Log to Langfuse - LOG ĐÚNG PROMPT VÀ RESPONSE
        if trace and LANGFUSE_ENABLED:
            token_usage = response.response_metadata.get("token_usage", {})
            log_generation(
                trace=trace,
                name="llm_classify_utterances",
                model=self.model,
                input_data=prompt_text,
                # ✅ Input = Prompt thực tế
                output_data=response_content,  # ✅ Output = Response từ LLM
                metadata={
                    "step": "classify_utterances_to_criteria",
                    "num_utterances": len(utterances),
                    "num_criteria": len(self.criteria_score),
                    "temperature": 0.0,
                },
                usage={
                    "prompt_tokens": token_usage.get("prompt_tokens", 0),
                    "completion_tokens": token_usage.get("completion_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0),
                },
            )

        # Parse response - try multiple parsers
        sale_texts = None
        parse_error = None

        for parse_fn in (json.loads, literal_eval):
            try:
                sale_texts = parse_fn(response_content)
                logger.info(f"✓ Successfully parsed with {parse_fn.__name__}")
                break
            except Exception as e:
                parse_error = e
                logger.debug(f"Parser {parse_fn.__name__} failed: {e}")

        if sale_texts is None:
            error_msg = f"Failed to parse classify response: {parse_error}"
            logger.error(error_msg)
            logger.error(f"Raw response:\n{response_content}")

            if parent_span:
                classify_span.end(
                    output={
                        "status": "error",
                        "error": str(parse_error),
                        "raw_response_preview": response_content[:200],
                    }
                )

            raise RuntimeError(error_msg)

        if parent_span:
            classify_span.end(
                output={
                    "status": "success",
                    "num_classified_groups": len(sale_texts),
                    "criteria_ids": [item.get("criteria_id") for item in sale_texts],
                }
            )

        logger.info(f"✓ Classified into {len(sale_texts)} criteria groups")

        return sale_texts

    def from_db_to_text(self, chroma_db: Chroma) -> str:
        """
        Convert Chroma database to text representation.

        Args:
            chroma_db: Chroma vector database

        Returns:
            String containing all criteria information
        """
        logger.info("Converting Chroma DB to text representation...")

        result = ""
        all_metadatas = chroma_db.get(include=["metadatas"])

        for idx, metadata in enumerate(all_metadatas["metadatas"], 1):
            criteria_id = metadata.get("criteria_id", "N/A")
            criteria_name = metadata.get("criteria_name", "N/A")
            criteria_actions = metadata.get("criteria_actions", "N/A")
            criteria_description = metadata.get("criteria_description", "N/A")

            result += (
                f"criteria ID: {criteria_id}\n"
                f"criteria Name: {criteria_name}\n"
                f"criteria Description: {criteria_description}\n"
                f"criteria Actions: {criteria_actions}\n\n"
            )

        logger.info(f"✓ Converted {len(all_metadatas['metadatas'])} criteria to text")

        return result

    def score_and_response(
        self, criteria_evals: List[Dict[str, Any]], criteria_score: Dict[int, float]
    ) -> List[Dict[str, Any]]:
        """
        Calculate scores for each criterion evaluation.

        Args:
            criteria_evals: List of criterion evaluations with status
            criteria_score: Dictionary mapping criteria_id to max score

        Returns:
            Updated criteria_evals with scores added
        """
        logger.info("Calculating scores for criteria evaluations...")

        for criteria_eval in criteria_evals:
            criteria_id = criteria_eval["criteria_id"]
            status = int(criteria_eval.get("status", 0))
            max_score = criteria_score.get(criteria_id, 0)

            # Score = max_score * status (1 or 0)
            criteria_eval["score"] = max_score * status

            logger.debug(
                f"Criteria {criteria_id}: status={status}, "
                f"max_score={max_score}, final_score={criteria_eval['score']}"
            )

        total_score = sum(c["score"] for c in criteria_evals)
        logger.info(f"✓ Total score calculated: {total_score:.2f}")

        return criteria_evals

    async def __call__(
        self, dialogue: List[Dict[str, Any]], trace=None, parent_span=None
    ) -> Dict:
        """
        Main evaluation pipeline with Langfuse tracing.

        Steps:
        1. Classify utterances to criteria
        2. Evaluate each criterion
        3. Calculate scores

        Args:
            dialogue: List of dialogue segments (sales utterances)
            trace: Langfuse trace object (optional)
            parent_span: Parent span for nested tracking (optional)

        Returns:
            Dictionary with:
            {
                'status': 1 (success) or -1 (failed),
                'criteria_evals': List of criterion evaluations,
                'message': Status message
            }
        """
        logger.info("=" * 60)
        logger.info("Starting sales script evaluation pipeline")
        logger.info("=" * 60)

        # ==========================================
        # Step 1: Classify Utterances
        # ==========================================
        logger.info("Step 1: Classifying utterances to criteria...")

        sale_texts = await self.classify_utterances_to_criteria(
            utterances=dialogue, trace=trace, parent_span=parent_span
        )

        # ==========================================
        # Step 2: Evaluate Each Criterion
        # ==========================================
        if parent_span:
            eval_span = parent_span.span(
                name="evaluate_criteria",
                input={"num_classified_groups": len(sale_texts)},
            )

        logger.info("Step 2: Evaluating each criterion...")
        self.eval_prompt = self.eval_prompt.format(
            sale_texts=sale_texts, step_detail=self.step_detail
        )
        messages = [{"role": "user", "content": self.eval_prompt}]

        response = await acompletion(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("BASE_URL"),
        )
        response_content = response.choices[0].message.content

        # Log to Langfuse
        if trace and LANGFUSE_ENABLED:
            token_usage = response.response_metadata.get("token_usage", {})
            log_generation(
                trace=trace,
                name="llm_evaluate_criteria",
                model=self.model,
                input_data={
                    "num_criteria_groups": len(sale_texts),
                    "step_detail_length": len(self.step_detail),
                },
                output_data=response_content,
                metadata={
                    "step": "evaluate_sales_criteria",
                    "evaluation_type": "sales_script",
                    "temperature": 0.0,
                },
                usage={
                    "prompt_tokens": token_usage.get("prompt_tokens", 0),
                    "completion_tokens": token_usage.get("completion_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0),
                },
            )

        logger.info(
            f"LLM evaluation response (first 500 chars):\n{response_content[:500]}"
        )

        # Parse evaluation results
        try:
            criteria_evals = literal_eval(response_content)

            if parent_span:
                eval_span.end(
                    output={
                        "status": "success",
                        "num_criteria_evaluated": len(criteria_evals),
                    }
                )

            logger.info(f"✓ Successfully evaluated {len(criteria_evals)} criteria")

        except (ValueError, SyntaxError) as parse_err:
            error_msg = f"Failed to parse evaluation response: {parse_err}"
            logger.error(error_msg)
            logger.error(f"Raw response:\n{response_content}")

            if parent_span:
                eval_span.end(
                    output={
                        "status": "error",
                        "error": str(parse_err),
                        "raw_response_preview": response_content[:200],
                    }
                )

            return {"status": -1, "criteria_evals": [], "message": error_msg}

            # ==========================================
            # Step 3: Calculate Scores
            # ==========================================
        logger.info("Step 3: Calculating scores...")

        criteria_evals = self.score_and_response(
            criteria_evals, self.criteria_score
        )

        total_score = sum(c["score"] for c in criteria_evals)

        logger.info("=" * 60)
        logger.info(f"✓ Script evaluation completed successfully")
        logger.info(f"✓ Total score: {total_score:.2f}")
        logger.info("=" * 60)

        return {"status": 1, "criteria_evals": criteria_evals, "message": "Success"}
