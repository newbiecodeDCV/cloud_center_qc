import ast
import os

from typing import List, Dict, Any
from loguru import logger
from dotenv import load_dotenv
from litellm import acompletion

from src.qa_communicate.core.langfuse_config import log_generation, LANGFUSE_ENABLED


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("BASE_URL")


class DialogueProcessor:

    async def extract_speaker_roles(
        self,
        prompt_template: str,
        dialogue: List[Dict[str, Any]],
        trace=None,
        parent_span=None,
    ):
        """
        Extract speaker roles from dialogue with Langfuse tracing.

        Args:
            dialogue: List of dialogue segments
            [{"speaker": "0", "text": "Alo em chao anh a"}, ...]

        Returns:
            Dialogue with speaker roles assigned
            [{"speaker": "nhan vien sale", "text": "Alo em chao anh a"}, ...]
        """
        try:
            # Build prompt
            if parent_span:
                prompt_build_span = parent_span.span(
                    name="build_speaker_prompt", input={"segments": dialogue}
                )

            prompt = open(prompt_template).read().format(dialogue=dialogue)

            if parent_span:
                prompt_build_span.end(output={"prompt": prompt})

            # Call LLM
            messages = [{"role": "user", "content": prompt}]

            logger.info("Calling LLM for speaker role extraction...")

            response = await acompletion(
                model="gpt-4.1-mini",
                messages=messages,
                temperature=0.0,
                api_key=api_key,
                base_url=base_url,
            )

            # Log to Langfuse
            if trace and LANGFUSE_ENABLED:
                log_generation(
                    trace=trace,
                    name="llm_speaker_identification",
                    model="gpt-4.1-mini",
                    input_data={
                        "messages": messages,
                    },
                    output_data=response.choices[0].message.content,
                    metadata={
                        "step": "speaker_role_extraction",
                        "prompt_template": prompt_template,
                    },
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                )

            # Parse result
            if parent_span:
                parse_span = parent_span.span(
                    name="parse_speaker_response",
                    input={"response": response.choices[0].message.content},
                )

            dialogue = response.choices[0].message.content
            dialogue = ast.literal_eval(dialogue)

            if parent_span:
                parse_span.end(
                    output={
                        "status": "success",
                        "sales_segments": [
                            s
                            for s in dialogue
                            if "sale" in s.get("speaker", "").lower()
                        ],
                    }
                )

            logger.info(
                f"✓ Speaker identification completed. Total segments: {len(dialogue)}"
            )

            return {
                "status": 1,
                "dialogue": dialogue,
                "tokens": response.usage.total_tokens,
                "message": "Success",
            }

        except Exception as e:
            logger.error(f"Error during speaker role extraction: {e}", exc_info=True)

            if parent_span:
                error_span = parent_span.span(
                    name="error_handling", input={"error": str(e)}
                )
                error_span.end(output={"status": "failed"})

            return {
                "status": -1,
                "dialogue": [],
                "tokens": 0,
                "message": f"Failed to extract speaker roles: {str(e)}",
            }

    async def __call__(
        self,
        prompt_template: str,
        dialogue: List[Dict[str, Any]],
        trace=None,
        parent_span=None,
    ):
        """Process dialogue and extract speaker roles."""
        result = await self.extract_speaker_roles(
            prompt_template=prompt_template,
            dialogue=dialogue,
            trace=trace,
            parent_span=parent_span,
        )
        return result
