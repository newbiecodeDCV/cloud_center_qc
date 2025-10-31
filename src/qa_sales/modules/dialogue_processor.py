from typing import List, Dict, Any
from litellm import completion, acompletion
from dotenv import load_dotenv
from logging import basicConfig, getLogger, INFO
import os
import ast
import openai

basicConfig(level=INFO)
logger = getLogger(__name__)


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("BASE_URL")
print(f"Using OpenAI API key: {api_key}, Base URL: {base_url}")
openai.api_key = api_key
openai.base_url = base_url


class DialogueProcessor:

    async def extract_speaker_roles(self,
                                    prompt_template: str,
                                    dialogue: List[Dict[str, Any]]):
        """Extract speaker roles from the dialogue.

        Args:
            dialogue (List[Dict[str, Any]]): The dialogue to process.
            [{"speaker": "0", "text": "Alo em chao anh a"}, ...]

        Returns:
            the dialogue with speaker roles assigned.
            [{"speaker": "nhan vien sale", "text": "Alo em chao anh a"}, ...]
        """
        # try:
        prompt = open(prompt_template).read().format(dialogue=dialogue)
        messages = [{"role": "user", "content": prompt}]
        
        response = await acompletion(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.0,
            api_key=api_key,
            base_url=base_url,
        )

        # ==========================
        # 🔹 Parse kết quả structured
        # ==========================
        dialogue = response.choices[0].message.content
        dialogue = ast.literal_eval(dialogue)
        return {
            "status": 1,
            "dialogue": dialogue,
            "tokens": response.usage.total_tokens,
            "message": "Success"
        }
        # except Exception as e:
        #     logger.error(f"Error during extract speaker role: {e}")
        #     return {
        #         "status": -1,
        #         "dialogue": [],
        #         "tokens": 0,
        #         "message": "Failed to extract speaker roles"
        #     }

    async def __call__(self,
                       prompt_template: str,
                       dialogue: List[Dict[str, Any]]):
        result = await self.extract_speaker_roles(prompt_template=prompt_template,
                                                  dialogue=dialogue)
        return result


