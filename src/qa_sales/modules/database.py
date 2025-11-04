from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from dotenv import load_dotenv
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
base_url = os.getenv("BASE_URL")
api_key = os.getenv("OPENAI_API_KEY")


def create_csvdatabase(csv_path: str,
                       db_path: str,
                       force_rebuild: bool = False
                       ) -> Chroma:
    if not os.path.exists(db_path) or force_rebuild:
        logger.info("Creating new Chroma database...")
        loader = CSVLoader(file_path=csv_path,
                           metadata_columns=['criteria_id', 'criteria_name', 'criteria_description', 'criteria_actions'],
                           csv_args={'delimiter': '\t'})
        documents = loader.load()
        chroma_db = Chroma.from_documents(documents=documents,
                                          persist_directory=db_path,
                                          embedding=OpenAIEmbeddings(model="text-embedding-3-small",
                                                                     api_key=api_key,
                                                                     base_url=base_url))
        chroma_db.persist()
        return chroma_db

    else:
        logger.info("Loading existing Chroma database...")
        chroma_db = Chroma(persist_directory=db_path)
        return chroma_db

