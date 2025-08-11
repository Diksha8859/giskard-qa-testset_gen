import os
import time
import random
import pandas as pd
import litellm
import giskard
from dotenv import load_dotenv
from giskard.rag import generate_testset, KnowledgeBase, QATestset
from giskard.rag.question_generators import conversational_questions
from typing import Optional, List, Callable, Any


class ConfigManager:
    """Manages configuration and environment setup."""
    
    def __init__(self, env_file: str = ".env"):
        self.env_file = env_file
        self._load_environment()
        self._setup_azure_config()
        self._setup_giskard_models()
    
    def _load_environment(self):
        """Load environment variables from .env file."""
        load_dotenv(self.env_file)
    
    def _setup_azure_config(self):
        """Setup Azure API configuration."""
        os.environ["AZURE_API_KEY"] = os.getenv("AZURE_API_KEY")
        os.environ["AZURE_API_BASE"] = os.getenv("AZURE_API_BASE")
    
    def _setup_giskard_models(self):
        """Setup Giskard LLM and embedding models."""
        giskard.llm.set_llm_model("azure/gpt-4o-mini")
        giskard.llm.set_embedding_model("azure/GPTVectorization")


class RetryHandler:
    """Handles retry logic with exponential backoff."""
    
    def __init__(self, max_retries: int = 5):
        self.max_retries = max_retries
    
    def retry_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry logic."""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except litellm.RateLimitError:
                wait_time = 2 ** attempt + random.uniform(0, 1)
                print(f"Rate limit hit, retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
        raise Exception("Exceeded maximum retries")


class KnowledgeBaseManager:
    """Manages knowledge base operations."""
    
    def __init__(self, retry_handler: RetryHandler):
        self.retry_handler = retry_handler
    
    def create_knowledge_base(self, csv_path: str, columns: List[str]) -> KnowledgeBase:
        """Create a knowledge base from CSV data."""
        df = pd.read_csv(csv_path)
        return KnowledgeBase.from_pandas(df, columns=columns)
    
    def generate_testset(
        self,
        knowledge_base: KnowledgeBase,
        num_questions: int = 100,
        language: str = 'en',
        agent_description: str = "An assistant to help understand the summarized texts",
        question_generators: Optional[List] = None
    ) -> QATestset:
        """Generate a testset from the knowledge base."""
        if question_generators is None:
            question_generators = [conversational_questions]
        
        return self.retry_handler.retry_with_backoff(
            generate_testset,
            knowledge_base=knowledge_base,
            num_questions=num_questions,
            language=language,
            agent_description=agent_description,
            question_generators=question_generators
        )


class TestsetManager:
    """Manages testset operations."""
    
    @staticmethod
    def save_testset(testset: QATestset, filename: str):
        """Save testset to file."""
        testset.save(filename)
    
    @staticmethod
    def load_testset(filename: str) -> QATestset:
        """Load testset from file."""
        return QATestset.load(filename)
    
    @staticmethod
    def testset_to_dataframe(testset: QATestset) -> pd.DataFrame:
        """Convert testset to pandas DataFrame."""
        return testset.to_pandas()


class RAGPipeline:
    """Main pipeline for RAG operations."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.retry_handler = RetryHandler()
        self.kb_manager = KnowledgeBaseManager(self.retry_handler)
        self.testset_manager = TestsetManager()
    
    def run_pipeline(
        self,
        csv_path: str,
        columns: List[str],
        num_questions: int = 100,
        output_filename: str = "summary_answer_1.jsonl"
    ) -> pd.DataFrame:
        """Run the complete RAG pipeline."""
        # Create knowledge base
        knowledge_base = self.kb_manager.create_knowledge_base(csv_path, columns)
        
        # Generate testset
        testset = self.kb_manager.generate_testset(
            knowledge_base=knowledge_base,
            num_questions=num_questions
        )
        
        # Save testset
        self.testset_manager.save_testset(testset, output_filename)
        
        # Load and convert to DataFrame
        loaded_testset = self.testset_manager.load_testset(output_filename)
        df = self.testset_manager.testset_to_dataframe(loaded_testset)
        
        return df


def main():
    """Main execution function."""
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # Run pipeline
    csv_path = "../data/summaryanswer(in).csv"
    columns = ["summary", "text"]
    
    try:
        df = pipeline.run_pipeline(
            csv_path=csv_path,
            columns=columns,
            num_questions=100,
            output_filename="summary_answer_1.jsonl"
        )
        
        print("Pipeline completed successfully!")
        print("Generated testset preview:")
        print(df.head())
        
    except Exception as e:
        print(f"Error running pipeline: {e}")


if __name__ == "__main__":
    main()
