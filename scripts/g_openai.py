import os
import time
import random
import pandas as pd
import litellm
import giskard
from dotenv import load_dotenv
from giskard.rag import generate_testset, KnowledgeBase, QATestset
from giskard.rag.question_generators import conversational_questions

load_dotenv()

os.environ["AZURE_API_KEY"] = os.getenv("AZURE_API_KEY")
os.environ["AZURE_API_BASE"] = os.getenv("AZURE_API_BASE")

giskard.llm.set_llm_model("azure/gpt-4o-mini")
giskard.llm.set_embedding_model("azure/GPTVectorization")

df = pd.read_csv("/home/shtlp_0010/Desktop/giskard_dataset/summaryanswer(in).csv")

knowledge_base = KnowledgeBase.from_pandas(df, columns=["summary", "text"])

def retry_with_backoff(func, retries=5):
    for attempt in range(retries):
        try:
            return func()
        except litellm.RateLimitError:
            wait_time = 2 ** attempt + random.uniform(0, 1)
            print(f"Rate limit hit, retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
    raise Exception("Exceeded maximum retries")

testset = retry_with_backoff(lambda: generate_testset(
    knowledge_base=knowledge_base,
    num_questions=100,
    language='en',
    agent_description="An assistant to help understand the summarized texts",
    question_generators=[conversational_questions],
))

testset.save("summary_answer_1.jsonl")
loaded_testset = QATestset.load("summary_answer_1.jsonl")

df = loaded_testset.to_pandas()
print(df.head())
