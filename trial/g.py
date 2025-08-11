import os
import time
from dotenv import load_dotenv
import pandas as pd
import fitz  # PyMuPDF
import giskard  # type: ignore
import nest_asyncio  # type: ignore
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from giskard.rag import generate_testset, KnowledgeBase
from giskard.rag.question_generators import conversational_questions

load_dotenv()
# -------------------- Initialization -------------------- #
def initialize_environment():
    nest_asyncio.apply()
    os.environ["AZURE_API_KEY"] = os.getenv("AZURE_API_KEY")
    os.environ["AZURE_API_BASE"] = os.getenv("AZURE_API_BASE")

    giskard.llm.set_llm_model("azure/gpt-4o-mini")
    giskard.llm.set_embedding_model("azure/GPTVectorization")

# -------------------- PDF Text Extraction -------------------- #
def extract_text_from_pdf(pdf_path: str) -> list[str]:
    doc = fitz.open(pdf_path)
    return [page.get_text().strip() for page in doc if page.get_text().strip()]

# -------------------- Summary and Agent Description -------------------- #
def get_pdf_summary_text(pdf_texts: list[str]) -> list[str]:
    meaningful_chunks = []
    for page in pdf_texts:
        cleaned = page.strip()
        if len(cleaned) > 100:
            meaningful_chunks.append(cleaned[:500])
        if len(meaningful_chunks) >= 5:
            break
    return meaningful_chunks or ["General topics related to the document."]

def create_agent_description(summary_chunks: list[str]) -> str:
    if not summary_chunks:
        return (
            "An AI assistant trained to provide helpful answers based on the content of a document. "
            "The topics covered are diverse and informative."
        )
    
    # Join top summary snippets into a readable topic overview
    joined_summary = " ".join(chunk.strip().replace("\n", " ") for chunk in summary_chunks)
    preview = joined_summary[:300].rsplit(".", 1)[0] + "." if "." in joined_summary[:300] else joined_summary[:300]

    return (
        "This AI assistant is designed to answer questions based on the content of a specific document. "
        "The document primarily discusses topics such as: "
        f"{preview} "
        "The assistant provides concise and context-aware responses to enhance user understanding."
    )
# -------------------- Retryable Testset Generator -------------------- #
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=5, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def safe_generate_testset(kb: KnowledgeBase, num_questions: int, agent_description: str):
    return generate_testset(
        kb,
        num_questions=num_questions,
        language="en",
        agent_description=agent_description,
        question_generators=[conversational_questions]
    )

# -------------------- Batch Processing -------------------- #
def process_batches(
    df_knowledge: pd.DataFrame,
    batch_size: int,
    questions_per_batch: int,
    agent_description: str,
    sleep_time: int = 10
) -> list[pd.DataFrame]:
    all_dataframes = []

    for i in range(0, len(df_knowledge), batch_size):
        print(f"\nProcessing batch {i // batch_size + 1}")
        batch_df = df_knowledge.iloc[i:i + batch_size]
        kb = KnowledgeBase.from_pandas(batch_df, columns=["content"])

        try:
            testset = safe_generate_testset(kb, num_questions=questions_per_batch, agent_description=agent_description)
            all_dataframes.append(testset.to_pandas())
        except Exception as e:
            print(f"Failed batch {i // batch_size + 1}: {e}")

        time.sleep(sleep_time)

    return all_dataframes

# -------------------- Save Output -------------------- #
def save_testset(all_dataframes: list[pd.DataFrame], output_path: str):
    if all_dataframes:
        final_df = pd.concat(all_dataframes, ignore_index=True)
        final_df.to_json(output_path, orient="records", indent=2)
        print(f"\nTestset saved to: {output_path}")
    else:
        print("No data was generated.")

# -------------------- Main Execution -------------------- #
def main():
    initialize_environment()

    pdf_path = "/home/shtlp_0010/Desktop/giskard_dataset/test.pdf"
    output_path = "/home/shtlp_0010/Desktop/giskard_dataset/generat.json"

    print("Extracting text from PDF...")
    pdf_texts = extract_text_from_pdf(pdf_path)
    df_knowledge = pd.DataFrame(pdf_texts, columns=["content"])

    summary_chunks = get_pdf_summary_text(pdf_texts)
    agent_description = create_agent_description(summary_chunks)

    print(f"\nGenerated Agent Description:\n{agent_description}\n")

    print("Generating test set from knowledge base...")
    all_dataframes = process_batches(
        df_knowledge,
        batch_size=20,
        questions_per_batch=1,
        agent_description=agent_description
    )
    save_testset(all_dataframes, output_path)

if __name__ == "__main__":
    main()
