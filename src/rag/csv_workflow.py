"""CSV ingestion workflow"""
from typing import List

import restate
from restate.serde import BytesSerde
import pandas as pd
import io

from langchain_text_splitters import RecursiveCharacterTextSplitter

from . types import NewCsvDocument
from . object_store import get_object_store_client
from . vector_store import get_vector_store
from . embeddings_service import compute_embedding

csv_workflow = restate.Workflow('csv')

def extract_csv_text_snippets(csv_bytes: bytes) -> List[str]:
    """Extract text from CSV"""
    # Read CSV into pandas DataFrame
    df = pd.read_csv(io.BytesIO(csv_bytes))
    
    # Convert each row to a string representation
    texts = []
    for _, row in df.iterrows():
        # Convert row to string, handling different data types appropriately
        row_text = " ".join([f"{col}: {str(val)}" for col, val in row.items()])
        texts.append(row_text)
    
    # Split texts if they are too long
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for text in texts:
        chunks.extend(text_splitter.split_text(text))
    
    return chunks

@csv_workflow.main()
async def process_csv(ctx: restate.WorkflowContext, request: NewCsvDocument):
    """CSV ingestion workflow"""
    #
    # 1. Download the CSV
    #
    async def download_csv() -> bytes:
        object_store = get_object_store_client()
        return await object_store.aget_object(request["bucket_name"], request["object_name"])

    csv_bytes = await ctx.run("Download CSV", download_csv, serde=BytesSerde())

    #
    # 2. Extract the snippets from the CSV
    #
    texts = extract_csv_text_snippets(csv_bytes)

    #
    # 3. Compute embeddings for the text snippets
    #
    vector_futures = [ctx.service_call(compute_embedding, arg=text) for text in texts]
    vectors = [await vector for vector in vector_futures]

    #
    # 4. Add the documents to the vector store
    #
    async def add_documents():
        metadata = { "object_name": request["object_name"], "bucket_name": request["bucket_name"] }
        store = get_vector_store()
        await store.aupsert(texts, vectors, metadata)

    await ctx.run("Add documents", add_documents)

    return "ok" 