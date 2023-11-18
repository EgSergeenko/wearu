from contextlib import asynccontextmanager
from typing import AsyncGenerator, TypedDict

from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import CLIPTextModelWithProjection, CLIPTokenizer


class Query(BaseModel):
    text: str


class TextPipeline(TypedDict):
    model: CLIPTextModelWithProjection
    tokenizer: CLIPTokenizer
    max_length: int


@asynccontextmanager
async def load_pipeline(app: FastAPI) -> AsyncGenerator[TextPipeline, None]:
    # pylint: disable=unused-argument
    model = CLIPTextModelWithProjection.from_pretrained(
        'patrickjohncyh/fashion-clip',
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        'patrickjohncyh/fashion-clip', use_fast=True,
    )
    yield {
        'model': model,
        'tokenizer': tokenizer,
        'max_length': model.config.max_position_embeddings,
    }


text_service = FastAPI(lifespan=load_pipeline)


@text_service.post('/compute_text_embedding')
def compute_text_embedding(query: Query, request: Request) -> list[float]:
    truncation = min(len(query.text), request.state.max_length)
    inputs = request.state.tokenizer(
        query.text[:truncation], return_tensors='pt',
    )
    outputs = request.state.model(**inputs)
    return outputs.text_embeds.squeeze().tolist()
