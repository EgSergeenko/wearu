from contextlib import asynccontextmanager
from io import BytesIO
from typing import Annotated, AsyncGenerator, TypedDict

from fastapi import FastAPI, File, Request
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


class ImagePipeline(TypedDict):
    model: CLIPVisionModelWithProjection
    processor: CLIPImageProcessor
    max_length: int

@asynccontextmanager
async def load_pipeline(app: FastAPI) -> AsyncGenerator[ImagePipeline, None]:
    # pylint: disable=unused-argument
    model = CLIPVisionModelWithProjection.from_pretrained(
        'patrickjohncyh/fashion-clip',
    )
    processor = CLIPImageProcessor.from_pretrained(
        'patrickjohncyh/fashion-clip',
    )
    yield {'model': model, 'processor': processor}


image_service = FastAPI(lifespan=load_pipeline)


@image_service.post('/compute_image_embedding')
def compute_image_embedding(
    query: Annotated[bytes, File()], request: Request,
) -> list[float]:
    image = Image.open(BytesIO(query))
    inputs = request.state.processor(image, return_tensors='pt')
    outputs = request.state.model(**inputs)
    return outputs.image_embeds.squeeze().tolist()
