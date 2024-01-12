import os
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated, AsyncGenerator, TypedDict

from fastapi import Depends, FastAPI, File, Request
from httpx import AsyncClient
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from starlette.staticfiles import StaticFiles


class Resources(TypedDict):
    database_client: AsyncQdrantClient
    request_client: AsyncClient
    

class Config(TypedDict):
    TEXT_INFERENCE_ENDPOINT: str
    IMAGE_INFERENCE_ENDPOINT: str
    VECTOR_DATABASE_URL: str
    COLLECTION_NAME: str


class TextQuery(BaseModel):
    text: str


class Item(BaseModel):
    item_id: int
    image_path: str
    product_name: str
    product_type_name: str
    product_group_name: str
    colour_group_name: str
    perceived_colour_value_name: str
    perceived_colour_master_name: str
    department_name: str
    index_name: str
    index_group_name: str
    section_name: str
    garment_group_name: str
    description: str


@lru_cache
def get_config() -> Config:
    params = [
        'TEXT_INFERENCE_ENDPOINT',
        'IMAGE_INFERENCE_ENDPOINT',
        'VECTOR_DATABASE_URL',
        'COLLECTION_NAME',
    ]
    config = {}
    for param in params:
        config[param] = os.getenv(param)
    return config


@asynccontextmanager
async def load_resources(app: FastAPI) -> AsyncGenerator[Resources, None]:
    # pylint: disable=unused-argument
    config = get_config()
    database_client = AsyncQdrantClient(url=config['VECTOR_DATABASE_URL'])
    snapshots = await database_client.list_snapshots(config['COLLECTION_NAME'])
    snapshot_location = os.path.join(
        config['VECTOR_DATABASE_URL'],
        'collections',
        config['COLLECTION_NAME'],
        'snapshots',
        snapshots[0].name,
    )
    await database_client.recover_snapshot(
        collection_name=config['COLLECTION_NAME'], location=snapshot_location,
    )
    request_client = AsyncClient()
    yield {
        'database_client': database_client,
        'request_client': request_client,
    }
    await database_client.close()
    await request_client.aclose()


backend_service = FastAPI(lifespan=load_resources)
backend_service.mount(
    '/static', StaticFiles(directory='/static'), name='static',
)


@backend_service.post('/items/{item_id}')
async def get_item(
    item_id: int, request: Request, config: Config = Depends(get_config),
) -> Item:
    items = await request.state.database_client.retrieve(
        collection_name=config['COLLECTION_NAME'], ids=[item_id],
    )
    return Item(item_id=items[0].id, **items[0].payload)


@backend_service.post('/search_text')
async def search_text(
    limit: int,
    offset: int,
    query: TextQuery,
    request: Request,
    config: Config = Depends(get_config),
) -> list[Item]:
    response = await request.state.request_client.post(
        config['TEXT_INFERENCE_ENDPOINT'], json={'text': query.text},
    )

    search_results = await request.state.database_client.search(
        collection_name=config['COLLECTION_NAME'],
        query_vector=('image', response.json()),
        limit=limit,
        offset=offset,
    )
    return [
        Item(item_id=result.id, **result.payload) for result in search_results
    ]


@backend_service.post('/search_image')
async def search_image(
    limit: int,
    offset: int,
    query: Annotated[bytes, File()],
    request: Request,
    config: Config = Depends(get_config),
) -> list[Item]:
    response = await request.state.request_client.post(
        config['IMAGE_INFERENCE_ENDPOINT'], files={'query': query},
    )

    search_results = await request.state.database_client.search(
        collection_name=config['COLLECTION_NAME'],
        query_vector=('image', response.json()),
        limit=limit,
        offset=offset,
    )

    return [
        Item(item_id=result.id, **result.payload) for result in search_results
    ]
