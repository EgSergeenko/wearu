ARG PYTHON_VERSION

FROM python:$PYTHON_VERSION

ARG POETRY_VERSION
ARG PIPX_VERSION
ARG IMAGE_INFERENCE_SERVICE_PORT

WORKDIR /src
EXPOSE $IMAGE_INFERENCE_SERVICE_PORT
ENV PATH="/root/.local/bin:$PATH"
ENV PORT=$IMAGE_INFERENCE_SERVICE_PORT

RUN pip install pipx==$PIPX_VERSION && pipx install poetry==$POETRY_VERSION

COPY pyproject.toml pyproject.toml
RUN poetry install --with inference,image --no-root

COPY wearu/image-inference-service .

CMD poetry run uvicorn service:image_service --host 0.0.0.0 --port $PORT
