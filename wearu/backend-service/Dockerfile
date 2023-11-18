ARG PYTHON_VERSION

FROM python:$PYTHON_VERSION

ARG POETRY_VERSION
ARG PIPX_VERSION
ARG BACKEND_SERVICE_PORT

WORKDIR /src
EXPOSE $BACKEND_SERVICE_PORT
ENV PATH="/root/.local/bin:$PATH"
ENV PORT=$BACKEND_SERVICE_PORT

RUN pip install pipx==$PIPX_VERSION && pipx install poetry==$POETRY_VERSION

COPY pyproject.toml pyproject.toml
RUN poetry install --with backend --no-root

COPY wearu/backend-service .

CMD poetry run uvicorn service:backend_service --host 0.0.0.0 --port $PORT
