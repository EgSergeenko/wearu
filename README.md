# WearU backend service

### Run service

* `docker compose up`

### Optimize models

1.`poetry install --with inference,image`

2.`poetry run python optimize_model.py`

By default, the optimized models saved at [./fashion-clip](./fashion-clip).
You can modify `MODEL_DIR` in [.env](.env) to change the destination.
