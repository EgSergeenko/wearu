import os

import onnxruntime
import torch
from dotenv import load_dotenv
from onnxruntime.transformers import optimizer
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

OPTIMIZATION_LEVEL = 99
ABSOLUTE_TOLERANCE = 1e-5

MODEL_NAME = 'patrickjohncyh/fashion-clip'

VALIDATION_TEXT = 'dummy text input'


def optimize_text_model(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    model = CLIPTextModelWithProjection.from_pretrained(MODEL_NAME).eval()

    input_size = (1, model.config.max_position_embeddings)
    input_ids = torch.ones(input_size, dtype=torch.long)
    attention_mask = torch.ones(input_size, dtype=torch.long)
    position_ids = torch.ones(input_size, dtype=torch.long)

    model_args = (
        input_ids,
        attention_mask,
        position_ids,
        {
            'output_attentions': False,
            'output_hidden_states': False,
            'return_dict': False,
        },
    )

    output_path = os.path.join(output_dir, 'text-fashion-clip.onnx')

    torch.onnx.export(
        model=model,
        args=model_args,
        f=output_path,
        input_names=['input_ids', 'attention_mask', 'position_ids'],
        output_names=['text_embeds'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'input_length'},
            'attention_mask': {0: 'batch_size', 1: 'input_length'},
            'position_ids': {0: 'batch_size', 1: 'input_length'},
            'text_embeds': {0: 'batch_size'},
        },
    )

    onnx_model = optimizer.optimize_model(
        output_path,
        model_type='clip',
        opt_level=OPTIMIZATION_LEVEL,
        num_heads=model.config.num_attention_heads,
        hidden_size=model.config.hidden_size,
    )

    onnx_model.save_model_to_file(output_path)

    validate_text_model(output_path, model)


@torch.no_grad()
def validate_text_model(
    onnx_model_path: str, torch_model: CLIPTextModelWithProjection,
) -> None:
    session = onnxruntime.InferenceSession(onnx_model_path)

    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)

    onnx_inputs = tokenizer([VALIDATION_TEXT], return_tensors='np')
    torch_inputs = tokenizer([VALIDATION_TEXT], return_tensors='pt')

    position_ids = torch.arange(
        torch_inputs.input_ids.size(-1), dtype=torch.long,
    ).unsqueeze(0).numpy()

    onnx_outputs = session.run(
        output_names=['text_embeds'],
        input_feed={
            'input_ids': onnx_inputs.input_ids,
            'attention_mask': onnx_inputs.attention_mask,
            'position_ids': position_ids,
        },
    )[0]

    torch_outputs = torch_model(**torch_inputs)[0]

    assert torch.allclose(
        torch.from_numpy(onnx_outputs), torch_outputs, atol=ABSOLUTE_TOLERANCE,
    ), 'Something went wrong during text model optimization...'


def optimize_image_model(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    model = CLIPVisionModelWithProjection.from_pretrained(MODEL_NAME).eval()

    pixel_values = torch.ones(
        (
            1,
            model.config.num_channels,
            model.config.image_size,
            model.config.image_size,
        ),
    )

    model_args = (
        pixel_values,
        {
            'output_attentions': False,
            'output_hidden_states': False,
            'return_dict': False,
        },
    )

    output_path = os.path.join(output_dir, 'image-fashion-clip.onnx')

    torch.onnx.export(
        model=model,
        args=model_args,
        f=output_path,
        input_names=['pixel_values'],
        output_names=['image_embeds'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'image_embeds': {0: 'batch_size'},
        },
    )

    onnx_model = optimizer.optimize_model(
        output_path,
        model_type='clip',
        opt_level=OPTIMIZATION_LEVEL,
        num_heads=model.config.num_attention_heads,
        hidden_size=model.config.hidden_size,
    )

    onnx_model.save_model_to_file(output_path)

    validate_image_model(output_path, model)


@torch.no_grad()
def validate_image_model(
    onnx_model_path: str, torch_model: CLIPVisionModelWithProjection,
) -> None:
    session = onnxruntime.InferenceSession(onnx_model_path)

    torch_inputs = torch.ones(
        (
            1,
            torch_model.config.num_channels,
            torch_model.config.image_size,
            torch_model.config.image_size,
        ),
    )

    onnx_outputs = session.run(
        output_names=['image_embeds'],
        input_feed={'pixel_values': torch_inputs.numpy()},
    )[0]

    torch_outputs = torch_model(torch_inputs)[0]

    assert torch.allclose(
        torch.from_numpy(onnx_outputs), torch_outputs, atol=ABSOLUTE_TOLERANCE,
    ), 'Something went wrong during image model optimization...'


if __name__ == '__main__':
    load_dotenv()
    model_dir = os.getenv('MODEL_DIR')
    optimize_text_model(model_dir)
    optimize_image_model(model_dir)
