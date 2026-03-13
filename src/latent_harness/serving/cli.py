from __future__ import annotations

import argparse

import uvicorn

from latent_harness.serving.openai_server import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a harness model behind an OpenAI-compatible API.")
    parser.add_argument("--config", required=True, help="Evaluation config containing the named model.")
    parser.add_argument("--model-name", required=True, help="Model spec name to serve.")
    parser.add_argument(
        "--external-model-id",
        default=None,
        help="Model ID to report through the OpenAI API, e.g. openai/codi-gpt2-local.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the API server to.")
    parser.add_argument("--port", type=int, default=8101, help="Port for the API server.")
    parser.add_argument("--device", default="auto", help="Torch device to use, e.g. auto or cuda:0.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    app = create_app(
        config_path=args.config,
        model_name=args.model_name,
        device_name=args.device,
        external_model_id=args.external_model_id,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
