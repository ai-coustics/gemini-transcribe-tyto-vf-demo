"""Modal deployment for the Gemini x Quail comparison demo.

    modal deploy modal_app.py

Modal has no built-in per-IP rate limiting, so limits live in app/limits.py.
That limiter keeps per-process state, which is only accurate while a single
container serves every request - hence max_containers=1 below. Raise it only
after moving the limiter to a modal.Dict.
"""

import modal

app = modal.App("aic-demos")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_dir("app", "/root/app")
    .add_local_dir("static", "/root/static")
)

# Create with:
#   modal secret create aic-demo-secrets GOOGLE_API_KEY=... AIC_SDK_LICENSE=...
secrets = [modal.Secret.from_name("aic-demo-secrets")]

# Quail and Tyto weights download on first use; a volume keeps them warm.
models = modal.Volume.from_name("aic-demo-models", create_if_missing=True)


@app.function(
    image=image,
    secrets=secrets,
    volumes={"/root/models": models},
    cpu=4,
    memory=8192,
    max_containers=1,      # keeps the in-process rate limiter authoritative
    scaledown_window=300,
    timeout=900,
)
@modal.concurrent(max_inputs=8)  # websockets idle a lot; compare() gates real work
@modal.asgi_app()
def web():
    from app.main import app as fastapi_app

    return fastapi_app
