import gc
import logging
import os
import uuid
import base64
from io import BytesIO

import torch


logger = logging.getLogger(__name__)


class Utils:
    def __init__(self, host: str = "0.0.0.0", port: int = 8500):
        env_url = os.getenv("SERVICE_URL")
        if env_url:
            self.service_url = env_url.rstrip("/")
        else:
            # Avoid returning 0.0.0.0 in URLs; prefer loopback if bound to all interfaces
            url_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
            self.service_url = f"http://{url_host}:{port}"
        # Use current working directory instead of system temp directory
        current_dir = os.getcwd()
        self.image_dir = os.path.join(current_dir, "images")
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir, mode=0o755)

        self.video_dir = os.path.join(current_dir, "videos")
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir, mode=0o755)

    def parse_size(self, size: str | None, default: tuple[int, int] = (1024, 1024)) -> tuple[int, int]:
        """Parse an OpenAI-style size string like "1024x1024" into (width, height).
        Fallbacks to default if None or invalid.
        """
        if not size:
            return default
        try:
            parts = size.lower().split("x")
            if len(parts) != 2:
                return default
            w = int(parts[0].strip())
            h = int(parts[1].strip())
            if w <= 0 or h <= 0:
                return default
            return (w, h)
        except Exception:
            return default

    def image_to_b64(self, image) -> str:
        """Convert a PIL image or torch Tensor to base64-encoded PNG string."""
        if hasattr(image, "to"):
            try:
                image = image.to("cpu")
            except Exception:
                pass

        if isinstance(image, torch.Tensor):
            from torchvision import transforms

            to_pil = transforms.ToPILImage()
            image = to_pil(image.squeeze(0).clamp(0, 1))

        buf = BytesIO()
        image.save(buf, format="PNG", optimize=True)
        data = buf.getvalue()
        buf.close()
        return base64.b64encode(data).decode("utf-8")

    def save_image(self, image):
        if hasattr(image, "to"):
            try:
                image = image.to("cpu")
            except Exception:
                pass

        if isinstance(image, torch.Tensor):
            from torchvision import transforms

            to_pil = transforms.ToPILImage()
            image = to_pil(image.squeeze(0).clamp(0, 1))

        filename = "img" + str(uuid.uuid4()).split("-")[0] + ".png"
        image_path = os.path.join(self.image_dir, filename)
        logger.info(f"Saving image to {image_path}")

        image.save(image_path, format="PNG", optimize=True)

        del image
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return os.path.join(self.service_url, "images", filename)
