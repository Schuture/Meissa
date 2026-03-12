import re
import base64
from io import BytesIO
import numpy as np
import cv2
from PIL import Image

try:
    from tool_server.utils.utils import *
    from tool_server.utils.server_utils import *
except ImportError:
    pass


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def load_image(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    elif isinstance(image, str):
        import os
        if os.path.exists(image):
            return Image.open(image).convert("RGB")
        else:
            return load_image_from_base64(image)
    raise ValueError(f"Cannot load image from {type(image)}")


def pil_to_base64(image):
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGB")
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

import logging
try:
    logger = build_logger("zoom_in_worker")
except NameError:
    logger = logging.getLogger("zoom_in_worker")

def get_bbox_from_mask(mask_pil):
    """
     Mask 
    """
    mask_np = np.array(mask_pil)
    
    if len(mask_np.shape) > 2:
        mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
    
    _, thresh = cv2.threshold(mask_np, 1, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    max_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_cnt)
    
    return [x, y, x + w, y + h]

def apply_padding(bbox, img_w, img_h, padding_ratio=0.1):
    """
     BBox  Padding
    bbox: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    
    pad_w = int(w * padding_ratio)
    pad_h = int(h * padding_ratio)
    
    new_x1 = max(0, x1 - pad_w)
    new_y1 = max(0, y1 - pad_h)
    new_x2 = min(img_w, x2 + pad_w)
    new_y2 = min(img_h, y2 + pad_h)
    
    return [new_x1, new_y1, new_x2, new_y2]

def generate(params):
    """
    ZoomIn 
    params: {
        "image":  base64,
        "param": /  Mask  base64 
    }
    """
    input_param = params.get("param", "")
    image_b64 = params.get("image", None)
    
    ret = {"text": "", "error_code": 0, "edited_image": None}

    if not image_b64:
        ret["text"] = "Error: Input image is missing."
        ret["error_code"] = 1
        return ret

    try:
        original_image = load_image(image_b64)
        img_w, img_h = original_image.size
        
        target_bbox = None
        
        if isinstance(input_param, list):
            if len(input_param) == 4:
                target_bbox = [float(p) for p in input_param]
            else:
                logger.warning(f"List input length is not 4: {input_param}")
                
        elif isinstance(input_param, str):
            coords_match = re.match(r'\[\s*([\d\.\,\s]+)\s*\]', input_param)
            if coords_match:
                try:
                    coords = list(map(float, coords_match.group(1).split(',')))
                    if len(coords) == 4:
                        target_bbox = coords
                except ValueError:
                    pass
            
            if target_bbox is None:
                try:
                    if len(input_param) > 100: 
                        mask_image = load_image(input_param)
                        target_bbox = get_bbox_from_mask(mask_image)
                        if target_bbox is None:
                            logger.warning("No valid region found in mask.")
                except Exception as e:
                    logger.warning(f"Failed to interpret param as mask: {e}")

        if target_bbox:
            x1, y1, x2, y2 = target_bbox
            
            if all(0 <= c <= 1000 for c in [x1, y1, x2, y2]):
                logger.info(f"Detected normalized coordinates (0-1000): {target_bbox}. Scaling to {img_w}x{img_h}.")
                x1 = x1 / 1000.0 * img_w
                y1 = y1 / 1000.0 * img_h
                x2 = x2 / 1000.0 * img_w
                y2 = y2 / 1000.0 * img_h
                target_bbox = [int(x1), int(y1), int(x2), int(y2)]
            else:
                target_bbox = [int(x1), int(y1), int(x2), int(y2)]

            final_bbox = apply_padding(target_bbox, img_w, img_h, padding_ratio=0.1)
            
            # Crop!
            cropped_image = original_image.crop(final_bbox)
            
            ret["edited_image"] = pil_to_base64(cropped_image)
            ret["text"] = f"Zoomed in successfully to region {final_bbox}."
            logger.info(f"ZoomIn success. Original: {original_image.size}, Crop: {cropped_image.size}")
        else:
            ret["text"] = f"Error: Could not parse parameter as BBox or Mask. Input: {str(input_param)[:50]}..."
            ret["error_code"] = 1

    except Exception as e:
        logger.error(f"ZoomIn failed: {e}")
        import traceback
        traceback.print_exc()
        ret["text"] = f"ZoomIn failed: {e}"
        ret["error_code"] = 1
    
    return ret