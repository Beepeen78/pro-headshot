"""
Pro AI Headshot Generator - Public Access Version
Transforms any selfie into professional headshots using AI.
"""
import cv2
import torch
import random
import numpy as np
import os
import time
import glob
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime, timedelta

import spaces
import PIL
from PIL import Image

import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from huggingface_hub import hf_hub_download
from insightface.app import FaceAnalysis

from style_template import styles
from pipeline_stable_diffusion_xl_instantid_full import (
    StableDiffusionXLInstantIDPipeline,
    draw_kps,
)
import gradio as gr

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import torch.nn.functional as F
from torchvision.transforms import Compose

from config import (
    TEMP_DIR,
    MAX_FILE_AGE_HOURS,
    MAX_IMAGE_SIZE_MB,
    ALLOWED_IMAGE_FORMATS,
    MIN_IMAGE_DIMENSION,
    MAX_IMAGE_DIMENSION,
    MAX_PROMPT_LENGTH,
    MAX_NEGATIVE_PROMPT_LENGTH,
    DEFAULT_NUM_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_SEED,
)


# ============ GLOBAL CONFIG ============
MAX_SEED = np.iinfo(np.int32).max

# Device detection - fixed logic
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
STYLE_NAMES = list(styles.keys())


# ============ FILE MANAGEMENT ============
def cleanup_old_files():
    """Remove files older than MAX_FILE_AGE_HOURS from temp directory."""
    if not TEMP_DIR.exists():
        return
    
    cutoff_time = time.time() - (MAX_FILE_AGE_HOURS * 3600)
    deleted_count = 0
    
    for file_path in TEMP_DIR.glob("*"):
        if file_path.is_file():
            try:
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
            except Exception as e:
                print(f"Warning: Could not delete {file_path}: {e}")
    
    if deleted_count > 0:
        print(f"Cleaned up {deleted_count} old files from temp directory")


def save_as_png(image: Image.Image, filename: str = "professional_headshot") -> str:
    """Save image as PNG in temp directory with cleanup."""
    # Cleanup old files before saving new ones
    cleanup_old_files()
    
    TEMP_DIR.mkdir(exist_ok=True, parents=True)
    timestamp = int(time.time())
    filepath = TEMP_DIR / f"{filename}_{timestamp}.png"

    # Ensure image is in RGB mode
    if image.mode in ("RGBA", "LA"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    image.save(filepath, "PNG", optimize=True)
    return str(filepath)


# ============ INPUT VALIDATION ============
def validate_image_file(file_path: str) -> Tuple[bool, Optional[str]]:
    """Validate uploaded image file."""
    if not file_path:
        return False, "Please upload an image file."
    
    # Check file exists
    if not os.path.exists(file_path):
        return False, "Uploaded file not found. Please try again."
    
    # Check file extension
    ext = Path(file_path).suffix.lower()
    if ext not in ALLOWED_IMAGE_FORMATS:
        return False, f"Unsupported file format. Allowed formats: {', '.join(ALLOWED_IMAGE_FORMATS)}"
    
    # Check file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > MAX_IMAGE_SIZE_MB:
        return False, f"File too large. Maximum size: {MAX_IMAGE_SIZE_MB}MB"
    
    # Try to open and validate image
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
                return False, f"Image too small. Minimum size: {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}px"
            if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                return False, f"Image too large. Maximum size: {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}px"
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"
    
    return True, None


def validate_prompt(prompt: str) -> Tuple[bool, Optional[str]]:
    """Validate prompt input."""
    if not prompt or not prompt.strip():
        return True, None  # Empty prompt is allowed (will use default)
    
    if len(prompt) > MAX_PROMPT_LENGTH:
        return False, f"Prompt too long. Maximum length: {MAX_PROMPT_LENGTH} characters"
    
    return True, None


def validate_negative_prompt(negative_prompt: str) -> Tuple[bool, Optional[str]]:
    """Validate negative prompt input."""
    if not negative_prompt:
        return True, None
    
    if len(negative_prompt) > MAX_NEGATIVE_PROMPT_LENGTH:
        return False, f"Negative prompt too long. Maximum length: {MAX_NEGATIVE_PROMPT_LENGTH} characters"
    
    return True, None


# ============ MODEL LOADING ============
print("Loading AI models... This may take a few minutes on first run.")

# InstantID checkpoints
hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="ControlNetModel/config.json",
    local_dir="./checkpoints",
)
hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="ControlNetModel/diffusion_pytorch_model.safetensors",
    local_dir="./checkpoints",
)
hf_hub_download(
    repo_id="InstantX/InstantID", filename="ip-adapter.bin", local_dir="./checkpoints"
)

# Face encoder
app = FaceAnalysis(
    name="antelopev2",
    root="./",
    providers=["CPUExecutionProvider"],
)
app.prepare(ctx_id=0, det_size=(640, 640))

# DepthAnything
depth_anything = DepthAnything.from_pretrained(
    "LiheYoung/depth_anything_vitl14"
).to(device).eval()

transform = Compose(
    [
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ]
)

face_adapter = "./checkpoints/ip-adapter.bin"
controlnet_path = "./checkpoints/ControlNetModel"

controlnet_identitynet = ControlNetModel.from_pretrained(
    controlnet_path, torch_dtype=dtype
)

controlnet_canny_model = "diffusers/controlnet-canny-sdxl-1.0"
controlnet_depth_model = "diffusers/controlnet-depth-sdxl-1.0-small"

controlnet_canny = ControlNetModel.from_pretrained(
    controlnet_canny_model, torch_dtype=dtype
).to(device)
controlnet_depth = ControlNetModel.from_pretrained(
    controlnet_depth_model, torch_dtype=dtype
).to(device)


def get_depth_map(image):
    """Generate depth map from image."""
    image = np.array(image) / 255.0
    h, w = image.shape[:2]
    image = transform({"image": image})["image"]
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    with torch.no_grad():
        depth = depth_anything(image)
    depth = F.interpolate(
        depth[None], (h, w), mode="bilinear", align_corners=False
    )[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.cpu().numpy().astype(np.uint8)
    depth_image = Image.fromarray(depth)
    return depth_image


def get_canny_image(image, t1=100, t2=200):
    """Generate canny edge map from image."""
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    edges = cv2.Canny(image, t1, t2)
    return Image.fromarray(edges, "L")


controlnet_map = {
    "canny": controlnet_canny,
    "depth": controlnet_depth,
}
controlnet_map_fn = {
    "canny": get_canny_image,
    "depth": get_depth_map,
}

pretrained_model_name_or_path = "wangqixun/YamerMIX_v8"

pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    pretrained_model_name_or_path,
    controlnet=[controlnet_identitynet],
    torch_dtype=dtype,
    safety_checker=None,
    feature_extractor=None,
).to(device)

# Standard scheduler
pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(
    pipe.scheduler.config
)

if device == "cuda":
    pipe.cuda()
    pipe.load_ip_adapter_instantid(face_adapter)
    pipe.image_proj_model.to("cuda")
    pipe.unet.to("cuda")
else:
    pipe.load_ip_adapter_instantid(face_adapter)

print("✅ Models loaded successfully!")


# ============ UTILS ============
def toggle_lcm_ui(value: bool) -> Tuple[dict, dict]:
    """Toggle UI for LCM mode."""
    if value:
        return (
            gr.update(minimum=0, maximum=100, step=1, value=5),
            gr.update(minimum=0.1, maximum=20.0, step=0.1, value=1.5),
        )
    else:
        return (
            gr.update(minimum=5, maximum=100, step=1, value=DEFAULT_NUM_STEPS),
            gr.update(minimum=0.1, maximum=20.0, step=0.1, value=DEFAULT_GUIDANCE_SCALE),
        )


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    """Randomize seed if requested."""
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def convert_from_cv2_to_image(img: np.ndarray) -> Image.Image:
    """Convert OpenCV image to PIL Image."""
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_image_to_cv2(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def resize_img(
    input_image,
    max_side=1280,
    min_side=1024,
    size=None,
    pad_to_max_side=False,
    mode=PIL.Image.BILINEAR,
    base_pixel_number=64,
):
    """Resize image maintaining aspect ratio."""
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y : offset_y + h_resize_new, offset_x : offset_x + w_resize_new] = (
            np.array(input_image)
        )
        input_image = Image.fromarray(res)
    return input_image


def apply_style(style_name: str, positive: str, negative: str = "") -> Tuple[str, str]:
    """Apply style template to prompts."""
    if style_name == "No Style":
        return positive, negative
    p, n = styles.get(style_name, ("{prompt}", ""))
    return p.replace("{prompt}", positive), n + " " + negative


# ============ GENERATION FUNCTION ============
@spaces.GPU  # ZeroGPU will allocate GPU for this function
def generate_image(
    face_image_path: str,
    prompt: str,
    negative_prompt: str,
    style_name: str,
    num_steps: int,
    identitynet_strength_ratio: float,
    adapter_strength_ratio: float,
    canny_strength: float,
    depth_strength: float,
    controlnet_selection: list,
    guidance_scale: float,
    seed: int,
    scheduler: str,
    enable_LCM: bool,
    enhance_face_region: bool,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate professional headshot from face image."""
    try:
        # Validate inputs
        is_valid, error_msg = validate_image_file(face_image_path)
        if not is_valid:
            raise gr.Error(error_msg)
        
        is_valid, error_msg = validate_prompt(prompt)
        if not is_valid:
            raise gr.Error(error_msg)
        
        is_valid, error_msg = validate_negative_prompt(negative_prompt)
        if not is_valid:
            raise gr.Error(error_msg)

        # Randomize seed if needed
        if seed < 0:
            seed = random.randint(0, MAX_SEED)

        # Configure scheduler
        scheduler_class_name = scheduler.split("-")[0]
        add_kwargs = {}
        if len(scheduler.split("-")) > 1:
            add_kwargs["use_karras_sigmas"] = True
        if len(scheduler.split("-")) > 2:
            add_kwargs["algorithm_type"] = "sde-dpmsolver++"
        scheduler_cls = getattr(diffusers, scheduler_class_name)
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config, **add_kwargs)

        # Apply style
        if not prompt:
            prompt = "a person"
        
        prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)

        # Load and process face image
        face_image = load_image(face_image_path)
        face_image = resize_img(face_image, max_side=1024)
        face_image_cv2 = convert_from_image_to_cv2(face_image)
        height, width, _ = face_image_cv2.shape

        # Detect face
        face_info = app.get(face_image_cv2)
        if len(face_info) == 0:
            raise gr.Error(
                "Unable to detect a face in the image. Please upload a different photo with a clear face."
            )

        # Use largest detected face
        face_info = sorted(
            face_info,
            key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
        )[-1]
        face_emb = face_info["embedding"]
        face_kps = draw_kps(convert_from_cv2_to_image(face_image_cv2), face_info["kps"])
        img_controlnet = face_image

        # Create control mask if requested
        if enhance_face_region:
            control_mask = np.zeros([height, width, 3])
            x1, y1, x2, y2 = face_info["bbox"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            control_mask[y1:y2, x1:x2] = 255
            control_mask = Image.fromarray(control_mask.astype(np.uint8))
        else:
            control_mask = None

        # Configure ControlNet
        if len(controlnet_selection) > 0:
            controlnet_scales = {
                "canny": canny_strength,
                "depth": depth_strength,
            }
            pipe.controlnet = MultiControlNetModel(
                [controlnet_identitynet]
                + [controlnet_map[s] for s in controlnet_selection]
            )
            control_scales = [float(identitynet_strength_ratio)] + [
                controlnet_scales[s] for s in controlnet_selection
            ]
            control_images = [face_kps] + [
                controlnet_map_fn[s](img_controlnet).resize((width, height))
                for s in controlnet_selection
            ]
        else:
            pipe.controlnet = controlnet_identitynet
            control_scales = float(identitynet_strength_ratio)
            control_images = face_kps

        # Adjust steps for LCM if enabled
        if enable_LCM:
            num_steps = max(5, min(num_steps, 10))
            guidance_scale = max(1.0, min(guidance_scale, 2.0))

        generator = torch.Generator(device=device).manual_seed(seed)

        pipe.set_ip_adapter_scale(adapter_strength_ratio)
        
        # Generate image
        images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_embeds=face_emb,
            image=control_images,
            control_mask=control_mask,
            controlnet_conditioning_scale=control_scales,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        ).images

        final_image = images[0]
        save_as_png(final_image)
        
        return final_image

    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"An error occurred during generation: {str(e)}")


# ============ CSS STYLING ============
css = """
/* Main container styling */
.main-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}

/* Hero section */
.hero-title {
    font-size: 2.5em;
    font-weight: 700;
    margin-bottom: 10px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    font-size: 1.1em;
    color: #666;
    margin-bottom: 30px;
}

/* Control cards */
.control-card {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.control-header {
    display: flex;
    align-items: center;
    margin-bottom: 15px;
}

.control-icon {
    font-size: 1.5em;
    margin-right: 10px;
}

.control-title {
    font-size: 1.2em;
    font-weight: 600;
    margin: 0;
}

/* Upload area */
.upload-area {
    border: 2px dashed #667eea;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}

/* Tips card */
.tips-card {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 20px;
}

.tips-header {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
}

.tips-icon {
    font-size: 1.3em;
    margin-right: 8px;
}

/* Result card */
.result-card {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.result-header {
    margin-bottom: 20px;
}

.result-title {
    font-size: 1.5em;
    font-weight: 600;
    margin-bottom: 5px;
}

.result-subtitle {
    color: #666;
    font-size: 0.95em;
}

/* Image container */
.image-container {
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

/* Success banner */
.success-banner {
    background: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 8px;
    padding: 15px;
    margin-top: 15px;
}

/* Primary button */
.btn-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    font-weight: 600;
    padding: 12px 30px;
}

.btn-primary:hover {
    opacity: 0.9;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
"""


# ============ UI / GRADIO ============
def show_success():
    """Show success message after generation."""
    return gr.update(
        value="""
        <div class="success-banner">
            <h4 style="margin: 0 0 8px 0;">✅ Success! Your Professional Headshot is Ready</h4>
            <p style="margin: 0; opacity: 0.9;">Download your high-quality PNG file for LinkedIn, professional profiles, or portfolios.</p>
        </div>
        """,
    )


with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_classes="main-container"):
        with gr.Column(elem_classes="hero-section"):
            gr.HTML(
                """
                <div style="position: relative; z-index: 2;">
                    <h1 class="hero-title">🎯 Pro AI Headshot Generator</h1>
                    <p class="hero-subtitle">Transform any selfie into professional headshots in seconds. Perfect for LinkedIn, corporate profiles, and professional portfolios.</p>
                </div>
                """
            )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=400):
                with gr.Column(elem_classes="control-card"):
                    gr.HTML(
                        """
                        <div class="control-header">
                            <div class="control-icon">📸</div>
                            <h3 class="control-title">Upload Your Photo</h3>
                        </div>
                        """
                    )
                    gr.HTML(
                        """
                        <p style="color: var(--text-secondary); margin-bottom: 20px; font-size: 0.95em;">
                            For best results, use a clear, well-lit photo where your face is clearly visible.
                        </p>
                        """
                    )
                    face_file = gr.Image(
                        label="",
                        type="filepath",
                        height=200,
                        show_label=False,
                        elem_classes="upload-area",
                    )

                with gr.Column(elem_classes="control-card"):
                    gr.HTML(
                        """
                        <div class="control-header">
                            <div class="control-icon">✍️</div>
                            <h3 class="control-title">Describe Your Look</h3>
                        </div>
                        """
                    )
                    prompt = gr.Textbox(
                        label="",
                        placeholder="Describe how you want to appear...",
                        value="modern professional headshot, creative director style, soft natural lighting, authentic expression, contemporary business portrait, premium quality photo",
                        show_label=False,
                        lines=3,
                    )
                    gr.HTML(
                        """
                        <div style="font-size: 0.85em; color: var(--text-secondary); margin-top: 8px;">
                            💡 Examples: "professional business headshot", "friendly corporate portrait", "creative director style"
                        </div>
                        """
                    )

                with gr.Column(elem_classes="control-card"):
                    gr.HTML(
                        """
                        <div class="control-header">
                            <div class="control-icon">🎨</div>
                            <h3 class="control-title">Style Options</h3>
                        </div>
                        """
                    )
                    style = gr.Dropdown(
                        label="Style Theme",
                        choices=["No Style"] + STYLE_NAMES,
                        value="No Style",
                        info="'No Style' recommended for natural professional results",
                    )

                with gr.Column(elem_classes="control-card"):
                    gr.HTML(
                        """
                        <div class="control-header">
                            <div class="control-icon">⚙️</div>
                            <h3 class="control-title">Quality Settings</h3>
                        </div>
                        """
                    )

                    identitynet_strength_ratio = gr.Slider(
                        label="Face Similarity",
                        minimum=0.5,
                        maximum=1.2,
                        step=0.05,
                        value=0.80,
                        info="How closely the headshot resembles your photo",
                    )

                    adapter_strength_ratio = gr.Slider(
                        label="Detail Quality",
                        minimum=0.3,
                        maximum=1.2,
                        step=0.05,
                        value=0.55,
                        info="Level of detail in the final image",
                    )

                    enable_LCM = gr.Checkbox(
                        label="Enable Fast Generation Mode",
                        value=False,
                        info="Faster results with slightly lower quality",
                    )

                with gr.Column(elem_classes="tips-card"):
                    gr.HTML(
                        """
                        <div class="tips-header">
                            <div class="tips-icon">💡</div>
                            <h4 style="margin: 0; color: #92400e;">Pro Tips for Best Results</h4>
                        </div>
                        <ul style="margin: 0; color: #92400e; font-size: 0.9em;">
                            <li>Use clear, well-lit face photos</li>
                            <li>Face should be visible and not too small</li>
                            <li>Avoid blurry or dark images</li>
                            <li>Single person in photo works best</li>
                        </ul>
                        """
                    )

                submit = gr.Button(
                    "✨ Generate Professional Headshot",
                    variant="primary",
                    size="lg",
                    elem_classes="btn-primary",
                    scale=1,
                )

            with gr.Column(scale=1, min_width=500):
                with gr.Column(elem_classes="result-card"):
                    gr.HTML(
                        """
                        <div class="result-header">
                            <h2 class="result-title">Your Professional Headshot</h2>
                            <p class="result-subtitle">Your AI-generated headshot will appear here. Download as high-quality PNG for professional use.</p>
                        </div>
                        """
                    )

                    gallery = gr.Image(
                        label="Output",
                        height=400,
                        show_label=False,
                        type="pil",
                        elem_classes="image-container",
                    )

                    success_msg = gr.HTML(
                        """
                        <div class="success-banner" style="display: none;">
                            <h4 style="margin: 0 0 8px 0;">✅ Success! Your Professional Headshot is Ready</h4>
                            <p style="margin: 0; opacity: 0.9;">Download your high-quality PNG file for LinkedIn, professional profiles, or portfolios.</p>
                        </div>
                        """
                    )

                    progress_info = gr.HTML(
                        """
                        <div class="progress-container">
                            <div style="font-size: 0.9em; color: var(--text-secondary);">
                                ⏱️ Generation takes 20-30 seconds
                            </div>
                        </div>
                        """
                    )

    # Hidden advanced settings
    negative_prompt = gr.Textbox(
        value="(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        visible=False,
    )
    num_steps = gr.Slider(
        minimum=5,
        maximum=100,
        step=1,
        value=DEFAULT_NUM_STEPS,
        label="Number of steps",
        visible=False,
    )
    guidance_scale = gr.Slider(
        minimum=0.1,
        maximum=20.0,
        step=0.1,
        value=DEFAULT_GUIDANCE_SCALE,
        label="Guidance scale",
        visible=False,
    )
    seed = gr.Slider(
        minimum=-1,
        maximum=MAX_SEED,
        step=1,
        value=-1,
        label="Seed (-1 for random)",
        visible=False,
    )
    scheduler = gr.Dropdown(
        value="EulerDiscreteScheduler",
        choices=[
            "EulerDiscreteScheduler",
            "EulerAncestralDiscreteScheduler",
            "DPMSolverMultistepScheduler",
        ],
        visible=False,
    )
    randomize_seed = gr.Checkbox(value=True, visible=False)
    enhance_face_region = gr.Checkbox(value=True, visible=False)
    controlnet_selection = gr.CheckboxGroup(
        choices=["canny", "depth"], value=["depth"], label="Controlnet", visible=False
    )
    canny_strength = gr.Slider(
        minimum=0,
        maximum=1.5,
        step=0.01,
        value=0.4,
        label="Canny strength",
        visible=False,
    )
    depth_strength = gr.Slider(
        minimum=0,
        maximum=1.5,
        step=0.01,
        value=0.4,
        label="Depth strength",
        visible=False,
    )

    submit.click(
        fn=generate_image,
        inputs=[
            face_file,
            prompt,
            negative_prompt,
            style,
            num_steps,
            identitynet_strength_ratio,
            adapter_strength_ratio,
            canny_strength,
            depth_strength,
            controlnet_selection,
            guidance_scale,
            seed,
            scheduler,
            enable_LCM,
            enhance_face_region,
        ],
        outputs=[gallery],
    ).then(fn=show_success, outputs=success_msg)

    enable_LCM.input(
        fn=toggle_lcm_ui,
        inputs=[enable_LCM],
        outputs=[num_steps, guidance_scale],
        queue=False,
    )

    # Cleanup on startup
    cleanup_old_files()


if __name__ == "__main__":
    demo.queue(api_open=False)
    demo.launch(
        share=True,
    )
