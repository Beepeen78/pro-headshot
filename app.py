import cv2
import torch
import random
import numpy as np
import os
import time
import secrets
import json
from datetime import datetime, timedelta
from typing import Tuple

import spaces

import PIL
from PIL import Image, ImageDraw, ImageFont

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


# ============ GLOBAL CONFIG ============
MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32
STYLE_NAMES = list(styles.keys())

LICENSE_FILE = "valid_licenses.json"
TRIAL_FILE = "user_trials.json"
MAX_FREE_TRIALS = 3

ADMIN_KEYS = {
    "HEADSHOT-TEST123456",
    "HEADSHOT-OWNERACCESS",
    "HEADSHOT-DEVELOPER123",
    "HEADSHOT-ADMIN789012",
}


# ============ LICENSE / TRIAL SYSTEM ============
def load_licenses():
    try:
        with open(LICENSE_FILE, "r") as f:
            return set(json.load(f))
    except FileNotFoundError:
        initial_licenses = ADMIN_KEYS.copy()
        save_licenses(initial_licenses)
        return initial_licenses


def save_licenses(licenses):
    with open(LICENSE_FILE, "w") as f:
        json.dump(list(licenses), f)


def generate_license_key():
    license_key = f"HEADSHOT-{secrets.token_hex(6).upper()}"
    valid_licenses = load_licenses()
    valid_licenses.add(license_key)
    save_licenses(valid_licenses)
    return license_key


def verify_license(license_key):
    if not license_key or not license_key.strip():
        return False

    license_upper = license_key.strip().upper()

    if license_upper in ADMIN_KEYS:
        print(f"✅ Admin access granted with: {license_upper}")
        return True

    valid_licenses = load_licenses()
    return license_upper in valid_licenses


def load_trials():
    try:
        with open(TRIAL_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_trials(trials_data):
    with open(TRIAL_FILE, "w") as f:
        json.dump(trials_data, f)


def get_user_identifier():
    # Simple identifier for Spaces
    return "gradio_user"


def can_use_free_trial(user_id):
    trials_data = load_trials()

    if user_id not in trials_data:
        return True, MAX_FREE_TRIALS

    user_data = trials_data[user_id]
    trials_used = user_data.get("trials_used", 0)
    first_trial_date = user_data.get("first_trial_date")

    if first_trial_date:
        first_date = datetime.fromisoformat(first_trial_date)
        if datetime.now() - first_date > timedelta(days=30):
            trials_used = 0
            user_data["trials_used"] = 0
            user_data["first_trial_date"] = datetime.now().isoformat()
            save_trials(trials_data)

    trials_left = MAX_FREE_TRIALS - trials_used
    return trials_left > 0, trials_left


def record_trial_usage(user_id):
    trials_data = load_trials()

    if user_id not in trials_data:
        trials_data[user_id] = {
            "trials_used": 1,
            "first_trial_date": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat(),
        }
    else:
        trials_data[user_id]["trials_used"] += 1
        trials_data[user_id]["last_used"] = datetime.now().isoformat()

    save_trials(trials_data)


def apply_watermark(image):
    if hasattr(image, "mode"):
        pil_image = image
    else:
        pil_image = Image.fromarray(image)

    draw = ImageDraw.Draw(pil_image, "RGBA")
    width, height = pil_image.size

    watermark_text = "PREVIEW - UPGRADE TO DOWNLOAD"

    try:
        font = ImageFont.truetype("arial.ttf", min(width, height) // 20)
    except Exception:
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                min(width, height) // 20,
            )
        except Exception:
            font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), watermark_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (width - text_width) // 2
    y = height - text_height - 50

    draw.rectangle(
        [x - 10, y - 10, x + text_width + 10, y + text_height + 10],
        fill=(0, 0, 0, 128),
    )
    draw.text((x, y), watermark_text, fill=(255, 255, 255, 255), font=font)

    return pil_image


VALID_LICENSES = load_licenses()
print(f"✅ License system initialized. Admin keys: {ADMIN_KEYS}")


# ============ MODEL LOADING ============

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

from insightface.app import FaceAnalysis

# Face encoder
app = FaceAnalysis(
    name="buffalo_l",
    root="./models",
    allowed_modules=["detection", "recognition"],
    providers=["CPUExecutionProvider"],
)
app.prepare(ctx_id=0, det_size=(640, 640))

face_adapter = "./checkpoints/ip-adapter.bin"
controlnet_path = "./checkpoints/ControlNetModel"

controlnet_identitynet = ControlNetModel.from_pretrained(
    controlnet_path, torch_dtype=dtype
)

controlnet_canny_model = "diffusers/controlnet-canny-sdxl-1.0"

controlnet_canny = ControlNetModel.from_pretrained(
    controlnet_canny_model, torch_dtype=dtype
).to(device)


def get_canny_image(image, t1=100, t2=200):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    edges = cv2.Canny(image, t1, t2)
    return Image.fromarray(edges, "L")


controlnet_map = {
    "canny": controlnet_canny,
}
controlnet_map_fn = {
    "canny": get_canny_image,
}

pretrained_model_name_or_path = "wangqixun/YamerMIX_v8"

pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    pretrained_model_name_or_path,
    controlnet=[controlnet_identitynet],
    torch_dtype=dtype,
    safety_checker=None,
    feature_extractor=None,
).to(device)

# Standard scheduler (no LCM / LoRA here)
pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(
    pipe.scheduler.config
)

# Move pipeline and submodules to selected device
pipe.load_ip_adapter_instantid(face_adapter)
pipe.to(device, dtype=dtype)
pipe.image_proj_model.to(device)
pipe.unet.to(device)


# ============ UTILS ============
def toggle_lcm_ui(value):
    if value:
        return (
            gr.update(minimum=0, maximum=100, step=1, value=5),
            gr.update(minimum=0.1, maximum=20.0, step=0.1, value=1.5),
        )
    else:
        return (
            gr.update(minimum=5, maximum=100, step=1, value=30),
            gr.update(minimum=0.1, maximum=20.0, step=0.1, value=5),
        )


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
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
    if style_name == "No Style":
        return positive, negative
    p, n = styles.get(style_name, ("{prompt}", ""))
    return p.replace("{prompt}", positive), n + " " + negative


def save_as_png(image, filename="professional_headshot"):
    temp_dir = "temp_downloads"
    os.makedirs(temp_dir, exist_ok=True)

    timestamp = int(time.time())
    filepath = os.path.join(temp_dir, f"{filename}_{timestamp}.png")

    if image.mode in ("RGBA", "LA"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    image.save(filepath, "PNG", optimize=True)
    return filepath


# ============ GENERATION FUNCTION ============
@spaces.GPU
def generate_image(
    face_image_path,
    license_key,
    prompt,
    negative_prompt,
    style_name,
    num_steps,
    identitynet_strength_ratio,
    adapter_strength_ratio,
    canny_strength,
    depth_strength,          # kept in signature but unused
    controlnet_selection,
    guidance_scale,
    seed,
    scheduler,
    enable_LCM,
    enhance_face_region,
    progress=gr.Progress(track_tqdm=True),
):
    user_id = get_user_identifier()
    has_valid_license = verify_license(license_key)

    if not has_valid_license:
        can_use_trial, trials_left = can_use_free_trial(user_id)

        if not can_use_trial:
            raise gr.Error(
                f"""
❌ Your free trial has ended.
You’ve used all {MAX_FREE_TRIALS} free generations for this account.

🔑 To continue generating headshots:
1. Purchase a premium license for unlimited HD downloads (no watermark), or
2. Enter an existing license key if you already purchased.

Visit: https://canadianheadshotpro.carrd.co
Support: bee.tools@zohomailcloud.ca
"""
            )

    scheduler_class_name = scheduler.split("-")[0]
    add_kwargs = {}
    if len(scheduler.split("-")) > 1:
        add_kwargs["use_karras_sigmas"] = True
    if len(scheduler.split("-")) > 2:
        add_kwargs["algorithm_type"] = "sde-dpmsolver++"
    scheduler_cls = getattr(diffusers, scheduler_class_name)
    pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config, **add_kwargs)

    if face_image_path is None:
        raise gr.Error("Please upload a face image.")

    if not prompt:
        prompt = "a person"

    prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)

    face_image = load_image(face_image_path)
    face_image = resize_img(face_image, max_side=1024)
    face_image_cv2 = convert_from_image_to_cv2(face_image)
    height, width, _ = face_image_cv2.shape

    face_info = app.get(face_image_cv2)
    if len(face_info) == 0:
        raise gr.Error(
            "Unable to detect a face in the image. Please upload a different photo with a clear face."
        )

    face_info = sorted(
        face_info,
        key=lambda x: (x["bbox"][2] - x["bbox"][0]) * x["bbox"][3] - x["bbox"][1],
    )[-1]
    face_emb = face_info["embedding"]
    face_kps = draw_kps(convert_from_cv2_to_image(face_image_cv2), face_info["kps"])
    img_controlnet = face_image

    if enhance_face_region:
        control_mask = np.zeros([height, width, 3])
        x1, y1, x2, y2 = face_info["bbox"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        control_mask[y1:y2, x1:x2] = 255
        control_mask = Image.fromarray(control_mask.astype(np.uint8))
    else:
        control_mask = None

    # Force only canny controlnet
    if len(controlnet_selection) > 0:
        controlnet_scales = {
            "canny": canny_strength,
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

    generator = torch.Generator(device=device).manual_seed(seed)

    pipe.set_ip_adapter_scale(adapter_strength_ratio)
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

    if not has_valid_license:
        record_trial_usage(user_id)
        final_image = apply_watermark(final_image)
        _, trials_left = can_use_free_trial(user_id)
        gr.Info(
            f"Free trial used! {trials_left} generations remaining. Upgrade for watermark-free HD downloads."
        )

    save_as_png(final_image)

    return final_image


# ============ UI / GRADIO ============
css = """/* your CSS from earlier unchanged */"""


def show_success():
    return gr.update(
        value="""
        <div class="success-banner">
            <h4 style="margin: 0 0 8px 0;">✅ Success! Your Professional Headshot is Ready</h4>
            <p style="margin: 0; opacity: 0.9;">Download your high-quality PNG file for LinkedIn, professional profiles, or portfolios.</p>
        </div>
        """,
    )


def update_trial_display(license_key):
    if verify_license(license_key):
        return """
        <div class="trial-banner">
            <h4 style="margin: 0 0 8px 0;">✅ Premium License Active</h4>
            <p style="margin: 0; font-size: 0.9em;">Unlimited HD downloads - no watermark</p>
        </div>
        """

    user_id = get_user_identifier()
    can_use, trials_left = can_use_free_trial(user_id)

    if not can_use:
        return """
        <div class="trial-banner-error">
            <h4 style="margin: 0 0 8px 0;">❌ No Free Trials Left</h4>
            <p style="margin: 0; font-size: 0.9em;">Please purchase a license to continue</p>
        </div>
        """

    return f"""
    <div class="trial-banner">
        <h4 style="margin: 0 0 8px 0;">🎉 {trials_left} Free Generations Left!</h4>
        <p style="margin: 0; font-size: 0.9em;">Watermarked previews - upgrade for HD downloads</p>
    </div>
    """


with gr.Blocks() as demo:
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
                            <div class="control-icon">🔑</div>
                            <h3 class="control-title">Access Options</h3>
                        </div>
                        """
                    )

                    trial_status = gr.HTML(
                        f"""
                        <div class="trial-banner">
                            <h4 style="margin: 0 0 8px 0;">🎉 {MAX_FREE_TRIALS} FREE Trials Available!</h4>
                            <p style="margin: 0; font-size: 0.9em;">Try our AI headshot generator - no credit card required</p>
                        </div>
                        """
                    )

                    license_input = gr.Textbox(
                        label="",
                        placeholder="Enter license key (or leave blank for free trial)",
                        show_label=False,
                        info=f"💡 You get {MAX_FREE_TRIALS} free generations. Purchase license for HD downloads without watermark.",
                    )

                    gr.HTML(
                        f"""
                        <div style="font-size: 0.85em; color: var(--text-secondary); margin-top: 8px;">
                            <strong>Free Trial:</strong> {MAX_FREE_TRIALS} watermarked previews<br>
                            <strong>Premium License:</strong> Unlimited HD downloads, no watermark<br>
                            <strong>Professional Use:</strong> Commercial rights included<br>
                            <a href="https://canadianheadshotpro.carrd.co" target="_blank" style="color: var(--primary); font-weight: 600;">👉 Click here to purchase license</a>
                        </div>
                        """
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

    negative_prompt = gr.Textbox(
        value="(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        visible=False,
    )
    num_steps = gr.Slider(
        minimum=5,
        maximum=100,
        step=1,
        value=30,
        label="Number of steps",
        visible=False,
    )
    guidance_scale = gr.Slider(
        minimum=0.1,
        maximum=20.0,
        step=0.1,
        value=5.0,
        label="Guidance scale",
        visible=False,
    )
    seed = gr.Slider(
        minimum=0,
        maximum=MAX_SEED,
        step=1,
        value=42,
        label="Seed",
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
        choices=["canny"], value=["canny"], label="Controlnet", visible=False
    )
    canny_strength = gr.Slider(
        minimum=0,
        maximum=1.5,
        step=0.01,
        value=0.4,
        label="Canny strength",
        visible=False,
    )
    depth_strength = gr.Slider(  # kept but unused
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
            license_input,
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

    license_input.change(
        fn=update_trial_display, inputs=[license_input], outputs=[trial_status]
    )

    enable_LCM.input(
        fn=toggle_lcm_ui,
        inputs=[enable_LCM],
        outputs=[num_steps, guidance_scale],
        queue=False,
    )


if __name__ == "__main__":
    demo.queue(api_open=False)
    demo.launch(
        share=True,
        css=css,
        theme=gr.themes.Soft()
    )
