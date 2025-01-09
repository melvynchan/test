import os
import torch
import io
import base64
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from model.pipeline import CatVTONPipeline
from model.cloth_masker import AutoMasker
from utils import init_weight_dtype, resize_and_crop, resize_and_padding
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="CatVTON API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for models
pipeline = None
automasker = None

def load_models():
    global pipeline, automasker
    if pipeline is None:
        print("Loading models...")
        try:
            # Download models only if not already downloaded
            if not os.path.exists("models_cache"):
                os.makedirs("models_cache")
                repo_path = snapshot_download(
                    repo_id="zhengchong/CatVTON",
                    local_dir="models_cache",
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    force_download=False
                )
            else:
                repo_path = "models_cache"

            # Initialize pipeline with safety settings
            pipeline = CatVTONPipeline(
                base_ckpt="runwayml/stable-diffusion-inpainting",
                attn_ckpt=repo_path,
                attn_ckpt_version="vitonhd",
                weight_dtype=init_weight_dtype("fp16"),
                use_tf32=False,
                device='cuda',
                skip_safety_check=True
            )
            
            # Initialize automasker
            automasker = AutoMasker(
                densepose_ckpt=os.path.join(repo_path, "DensePose"),
                schp_ckpt=os.path.join(repo_path, "SCHP"),
                device='cuda'
            )
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise

async def process_images(person_image: Image.Image, cloth_image: Image.Image):
    # Resize images
    person_image = resize_and_padding(person_image, (512, 768))
    cloth_image = resize_and_crop(cloth_image, (512, 768))
    
    # Process with automasker
    person_parse = automasker(person_image)['mask']
    cloth_parse = automasker(cloth_image)['mask']
    
    # Generate try-on result
    result_image = pipeline(
        image=person_image,
        condition_image=cloth_image,
        mask=person_parse,
        num_inference_steps=15,
        guidance_scale=2,
        height=768,
        width=512
    )[0]
    
    return result_image

@app.on_event("startup")
async def startup_event():
    load_models()

@app.post("/try-on")
async def try_on(person_image: UploadFile = File(...), cloth_image: UploadFile = File(...)):
    try:
        # Read and convert uploaded files to PIL Images
        person_img = Image.open(io.BytesIO(await person_image.read()))
        cloth_img = Image.open(io.BytesIO(await cloth_image.read()))
        
        # Process images
        result = await process_images(person_img, cloth_img)
        
        # Convert result to base64
        buffered = io.BytesIO()
        result.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return JSONResponse(content={"status": "success", "image": img_str})
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/try-on-base64")
async def try_on_base64(
    person_image_base64: str = Body(..., embed=True),
    cloth_image_base64: str = Body(..., embed=True)
):
    try:
        # Convert base64 to PIL Images
        person_img = Image.open(io.BytesIO(base64.b64decode(person_image_base64)))
        cloth_img = Image.open(io.BytesIO(base64.b64decode(cloth_image_base64)))
        
        # Process images
        result = await process_images(person_img, cloth_img)
        
        # Convert result to base64
        buffered = io.BytesIO()
        result.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return JSONResponse(content={"status": "success", "image": img_str})
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)