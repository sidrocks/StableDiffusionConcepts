import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
import os

# Monkeypatch fixes for environment compatibility
def apply_patches():
    """Apply necessary patches for tqdm and symlinks"""
    import sys
    import shutil
    
    # 1. Fix tqdm Jupyter/Thread Error
    try:
        from tqdm.auto import tqdm
        if not hasattr(tqdm, '_is_patched'):
            import tqdm.notebook
            import tqdm.std
            tqdm.notebook.tqdm = tqdm.std.tqdm
            tqdm.notebook.trange = tqdm.std.trange
            if 'tqdm.auto' in sys.modules:
                sys.modules['tqdm.auto'].tqdm = tqdm.std.tqdm
                sys.modules['tqdm.auto'].trange = tqdm.std.trange
            tqdm._is_patched = True
    except ImportError:
        pass
    
    # 2. Fix Windows Symlink Permissions
    try:
        from huggingface_hub import file_download
        if not hasattr(file_download, '_original_create_symlink'):
            file_download._original_create_symlink = file_download._create_symlink
            
            def patched_create_symlink(src, dst, new_blob=False):
                try:
                    file_download._original_create_symlink(src, dst, new_blob)
                except OSError as e:
                    if getattr(e, 'winerror', 0) == 1314:
                        if os.path.isdir(src):
                            shutil.copytree(src, dst)
                        else:
                            shutil.copy2(src, dst)
                    else:
                        raise
            
            file_download._create_symlink = patched_create_symlink
    except ImportError:
        pass

# Apply patches before loading models
apply_patches()

# Style configurations with default seeds
STYLES = {
    "Cat Toy": {
        "repo": "sd-concepts-library/cat-toy",
        "token": "<cat-toy>",
        "description": "Cute cat toy aesthetic",
        "default_seed": 42
    },
    "Seletti": {
        "repo": "sd-concepts-library/seletti",
        "token": "<seletti>",
        "description": "Seletti design style",
        "default_seed": 142
    },
    "Madhubani Art": {
        "repo": "sd-concepts-library/madhubani-art",
        "token": "<madhubani-art>",
        "description": "Traditional Indian Madhubani art style",
        "default_seed": 242
    },
    "Chucky": {
        "repo": "sd-concepts-library/chucky",
        "token": "<chucky>",
        "description": "Chucky horror character style",
        "default_seed": 342
    },
    "Indian Watercolor Portraits": {
        "repo": "sd-concepts-library/indian-watercolor-portraits",
        "token": "<indian-watercolor-portraits>",
        "description": "Indian watercolor portrait art style",
        "default_seed": 442
    },
    "Anime Boy": {
        "repo": "sd-concepts-library/anime-boy",
        "token": "<anime-boy>",
        "description": "Anime boy character style",
        "default_seed": 542
    }
}

# Global pipeline variable
pipe = None
current_style = None

def contrast_loss(images):
    """Calculate High-Contrast loss (maximizes variance/extremes)"""
    return -torch.mean((images - 0.5) ** 2)

def complexity_loss(images):
    """Calculate Complexity loss (maximizes local detail/edges)"""
    diff_h = torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :])
    diff_v = torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1])
    return torch.mean(diff_h) + torch.mean(diff_v)

def vibrancy_loss(images):
    """Calculate Vibrancy loss (maximizes color saturation/variety)"""
    # Maximize standard deviation across color channels
    # Or boost the distance from grayscale
    means = torch.mean(images, dim=1, keepdim=True)
    return -torch.mean((images - means) ** 2)

def custom_sampling_loop(prompt, pipe, guidance_scale=7.5, contrast_scale=0.0, complexity_scale=0.0, vibrancy_scale=0.0, num_inference_steps=50, generator=None, num_images=1):
    device = pipe.device
    dtype = pipe.unet.dtype
    text_input = pipe.tokenizer([prompt] * num_images, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]
    uncond_input = pipe.tokenizer([""] * num_images, padding="max_length", max_length=text_input.input_ids.shape[-1], return_tensors="pt")
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    latents = torch.randn((num_images, pipe.unet.config.in_channels, 512 // 8, 512 // 8), generator=generator, device=device, dtype=dtype)
    pipe.scheduler.set_timesteps(num_inference_steps)
    latents = latents * pipe.scheduler.init_noise_sigma
    from tqdm.auto import tqdm
    for t in tqdm(pipe.scheduler.timesteps):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        with torch.no_grad():
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # COMBINED GUIDANCE GRADIENT STEP
        if contrast_scale > 0 or complexity_scale > 0 or vibrancy_scale > 0:
            latents = latents.detach().requires_grad_(True)
            image = pipe.vae.decode(1 / 0.18215 * latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            
            loss = 0
            if contrast_scale > 0:
                loss = loss + contrast_loss(image) * contrast_scale
            if complexity_scale > 0:
                loss = loss - complexity_loss(image) * complexity_scale
            if vibrancy_scale > 0:
                loss = loss + vibrancy_loss(image) * vibrancy_scale
            
            cond_grad = torch.autograd.grad(loss, latents)[0]
            latents = latents.detach() - cond_grad
            
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
    with torch.no_grad():
        image = pipe.vae.decode(1 / 0.18215 * latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    return pipe.numpy_to_pil(image)

def initialize_pipeline():
    """Initialize the Stable Diffusion pipeline"""
    global pipe
    
    if pipe is None:
        print("Loading Stable Diffusion pipeline...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=dtype,
            use_safetensors=True,
            safety_checker=None
        ).to(device)
        
        # Performance optimizations
        if device == "cuda":
            pipe.enable_attention_slicing()
            # Try to use xformers if available
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("xformers enabled")
            except Exception:
                pass
        
        print(f"Pipeline loaded on {device} with dtype {dtype}")
    
    return pipe

def load_style(style_name):
    """Load a textual inversion style idempotently"""
    global current_style, pipe
    
    if pipe is None:
        initialize_pipeline()
    
    style_config = STYLES[style_name]
    token = style_config["token"]
    
    # Check if the token is already in the tokenizer to avoid ValueError
    if token not in pipe.tokenizer.get_vocab():
        print(f"Loading style: {style_name} with token {token}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # Load the inversion
            pipe.load_textual_inversion(style_config["repo"])
            # Crucial: move back to device as load_textual_inversion 
            # can sometimes mess with device placement of embeddings
            pipe.to(device)
            print(f"Style {style_name} loaded successfully")
        except Exception as e:
            print(f"Error loading style {style_name}: {e}")
            if "already in tokenizer vocabulary" in str(e):
                print(f"Token {token} already exists, skipping load.")
            else:
                raise e
    else:
        print(f"Style {style_name} (token {token}) already in tokenizer, skipping load.")
    
    current_style = style_name

def generate_image(prompt, style_name, seed, num_inference_steps, guidance_scale, contrast_scale, complexity_scale, vibrancy_scale, num_images=3):
    """Generate multiple images with the selected style"""
    try:
        load_style(style_name)
        style_token = STYLES[style_name]["token"]
        final_prompt = prompt.replace("<style>", style_token)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        
        # Robust numeric conversions with defaults
        try:
            seed_val = int(seed)
        except (ValueError, TypeError, NameError):
            seed_val = 42
            
        try:
            steps = int(num_inference_steps)
        except (ValueError, TypeError, NameError):
            steps = 50
            
        try:
            guidance = float(guidance_scale)
        except (ValueError, TypeError, NameError):
            guidance = 7.5
            
        try:
            contrast = float(contrast_scale)
        except (ValueError, TypeError, NameError):
            contrast = 0.0
            
        try:
            complexity = float(complexity_scale)
        except (ValueError, TypeError, NameError):
            complexity = 0.0
            
        try:
            vibrancy = float(vibrancy_scale)
        except (ValueError, TypeError, NameError):
            vibrancy = 0.0
            
        try:
            num_ims = int(num_images)
        except (ValueError, TypeError, NameError):
            num_ims = 3
            
        generator = torch.Generator(device=device).manual_seed(seed_val)
        print(f"Generating {num_ims} images: '{final_prompt}' with seed {seed_val}, guidance {guidance}, contrast {contrast}, complexity {complexity}, vibrancy {vibrancy}")
        
        if contrast > 0 or complexity > 0 or vibrancy > 0:
            images = custom_sampling_loop(final_prompt, pipe, guidance_scale=guidance, contrast_scale=contrast, complexity_scale=complexity, vibrancy_scale=vibrancy, num_inference_steps=steps, generator=generator, num_images=num_ims)
        else:
            result = pipe([final_prompt] * num_ims, num_inference_steps=steps, guidance_scale=guidance, generator=generator)
            images = result.images
            
        info_text = f"**Style:** {style_name}\n**Seed:** {seed_val}\n**Prompt:** {final_prompt}\n**Guidance:** {guidance}\n**Contrast:** {contrast}\n**Complexity:** {complexity}\n**Vibrancy:** {vibrancy}\n**Images Generated:** {len(images)}"
        return images, info_text
    except Exception as e:
        import traceback
        traceback.print_exc()
        return [], f"Error: {str(e)}"

def get_default_seed(style_name):
    if isinstance(style_name, str) and style_name in STYLES:
        return STYLES[style_name]["default_seed"]
    return 42

def generate_all_styles(prompt, seed1, seed2, seed3, seed4, seed5, seed6, num_inference_steps, guidance_scale, contrast_scale, complexity_scale, vibrancy_scale, num_images_per_style):
    """Generate multiple images for all 6 styles with individual seeds"""
    all_images = []
    info_texts = []
    seeds = [seed1, seed2, seed3, seed4, seed5, seed6]
    for idx, (style_name, seed) in enumerate(zip(STYLES.keys(), seeds)):
        style_images, info = generate_image(prompt, style_name, seed, num_inference_steps, guidance_scale, contrast_scale=contrast_scale, complexity_scale=complexity_scale, vibrancy_scale=vibrancy_scale, num_images=num_images_per_style)
        all_images.append(style_images[:])
        info_texts.append(info)
    return all_images[0], all_images[1], all_images[2], all_images[3], all_images[4], all_images[5], "\n\n---\n\n".join(info_texts)

# Create Gradio interface
with gr.Blocks(title="Stable Diffusion Style Explorer") as demo:
    
    gr.Markdown("""
    # 🎨 Stable Diffusion Style Explorer
    
    Generate images using different textual inversion styles from the SD Concepts Library.
    
    **Tip:** Use `<style>` in your prompt as a placeholder - it will be replaced with the appropriate style token.
    """)
    
    with gr.Tabs():
        # Tab 1: Single Style Generation
        with gr.Tab("Single Style"):
            with gr.Row():
                with gr.Column():
                    prompt_single = gr.Textbox(
                        label="Prompt",
                        placeholder="a grafitti in a favela wall with a <style> on it",
                        value="a grafitti in a favela wall with a <style>  on it",
                        lines=3
                    )
                    
                    style_dropdown = gr.Dropdown(
                        choices=list(STYLES.keys()),
                        value=list(STYLES.keys())[0],
                        label="Select Style"
                    )
                    
                    with gr.Row():
                        seed_single = gr.Textbox(
                            label="Seed",
                            value=STYLES[list(STYLES.keys())[0]]["default_seed"]
                        )
                        
                        steps_single = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=50,
                            step=1,
                            label="Inference Steps"
                        )
                    
                        guidance_single = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=7.5,
                            step=0.5,
                            label="Guidance Scale"
                        )
                        
                        with gr.Column(variant="panel"):
                            gr.Markdown("### 🎨 Loss Functions")
                            contrast_single = gr.Slider(
                                minimum=0,
                                maximum=2000,
                                value=0,
                                step=50,
                                label="Contrast Strength",
                                info="Steer generation towards higher contrast"
                            )
                            
                            complexity_single = gr.Slider(
                                minimum=0,
                                maximum=2000,
                                value=0,
                                step=50,
                                label="Complexity Strength",
                                info="Steer generation towards higher detail/edges"
                            )
                            
                            vibrancy_single = gr.Slider(
                                minimum=0,
                                maximum=2000,
                                value=0,
                                step=50,
                                label="Vibrancy Strength",
                                info="Steer generation towards higher saturation"
                            )
                        
                        num_images_single = gr.Slider(
                            minimum=1,
                            maximum=4,
                            value=3,
                            step=1,
                            label="Number of Images"
                        )
                    
                    generate_btn = gr.Button("Generate Images", variant="primary")
                
                with gr.Column():
                    output_gallery = gr.Gallery(label="Generated Images", show_label=False, elem_id="gallery", columns=3, object_fit="contain")
                    output_info = gr.Markdown()
            
            # Update seed when style changes
            style_dropdown.change(
                fn=get_default_seed,
                inputs=[style_dropdown],
                outputs=[seed_single],
                queue=False
            )
            
            generate_btn.click(
                fn=generate_image,
                inputs=[prompt_single, style_dropdown, seed_single, steps_single, guidance_single, contrast_single, complexity_single, vibrancy_single, num_images_single],
                outputs=[output_gallery, output_info]
            )
        
        # Tab 2: All Styles Comparison
        with gr.Tab("Compare All Styles"):
            gr.Markdown("""
            Generate the same prompt across all 6 styles.
            
            **Default seeds** are pre-configured for each style, but you can override them below.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_all = gr.Textbox(
                        label="Prompt",
                        placeholder="a grafitti in a favela wall with a <style>  on it",
                        value="a grafitti in a favela wall with a <style>  on it",
                        lines=3
                    )
                    
                    gr.Markdown("### 🎲 Seed Configuration")
                    gr.Markdown("*Each style has a default seed. Override below if desired.*")
                    
                    style_names = list(STYLES.keys())
                    with gr.Row():
                        seed1 = gr.Number(
                            label=f"🎨 {style_names[0]} Seed",
                            value=STYLES[style_names[0]]["default_seed"],
                            precision=0,
                            info=f"Default: {STYLES[style_names[0]]['default_seed']}"
                        )
                        seed2 = gr.Number(
                            label=f"🎨 {style_names[1]} Seed",
                            value=STYLES[style_names[1]]["default_seed"],
                            precision=0,
                            info=f"Default: {STYLES[style_names[1]]['default_seed']}"
                        )
                    
                    with gr.Row():
                        seed3 = gr.Number(
                            label=f"🎨 {style_names[2]} Seed",
                            value=STYLES[style_names[2]]["default_seed"],
                            precision=0,
                            info=f"Default: {STYLES[style_names[2]]['default_seed']}"
                        )
                        seed4 = gr.Number(
                            label=f"🎨 {style_names[3]} Seed",
                            value=STYLES[style_names[3]]["default_seed"],
                            precision=0,
                            info=f"Default: {STYLES[style_names[3]]['default_seed']}"
                        )
                    
                    with gr.Row():
                        seed5 = gr.Number(
                            label=f"🎨 {style_names[4]} Seed",
                            value=STYLES[style_names[4]]["default_seed"],
                            precision=0,
                            info=f"Default: {STYLES[style_names[4]]['default_seed']}"
                        )
                        seed6 = gr.Number(
                            label=f"🎨 {style_names[5]} Seed",
                            value=STYLES[style_names[5]]["default_seed"],
                            precision=0,
                            info=f"Default: {STYLES[style_names[5]]['default_seed']}"
                        )
                    
                    steps_all = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Inference Steps"
                    )
                    
                    guidance_all = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=7.5,
                        step=0.5,
                        label="Guidance Scale"
                    )

                    with gr.Column(variant="panel"):
                        gr.Markdown("### 🎨 Loss Functions")
                        contrast_all = gr.Slider(
                            minimum=0,
                            maximum=2000,
                            value=0,
                            step=50,
                            label="Contrast Strength",
                            info="Steer generation towards higher contrast"
                        )

                        complexity_all = gr.Slider(
                            minimum=0,
                            maximum=2000,
                            value=0,
                            step=50,
                            label="Complexity Strength",
                            info="Steer generation towards higher detail/edges"
                        )
                        
                        vibrancy_all = gr.Slider(
                            minimum=0,
                            maximum=2000,
                            value=0,
                            step=50,
                            label="Vibrancy Strength",
                            info="Steer generation towards higher saturation"
                        )
                    
                    num_images_all = gr.Slider(
                        minimum=1,
                        maximum=4,
                        value=3,
                        step=1,
                        label="Number of Images per Style"
                    )
                    
                    generate_all_btn = gr.Button("Generate All Styles", variant="primary")
            
            with gr.Row():
                style_names = list(STYLES.keys())
                output1 = gr.Gallery(label=style_names[0], columns=3, object_fit="contain")
                output2 = gr.Gallery(label=style_names[1], columns=3, object_fit="contain")
                output3 = gr.Gallery(label=style_names[2], columns=3, object_fit="contain")
            
            with gr.Row():
                output4 = gr.Gallery(label=style_names[3], columns=3, object_fit="contain")
                output5 = gr.Gallery(label=style_names[4], columns=3, object_fit="contain")
                output6 = gr.Gallery(label=style_names[5], columns=3, object_fit="contain")
            
            output_info_all = gr.Markdown()
            
            generate_all_btn.click(
                fn=generate_all_styles,
                inputs=[prompt_all, seed1, seed2, seed3, seed4, seed5, seed6, steps_all, guidance_all, contrast_all, complexity_all, vibrancy_all, num_images_all],
                outputs=[output1, output2, output3, output4, output5, output6, output_info_all]
            )
    
    gr.Markdown("""
    ---
    ### 📚 Available Styles
    """)
    
    for style_name, config in STYLES.items():
        gr.Markdown(f"**{style_name}**: {config['description']} | Token: `{config['token']}` | Default Seed: `{config['default_seed']}`")

# Initialize pipeline on startup
initialize_pipeline()

# Enable queue for long-running Stable Diffusion tasks (Required for Spaces)
demo.queue()

# Launch the app
if __name__ == "__main__":
    print("RUNNING THIS FILE:", __file__)
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())

