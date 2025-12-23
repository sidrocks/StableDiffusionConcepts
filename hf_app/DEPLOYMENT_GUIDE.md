# Hugging Face Spaces Deployment Guide

## Quick Start - Deploy to Hugging Face Spaces

### Option 1: Web Upload (Easiest)

1. **Create a New Space**
   - Go to https://huggingface.co/new-space
   - Name your space (e.g., "stable-diffusion-style-explorer")
   - Select **Gradio** as the SDK
   - Choose your preferred visibility (Public/Private)
   - Click "Create Space"

2. **Upload Files**
   - Click "Files" tab in your new Space
   - Click "Add file" → "Upload files"
   - Upload these files from `hf_app/` folder:
     - `app.py`
     - `requirements.txt`
     - `README.md`
   - Click "Commit changes to main"

3. **Wait for Build**
   - The Space will automatically build (takes 5-10 minutes first time)
   - You'll see build logs in the "Logs" tab
   - Once complete, your app will be live!

### Option 2: Git Push (Advanced)

```bash
# Navigate to the hf_app directory
cd C:\Users\sidhe\TSAIV4\Session15-Assignment\hf_app

# Initialize git (if not already done)
git init

# Add Hugging Face Space as remote
# Replace YOUR_USERNAME and YOUR_SPACE_NAME
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME

# Add and commit files
git add .
git commit -m "Initial commit: Stable Diffusion Style Explorer"

# Push to Hugging Face
git push origin main
```

## Local Testing (Optional)

Test the app locally before deploying:

```bash
# Navigate to hf_app directory
cd C:\Users\sidhe\TSAIV4\Session15-Assignment\hf_app

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

The app will open at `http://localhost:7860`

## Hardware Requirements

### For Hugging Face Spaces:
- **Free Tier (CPU)**: Works but slow (~2-3 minutes per image)
- **Upgraded (GPU)**: Recommended for production
  - T4 GPU: ~10-15 seconds per image
  - A10G GPU: ~5-8 seconds per image

### To Upgrade Space Hardware:
1. Go to your Space settings
2. Click "Hardware" tab
3. Select GPU tier (requires payment)

## Troubleshooting

### Build Fails
- Check `requirements.txt` versions are compatible
- Review build logs in Spaces "Logs" tab
- Ensure all imports in `app.py` are in `requirements.txt`

### Out of Memory
- Reduce `num_inference_steps` default value
- Use CPU instead of GPU (slower but more memory)
- Upgrade to larger GPU tier

### Slow Generation
- Upgrade to GPU hardware
- Reduce inference steps (trade quality for speed)
- Consider caching the pipeline

## Customization

### Add More Styles
Edit `app.py` and add to the `STYLES` dictionary:

```python
STYLES = {
    # ... existing styles ...
    "Your Style Name": {
        "repo": "sd-concepts-library/your-concept",
        "token": "<your-token>",
        "description": "Your style description"
    }
}
```

### Change Base Model
Replace in `app.py`:
```python
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",  # or another model
    torch_dtype=dtype,
    safety_checker=None
).to(device)
```

### Adjust Default Parameters
Modify the default values in the Gradio components:
- `steps_single = gr.Slider(..., value=30, ...)` - Change default steps
- `guidance_single = gr.Slider(..., value=7.5, ...)` - Change guidance scale

## Next Steps

1. ✅ Deploy to Hugging Face Spaces
2. ✅ Test with different prompts and styles
3. ✅ Share your Space URL with others
4. ✅ Monitor usage in Space analytics
5. ✅ Iterate based on user feedback

## Support

- Hugging Face Spaces Docs: https://huggingface.co/docs/hub/spaces
- Gradio Documentation: https://gradio.app/docs
- Diffusers Documentation: https://huggingface.co/docs/diffusers
