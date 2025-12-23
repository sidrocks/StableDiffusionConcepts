# Quick Start Guide - Local Testing

## Installation Complete! ✅

All dependencies have been installed successfully. Here's what to do next:

## Running the App

### Option 1: Simple Run
```bash
python app.py
```

The app will start and show you a URL like:
```
Running on local URL:  http://127.0.0.1:7860
```

Open that URL in your browser!

### Option 2: Share Publicly (Temporary)
```bash
# Edit app.py, change the last line to:
demo.launch(share=True)
```

This creates a temporary public URL you can share with others.

## What to Expect

### First Run
- The app will download the Stable Diffusion model (~4GB)
- This happens only once - subsequent runs are fast
- Download location: `~/.cache/huggingface/`

### Performance
- **With GPU (CUDA)**: ~10-15 seconds per image
- **Without GPU (CPU)**: ~2-3 minutes per image

Check if CUDA is available:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Using the App

### Single Style Tab
1. Enter a prompt (use `<style>` as placeholder)
   - Example: `"a portrait of a warrior in <style>"`
2. Select a style from dropdown
3. Set seed (e.g., 42)
4. Click "Generate Image"

### Compare All Styles Tab
1. Enter a prompt with `<style>` placeholder
2. Set base seed (e.g., 100)
3. Click "Generate All Styles"
4. See all 5 styles side-by-side!

## Troubleshooting

### "Out of Memory" Error
- Reduce inference steps to 20-30
- Close other GPU applications
- Use CPU mode (slower but works)

### Slow Generation
- This is normal on CPU
- Consider using GPU for faster results
- Reduce inference steps for speed

### Model Download Fails
- Check internet connection
- Ensure ~5GB free disk space
- Try again - downloads resume automatically

## Next Steps

1. ✅ Test the app locally
2. ✅ Try different prompts and styles
3. ✅ Deploy to Hugging Face Spaces (see DEPLOYMENT_GUIDE.md)

Enjoy your Stable Diffusion Style Explorer! 🎨
