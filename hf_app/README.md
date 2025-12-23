---
title: Stable Diffusion Style Explorer
emoji: 🎨
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
pinned: false
license: mit
---

# Stable Diffusion Style Explorer

An interactive web application for exploring different artistic styles using Stable Diffusion with textual inversion.

## Features

- **5 Pre-configured Styles**: Cat Toy, GTA5 Artwork, Birb Style, Midjourney Style, and Arcane Style
- **Single Style Mode**: Generate images with a specific style, custom seed, and parameters
- **Compare All Styles**: Generate the same prompt across all 5 styles simultaneously
- **Seed Control**: Full control over random seeds for reproducible results
- **Adjustable Parameters**: Configure inference steps and guidance scale

## Usage

### Single Style Mode
1. Enter your prompt (use `<style>` as a placeholder for the style token)
2. Select a style from the dropdown
3. Set your desired seed value
4. Adjust inference steps and guidance scale if needed
5. Click "Generate Image"

### Compare All Styles Mode
1. Enter your prompt (use `<style>` as a placeholder)
2. Set a base seed value
3. Each style will use: `base_seed + (style_index * 100)`
4. Click "Generate All Styles" to see all variations

## Styles

- **Cat Toy**: Cute cat toy aesthetic
- **GTA5 Artwork**: GTA V game art style
- **Birb Style**: Artistic bird illustration style
- **Midjourney Style**: Midjourney AI art aesthetic
- **Arcane Style**: Arcane Netflix series art style

## Technical Details

- **Base Model**: CompVis/stable-diffusion-v1-4
- **Textual Inversion**: Concepts from [SD Concepts Library](https://huggingface.co/sd-concepts-library)
- **Framework**: Gradio + Diffusers
- **GPU**: Recommended for faster generation

## Local Development

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-name>

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

## Deployment to Hugging Face Spaces

1. Create a new Space on Hugging Face
2. Select "Gradio" as the SDK
3. Upload `app.py`, `requirements.txt`, and `README.md`
4. The app will automatically build and deploy

## License

MIT License - Feel free to use and modify!
