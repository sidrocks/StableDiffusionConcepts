# 🎨 Stable Diffusion Guided Style Explorer

This project demonstrates the power of **Textual Inversion** combined with **Artistic Guided Sampling** to steer Stable Diffusion v1-4 towards specific artistic styles and visual attributes.

## 🚀 Overview

The project consists of two main components:
1.  **Jupyter Notebook (`SD_conceptualizer_inference.ipynb`)**: A research-focused environment for experimenting with loss-based guidance and exploring the mechanics of Textual Inversion.
2.  **Gradio Web App (`hf_app/app.py`)**: A user-friendly interface for generating images with real-time control over artistic guidance.

---

## 🔬 Core Concepts

### 1. Textual Inversion
Instead of retraining the entire Stable Diffusion model, we use **Textual Inversion** to learn new "pseudo-words" (tokens) that represent specific concepts or styles.
*   **Token Injection**: By adding a learned embedding to the CLIP text encoder, we can invoke complex styles with a simple keyword like `<style>`.
*   **Integrated Styles**: The project includes styles from the SD Concepts Library such as:
    *   **Madhubani Art**: Traditional Indian folk art.
    *   **Cat Toy**: Plastic, cute aesthetic.
    *   **Seletti**: Porcelain/ceramic design.
    *   **Indian Watercolor**: Expressive portraits.
    *   **Chucky & Anime Boy**: Character-specific styles.

### 2. Guided Sampling (Artistic Steering)
While standard sampling follows the prompt, our **Guided Sampling** implementation injects extra gradients during the diffusion process to maximize specific visual features:

| Guidance Type | Technical Implementation | Artistic Effect |
| :--- | :--- | :--- |
| **Contrast** | Maximizes pixel variance from 0.5 | Dramatic lighting, deep shadows. |
| **Complexity** | Edge-detection (Sobel-like) gradients | Intricate details, sharp textures. |
| **Vibrancy** | Maximizes color channel variance | Vivid, punchy color saturation. |

---

## 🛠 Features

### 📓 Jupyter Notebook
*   **Step-by-Step implementation**: Detailed code for applying loss gradients during the UNet sampling loop.
*   **Reproducibility**: Uses fixed seeds (`torch.Generator`) to isolate the effect of guidance scales, ensuring that changes in output are purely due to artistic steering.
*   **Deep Dive Documentation**: Explains the math and logic behind each loss function.

## 📱 User Guide: Gradio App

The **Stable Diffusion Style Explorer** provides two distinct ways to interact with the model.

### 1. Single Style Generation
Use this tab for granular control over a specific aesthetic.
*   **Select Style**: Choose one of the 5+ pre-loaded textual inversion concepts.
*   **Prompt**: Write your prompt using the `<style>` placeholder. (e.g., `"a futuristic city in the style of <style>"`).
*   **Artistic Sliders**:
    *   **Contrast Strength**: Increase to add drama and deeper shadows.
    *   **Complexity Strength**: Increase to force intricate patterns and fine details.
    *   **Vibrancy Strength**: Increase for more saturated and "glowing" colors.
*   **Seed Management**: Each style comes with a pre-configured "best" seed, but you can override it to explore different variations.

### 2. Compare All Styles
Use this tab to see how a single prompt manifests across different artistic interpretations.
*   **Batch Generation**: Generates 3 images (default) for every single style simultaneously.
*   **Unified Guidance**: Apply the same Contrast, Complexity, and Vibrancy scales across all styles to compare their response to guidance.
*   **Style-Specific Seeds**: Configure individual seeds for each style to ensure reproducibility across separate runs.

---

## 🚦 Getting Started

### Prerequisites
*   Python 3.10+
*   GPU with 8GB+ VRAM (Recommended)
*   Hugging Face Token (for model and style loading)

### Setup
1.  **Clone the repository**.
2.  **Environment Setup**:
    ```bash
    cd hf_app
    pip install -r requirements.txt
    ```
3.  **Run the App**:
    ```bash
    python app.py
    ```

## 💡 Tips for Best Results
*   **Guidance Scales**: Typical effective values for Contrast/Complexity/Vibrancy range from **500 to 1500**. Start low and increase gradually.
*   **Prompting**: Keep prompts relatively simple to let the Textual Inversion style shine.
*   **Seeds**: If you find an image layout you like, keep the seed fixed while adjusting the loss sliders to see exactly how the guidance "sculpts" that specific composition.

## Demo
https://huggingface.co/spaces/sidharthg/SDConcepts
<img width="1915" height="1027" alt="Screenshot 2025-12-23 215106" src="https://github.com/user-attachments/assets/b4779b83-2e35-4ab0-a9bd-7e732eadbb83" />


## 📜 Credits
*   **Model**: Stable Diffusion v1-4
*   **Concepts**: 🤗 Hugging Face [SD Concepts Library](https://huggingface.co/sd-concepts-library)
*   **Implementation**: Custom Triple-Loss Guidance Suite.



