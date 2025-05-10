# 🌐 C3 Benchmark: Cross-Cultural Image Generation and Evaluation Framework

This repository presents a novel pipeline for evaluating culturally diverse prompts using image generation and alignment tools. The project aims to highlight how well text-to-image models like Stable Diffusion can capture cultural context when paired with semantic evaluation tools like CLIP.

---

## 📦 Features

- ✨ Generate high-quality images using [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- 🤝 Evaluate alignment between text and image using [CLIP](https://huggingface.co/openai/clip-vit-base-patch32)
- 📊 Analyze performance metrics across cultures with Seaborn plots
- 🧪 Simulate object-level alignment scores for multi-modal benchmarking
- 💾 Export results and visualizations for downstream tasks

---

## 📂 Dataset

A miniature Cross-Cultural Captions (C3) dataset is used containing:
- **Text prompts** describing cultural scenes (e.g., "An Indian bride in red saree")
- **Associated culture** (e.g., India, Kenya, China)
- **Ground truth objects** expected in the scene (e.g., saree, jewelry)

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/c3-benchmark.git
cd c3-benchmark
