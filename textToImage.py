# Import required libraries
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel, BertTokenizer, BertForSequenceClassification
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# Load Stable Diffusion for Text-to-Image Generation
stable_diffusion_model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda" if torch.cuda.is_available() else "cpu")

# Load CLIP for Image-Text Alignment
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load BERT for Text-Text Alignment
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# Sample C3 Dataset (Cross-Cultural Prompts)
data = pd.DataFrame({
    "id": [1, 2, 3, 4, 5, 6],
    "caption": [
        "A student in uniform going to school",
        "An Indian bride in red saree with gold jewelry",
        "A street vendor selling tacos in a Mexican market",
        "A Maasai warrior in Kenya holding a spear",
        "A Chinese dragon dance performed during the Lunar New Year",
        "A Moroccan street filled with colorful rugs and lamps"
    ],
    "culture": ["Japan", "India", "Mexico", "Kenya", "China", "Morocco"],
    "ground_truth_object": [
        ["uniform", "school"],
        ["saree", "jewelry"],
        ["tacos", "market"],
        ["spear", "shield"],
        ["dragon", "crowd"],
        ["rugs", "lamps"]
    ]
})

# Generate images using Stable Diffusion
def generate_image(prompt):
    image = stable_diffusion_model(prompt).images[0]
    return image

# Evaluate image-text similarity using CLIP
def evaluate_clip_similarity(image, text):
    inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    return float(logits_per_image[0][0].item())

# Generate and evaluate images
results = []
for i, row in data.iterrows():
    prompt = row["caption"]
    culture = row["culture"]
    ground_truth = row["ground_truth_object"]
    
    # Generate Image
    generated_image = generate_image(prompt)
    
    # Save image for visualization
    img_path = f"image_{row['id']}.png"
    generated_image.save(img_path)
    
    # Evaluate using CLIP (Image-Text Alignment)
    clip_score = evaluate_clip_similarity(generated_image, prompt)
    
    # Dummy Object Detection Score (Simulated for Demo)
    object_alignment_score = np.random.uniform(0.5, 1.0)
    
    # Final Score: Weighted Average of I-T and O-T
    final_score = 0.7 * clip_score + 0.3 * object_alignment_score
    
    # Append results
    results.append({
        "id": row["id"],
        "caption": prompt,
        "culture": culture,
        "clip_score": clip_score,
        "object_score": object_alignment_score,
        "final_score": final_score,
        "image_path": img_path
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Plot CLIP Scores Across Cultures
plt.figure(figsize=(10, 6))
sns.barplot(x="culture", y="clip_score", data=results_df, palette="viridis")
plt.title("CLIP Scores Across Cultures in C3 Benchmark")
plt.xlabel("Culture")
plt.ylabel("CLIP Score (0 - Low, 1 - High)")
plt.xticks(rotation=45)
plt.show()

# Plot Final Scores Distribution
plt.figure(figsize=(8, 5))
sns.histplot(results_df["final_score"], kde=True, bins=10, color="blue")
plt.title("Distribution of Final Scores in C3 Benchmark")
plt.xlabel("Final Score")
plt.ylabel("Frequency")
plt.show()

# Save results to CSV
results_df.to_csv("c3_benchmark_results.csv", index=False)
print("C3 Benchmark Results Saved Successfully! ✅")
