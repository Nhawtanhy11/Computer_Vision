# Computer_Vision

Landmark Recognition System

This repository contains the implementation of a Landmark Recognition System, designed to identify specific unique landmarks in images at an instance level. The project addresses the challenges of the Google Landmark Recognition Competition hosted on Kaggle, with a focus on handling variations in lighting, camera angles, occlusions, and image quality.

📌 Project Overview

The objective of this project is to build a system that can accurately classify and name landmarks depicted in query images. The task involves distinguishing between over 100,000 global landmarks while managing significant intra-class variability, where images of the same landmark may include outdoor scenes, indoor views, or even artworks.

📊 Dataset: GLDv2-Use

We use a curated subset of the Google Landmarks Dataset v2 (GLDv2), the largest benchmark for large-scale instance recognition.

Due to the immense size of the original dataset (over 5 million images), the following refined versions were used:

GLDv2-Clean
Reduced to 1.5 million images using DELF and k-NN filtering.

GLDv2-Use
Final training set consisting of the top 100 most frequent landmarks.

Data Split

80% Training: 52,824 images

10% Validation: 6,606 images

10% Test: 6,651 images

Exploratory Data Analysis (EDA)

EDA revealed:

A long-tailed class distribution

Dominance of moderately bright images

Slight prevalence of warmer color hues

High concentration of blurry images (low Laplacian variance)

🛠️ Data Preprocessing

To improve model generalization, two distinct augmentation pipelines were implemented:

Pipeline 1

RandomResizedCrop

High-quality image compression

Preserves natural image features

Pipeline 2

Separate resizing and cropping

Image sharpening

Normalization using ImageNet statistics

🧠 Methodologies & Models

We explored four primary architectural approaches:

ResNet-50
A residual network baseline enhanced with a three-layer MLP classification head.

EfficientNet-B2
A lightweight, scalable architecture that balances accuracy and computational efficiency.

EfficientNet-B2 + DELF (Deep Local Features)

Two-phase pipeline: Embedding + Re-ranking

Captures fine-grained spatial details

Uses RANSAC for geometric verification

ResNet-50 + DOLG (Deep Orthogonal Local and Global Features)

Combines global semantic and local discriminative features

Orthogonal design reduces redundancy and improves robustness

🚧 Handling Out-of-Distribution (OOD) Images

To ensure real-world robustness, we implemented multiple strategies to detect non-landmark images (e.g., faces, animals, indoor scenes):

Binary Classifier

Pretrained MobileNetV2

Trained on non-landmark datasets:

Caltech101

CelebA

SUN397

ODIN Detector

Uses temperature scaling and input perturbations

Identifies OOD samples via confidence shifts

Feature Matching

Cosine similarity comparison

Matches input features against known landmark and non-landmark vectors

📈 Performance Results

Models were evaluated using Accuracy and Global Average Precision at 20 (GAP@20).

Model	Accuracy (%)	GAP@20
EfficientNet + DELF	98.45%	0.983
EfficientNet (Pipeline 1)	97.00%	0.9687
ResNet + DOLG	93.00%	0.9320
ResNet (Pipeline 1)	93.00%	0.9320

Key Insight:
Transfer learning is highly effective, and integrating DELF/DOLG significantly improves performance over baseline architectures.

🚀 Demonstration

The system is deployed using a Gradio interface. Users can upload an image and receive:

Similarity Check – Determines whether the image contains a landmark

Recognition Results – Predictions from all trained models with confidence scores

Top-5 Retrieved Images – Visually similar images from the training database

🧩 Analogy for Understanding

Think of this system as a world-class detective:

A standard model looks at the overall shape of a building (global features).

Our system, using DELF and DOLG, acts like a magnifying glass, identifying unique carvings, window styles, or structural details (local features).

Even if a photo is blurry or taken from an unusual angle, the system can still prove exactly which landmark it is.
