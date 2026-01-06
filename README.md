# Computer_Vision
##Landmark Recognition System

This repository implements a Landmark Recognition System for instance-level image recognition. The project is inspired by the Google Landmark Recognition Competition (Kaggle) and focuses on handling real-world challenges such as variations in lighting, camera angles, occlusions, and image quality.

##Project Overview

The goal of this project is to build a system capable of accurately identifying and classifying landmarks from query images. The task requires distinguishing between 100,000+ global landmarks while managing significant intra-class variability, where images of the same landmark may include outdoor views, indoor scenes, or artistic depictions.

## Dataset: GLDv2-Use

We use a curated subset of the Google Landmarks Dataset v2 (GLDv2), the largest benchmark for large-scale instance recognition.

Due to the original dataset’s massive size (over 5 million images), the following refined datasets were used:

GLDv2-Clean
Reduced to 1.5 million images using DELF and k-NN filtering.

GLDv2-Use
Final training set consisting of the top 100 most frequent landmarks.

Data Split

Training: 80% (52,824 images)

Validation: 10% (6,606 images)

Test: 10% (6,651 images)

Exploratory Data Analysis (EDA)

EDA revealed the following characteristics of the dataset:

Long-tailed class distribution

Dominance of moderately bright images

Slight prevalence of warmer color hues

High concentration of blurry images (low Laplacian variance)

🛠️ Data Preprocessing

To improve model generalization, two data augmentation pipelines were implemented:

Pipeline 1

RandomResizedCrop

High-quality image compression

Preserves natural image features

Pipeline 2

Separate resizing and cropping

Image sharpening

Normalization using ImageNet statistics

## Methodologies & Models

Four primary architectural approaches were explored:

1. ResNet-50

Residual network baseline

Enhanced with a three-layer MLP classification head

2. EfficientNet-B2

Lightweight and scalable architecture

Balances accuracy and computational efficiency

3. EfficientNet-B2 + DELF (Deep Local Features)

Two-phase pipeline: Embedding + Re-ranking

Captures fine-grained spatial details

Uses RANSAC for geometric verification

4. ResNet-50 + DOLG (Deep Orthogonal Local and Global Features)

Combines global semantic and local discriminative features

Orthogonal design reduces redundancy and improves robustness

## Handling Out-of-Distribution (OOD) Images

To ensure robustness in real-world scenarios, multiple strategies were implemented to detect non-landmark images (e.g., faces, animals, indoor scenes):

Binary Classifier

Pretrained MobileNetV2

Trained on non-landmark datasets:

Caltech101

CelebA

SUN397

ODIN Detector

Uses temperature scaling and input perturbations

Detects OOD samples via confidence shifts

Feature Matching

Cosine similarity-based comparison

Matches input features against known landmark and non-landmark embeddings

## Performance Results

Models were evaluated using Accuracy and Global Average Precision at 20 (GAP@20).

Model	Accuracy (%)	GAP@20
EfficientNet + DELF	98.45%	0.983
EfficientNet (Pipeline 1)	97.00%	0.9687
ResNet + DOLG	93.00%	0.9320
ResNet (Pipeline 1)	93.00%	0.9320

Key Insight:
Transfer learning is highly effective, and integrating DELF/DOLG significantly outperforms baseline architectures.

## Demonstration

The system is deployed using a Gradio interface. Users can upload an image and receive:

Similarity Check – Determines whether the image contains a landmark

Recognition Results – Predictions from all trained models with confidence scores

Top-5 Retrieved Images – Visually similar matches from the training database

## Analogy for Understanding

Think of this system as a world-class detective:

A standard model observes the overall shape of a building (global features).

Our system, powered by DELF and DOLG, acts like a magnifying glass, identifying unique carvings, window styles, or architectural details (local features).

Even when a photo is blurry or taken from an unusual angle, the system can still precisely identify the landmark.
