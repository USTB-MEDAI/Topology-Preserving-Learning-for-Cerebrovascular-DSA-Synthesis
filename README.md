# Topology-Preserving Learning for Cerebrovascular DSA Synthesis (SS-LDM)
## Overview
This repository contains the implementation of the Structure-preserving Sketch-guided Latent Diffusion Model (SS-LDM), a groundbreaking approach designed to synthesize high-fidelity cerebrovascular Digital Subtraction Angiography (DSA) images from sparse hand-drawn sketches. SS-LDM harnesses the power of multimodal sparse conditioning and state-of-the-art latent diffusion modeling techniques, effectively generating accurate vascular images while addressing the challenges of data scarcity in medical imaging.

## Research Context
Cerebrovascular diseases significantly contribute to global morbidity and mortality rates, making accurate diagnosis and effective treatment critical. Traditional DSA imaging, despite being the standard for these assessments, is limited by ethical concerns regarding radiation exposure and a lack of comprehensive data for training AI models. The SS-LDM framework aims to mitigate these issues by utilizing minimal sketch inputs to guide the generation of high-quality DSA images, ultimately enhancing the feasibility of intelligent diagnosis and visual therapy in clinical settings.

The SS-LDM framework has been rigorously evaluated across several public datasets, demonstrating superior performance in terms of structural consistency and image quality compared to both classical methods and existing state-of-the-art generative models.

## Features
- **Multimodal Sparse Condition Generation**: Utilizes three levels of hand-drawn structural sketches to guide the generation process from minimal details to complete vascular structures.
- **Two-Stage Generation Process**: Decouples structural and texture modeling to improve accuracy and generative control.
- **Skeleton-Preserving Regularization**: Introduces topological consistency measures through a novel auxiliary pathway, enhancing the anatomical realism of generated images.
- **Open-source Implementation**: Facilitates reproducibility and further advancements in research related to cerebrovascular imaging.

## Getting Started
To begin using SS-LDM, please follow the installation instructions and usage guidelines outlined below. Ensure that the necessary dependencies are met for proper functionality.

Feel free to tailor any section to better fit your specific context or to include further details that may be relevant!
