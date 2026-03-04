# PMSAC-OVF: Radiomics Module

This repository contains the source code for the **Radiomics analysis module** of the paper: **"PMSAC-OVF: a fully automated multi-institutional system for predicting osteoporotic vertebral fracture using paraspinal muscle signatures from lumbar MRI"**.

This module is designed to extract high-dimensional quantitative features from Paraspinal Muscles (PM) regions in lumbar MRI scans and construct machine learning models to predict the risk of Osteoporotic Vertebral Fracture (OVF).

## 📋 Features

*   **Feature Extraction**: Extracts texture, shape, and first-order statistical features from MRI images (ROIs: Psoas Major, Erector Spinae-Multifidus complex) using `PyRadiomics`.
*   **Preprocessing**: Includes voxel resampling, intensity normalization (Z-score), and discretization.
*   **Feature Selection**: Implements a rigorous 3-step selection strategy:
    1.  Mann-Whitney U test (univariate analysis)
    2.  Pearson correlation analysis (removing redundant features)
    3.  LASSO regression (feature sparsification and selection)
*   **Modeling**: Constructs an ensemble model using XGBoost and calculates the final **Radiomics Signature (RS)**.

