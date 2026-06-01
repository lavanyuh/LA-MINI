#  Image Convolution — Linear Algebra Mini Project

## Team Members 
- **Ipsa Ratish Mishra** -  PES1UG24CS191
- **Janhavi Ramakrishnan** - PES1IG24CS198
- **Jiya Datta Banik** - PES1UG24CS202
- **Sai Lavanya** - PES1UG24CS214

> A step-by-step interactive terminal application demonstrating blurring, sharpening, edge detection, and core linear algebra concepts through image convolution.

 

---

##  Overview

This project applies Linear Algebra concepts to real image processing using convolution kernels. Each step of the pipeline is explained mathematically, then visualized on an actual image — bridging theory and application.

Topics covered include:
- Matrix representation of images
- Convolution as a linear transformation
- Blur, sharpen, and edge detection kernels
- Eigenvalue analysis of kernels
- SVD-based image compression
- Orthogonality and inner products

---

##  Linear Algebra Concepts Demonstrated

| Concept | Where Used |
|---|---|
| Matrix addition | Adding noise to image |
| Inner product | Cross kernel (Average1) |
| Linear combination | Box kernel (Average2) |
| Linear transformation | Gaussian blur |
| Composition of transformations | Double Gaussian blur |
| Span | Large 5×5 kernel |
| Identity + operator | Sharpening (Sharp1) |
| Scaling / normalization | Sharpening (Sharp2) |
| Derivative operator | Sobel horizontal edge detection |
| Transpose | Sobel vertical (Sobel2 = Sobel1ᵀ) |
| Vector addition | Combined Sobel edges |
| Second derivative | Laplacian edge detection |
| Eigenvalues | Spectral analysis of kernel |
| SVD | Low-rank image compression |
| Orthogonality | Inner product check on singular vectors |

---
---

##  Requirements

**Python 3.8+** and the following libraries:

```bash
pip install numpy opencv-python matplotlib scipy
```

> Make sure `einstein.jpeg` is in the same folder as your project folder.

---

##  How to Run

```bash
python laa2.py
```

A terminal menu will launch. Run steps in order for best results — steps 3–17 require Step 1 (load image), and steps 3–6 require Step 2 (add noise).

---

## Step-by-Step Menu

| Step | Description |
|---|---|
| 1 | Load Image — represent image as a matrix |
| 2 | Add Noise — matrix addition |
| 3 | Average Blur — Cross Kernel (inner product) |
| 4 | Average Blur — Box Kernel (linear combination) |
| 5 | Gaussian Blur (linear transformation) |
| 6 | Gaussian Blur × 2 — Layered (composition) |
| 7 | Large 5×5 Kernel Blur (span) |
| 8 | Sharpen — Sharp1 (identity + operator) |
| 9 | Sharpen — Sharp2 (scaling/normalization) |
| 10 | Edge Detection — Sobel Horizontal (derivative) |
| 11 | Edge Detection — Sobel Vertical (transpose) |
| 12 | Edge Detection — Sobel Combined (vector addition) |
| 13 | Edge Detection — Laplacian (second derivative) |
| 14 | Q&A Answers Summary |
| 15 | Linearity of Convolution — verification |
| 16 | Eigenvalue Spectral Analysis of Kernel |
| 17 | SVD Compression — Low-Rank Approximation |

---
