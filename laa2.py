"""
Mini Project: Blurring, Sharpening, and Linear Algebra of Images Using Convolution
Linear Algebra Application — lab06_full.py

Run this file. A terminal menu will guide you step by step.
Each step prints the kernel matrix, explains it, then shows the image.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import os
import sys

# ═══════════════════════════════════════════════════════════════════════
# TERMINAL COLOURS  (work on Windows cmd, PowerShell, Mac, Linux)
# ═══════════════════════════════════════════════════════════════════════
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    CYAN    = "\033[96m"
    YELLOW  = "\033[93m"
    GREEN   = "\033[92m"
    RED     = "\033[91m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"
    BG_BLUE = "\033[44m"
    BG_DARK = "\033[40m"

# Enable ANSI on Windows
if sys.platform == "win32":
    os.system("color")

# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def hr(char="─", width=62, color=C.CYAN):
    print(f"{color}{char * width}{C.RESET}")

def section(title):
    print()
    hr("═")
    print(f"{C.BG_BLUE}{C.BOLD}{C.WHITE}  {title:<60}{C.RESET}")
    hr("═")

def info(label, text):
    print(f"  {C.YELLOW}{C.BOLD}{label:<18}{C.RESET} {C.WHITE}{text}{C.RESET}")

def bullet(text, color=C.GREEN):
    print(f"  {color}▸{C.RESET} {text}")

def print_matrix(matrix, name, color=C.CYAN):
    """Pretty-print a numpy matrix with borders."""
    print(f"\n  {C.BOLD}{color}{name}:{C.RESET}")
    rows, cols = matrix.shape
    col_w = max(len(f"{v:.4f}") for v in matrix.flatten()) + 2
    hr("┄", 62, C.DIM)
    for r in range(rows):
        row_str = "  │ "
        for c_idx in range(cols):
            val = matrix[r, c_idx]
            formatted = f"{val:>{col_w}.4f}"
            if val > 0:
                row_str += f"{C.GREEN}{formatted}{C.RESET}"
            elif val < 0:
                row_str += f"{C.RED}{formatted}{C.RESET}"
            else:
                row_str += f"{C.DIM}{formatted}{C.RESET}"
            row_str += "  "
        row_str += "│"
        print(row_str)
    hr("┄", 62, C.DIM)
    r_c, c_c = matrix.shape
    total = matrix.sum()
    print(f"  {C.DIM}  Shape: {r_c}×{c_c}   |   Sum of entries: {total:.4f}{C.RESET}")

def print_vector(vec, name, color=C.CYAN):
    """Pretty-print a 1D numpy array."""
    print(f"\n  {C.BOLD}{color}{name}:{C.RESET}")
    hr("┄", 62, C.DIM)
    row_str = "  │ "
    col_w = max(len(f"{v:.4f}") for v in vec) + 2
    for val in vec:
        formatted = f"{val:>{col_w}.4f}"
        if val > 0:
            row_str += f"{C.GREEN}{formatted}{C.RESET}  "
        elif val < 0:
            row_str += f"{C.RED}{formatted}{C.RESET}  "
        else:
            row_str += f"{C.DIM}{formatted}{C.RESET}  "
    row_str += "│"
    print(row_str)
    hr("┄", 62, C.DIM)

def wait(msg="  Press ENTER to continue..."):
    print(f"\n{C.DIM}{msg}{C.RESET}")
    input()

def show_image(img, title, cmap=None):
    """Display image in a matplotlib window."""
    plt.close('all')
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#111111')
    ax.imshow(img, cmap=cmap)
    ax.set_title(title, color='white', fontsize=13, pad=10)
    ax.axis('off')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

def apply_rgb(base, kernel):
    out = np.zeros_like(base, dtype=np.float64)
    for ch in range(3):
        out[:, :, ch] = convolve2d(base[:, :, ch], kernel,
                                   mode='same', boundary='symm')
    return np.uint8(np.clip(out, 0, 255))

def apply_gray(gray, kernel):
    out = convolve2d(gray, kernel, mode='same', boundary='symm')
    return np.uint8(np.clip(out, 0, 255))

# ═══════════════════════════════════════════════════════════════════════
# MENU DEFINITION
# ═══════════════════════════════════════════════════════════════════════

MENU_ITEMS = [
    " 1  Load Image",
    " 2  Add Noise (adding two matrices)",
    " 3  Average Blur  —  Cross Kernel (Average1) (inner product U2)",
    " 4  Average Blur  —  Box Kernel   (Average2) (linear combination U2)",
    " 5  Gaussian Blur (linear transformation U2)",
    " 6  Gaussian Blur × 2  (Layered) (composition U2)",
    " 7  Large Kernel Blur  (5×5) (span U2)",
    " 8  Sharpen  —  Kernel Sharp1 (identity + operator U2)",
    " 9  Sharpen  —  Kernel Sharp2 (scaling/norm U2)",
    "10  Edge Detection  —  Sobel Horizontal (derivative operator U2)",
    "11  Edge Detection  —  Sobel Vertical (transpose U1)",
    "12  Edge Detection  —  Sobel Combined (vector addition U2)",
    "13  Edge Detection  —  Laplacian (second derivative U2)",
    "14  Q&A Answers Summary",
    "15  Linear Transformation  —  Linearity of Convolution",
    "16  Eigenvalues  —  Spectral Analysis of Kernel (applying a filter)",
    "17  SVD Compression  —  Low-Rank Approximation ",
    " 0  Exit",
]

def print_menu(computed):
    print()
    hr("═")
    print(f"{C.BOLD}{C.CYAN}  IMAGE CONVOLUTION — STEP-BY-STEP MENU{C.RESET}")
    hr("─")
    for item in MENU_ITEMS:
        num = item.strip().split()[0]
        try:
            idx = int(num)
        except ValueError:
            idx = -1
        done = f"{C.GREEN}✔{C.RESET}" if idx in computed else " "
        print(f"  {done} {C.WHITE}{item}{C.RESET}")
    hr("═")
    print(f"  {C.DIM}Run steps in order for best results. Steps 3–13 require Step 1.{C.RESET}")
    print()

# ═══════════════════════════════════════════════════════════════════════
# STEPS 1–14  (Core Image Convolution Pipeline)
# ═══════════════════════════════════════════════════════════════════════

def step_load():
    section("STEP 1 — Load Image")
    bullet("We represent the image as a matrix of pixel values.")
    bullet("Each pixel is a number (0–255) per colour channel (R, G, B).")
    bullet("The image matrix has shape:  m × n × 3")
    print()

    img = cv2.imread('einstein.jpeg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"  {C.GREEN}Loaded: einstein.jpeg{C.RESET}")

    m, n, _ = img.shape
    print()
    info("Image shape:",  f"{m} rows  ×  {n} cols  ×  3 channels")
    info("Total pixels:", f"{m * n:,}")
    info("Data type:",    f"{img.dtype}  (values 0–255)")
    info("Memory size:",  f"{img.nbytes / 1024:.1f} KB")
    print()
    bullet("Showing a small sample of the RED channel (top-left 5×5 patch):")
    print_matrix(img[:5, :5, 0].astype(float), "Red channel — top-left 5×5 patch", C.RED)

    wait()
    show_image(img, "Step 1 — Original Image")
    print(f"  {C.GREEN}▸ Image displayed in window.{C.RESET}")
    wait()
    return img

def step_noise(ImJPG):
    m, n, _ = ImJPG.shape
    section("STEP 2 — Add Noise")
    bullet("We simulate a real-world noisy image by adding a random matrix N.")
    bullet("N has the same shape as the image: m × n × 3")
    bullet("Each entry of N is drawn from the interval (−25, +25).")
    print()
    print(f"  {C.WHITE}Formula:{C.RESET}")
    print(f"  {C.CYAN}  A_noisy  =  A  +  N{C.RESET}")
    print(f"  {C.DIM}  where N = 50 × (rand(m,n,3) − 0.5)   →   range: (−25, +25){C.RESET}")
    print()
    info("Noise amplitude:", "±25 shades per channel")
    info("rand() range:",    "(0, 1)  →  shifted to (−0.5, +0.5)  →  scaled to (−25, +25)")
    print()
    bullet("Why convert to float64 first?")
    print(f"  {C.DIM}  uint8 wraps around at 255. Float lets us add negative noise safely.")
    print(f"  {C.DIM}  We convert back to uint8 after filtering (with clipping).{C.RESET}")
    print()

    np.random.seed(42)
    noise = 50 * (np.random.rand(m, n, 3) - 0.5)
    noisy = np.double(ImJPG) + noise

    bullet("Sample noise values (top-left 5×5 patch, channel R):")
    print_matrix(noise[:5, :5, 0], "Noise N — Red channel top-left 5×5", C.YELLOW)

    wait()
    show_image(np.uint8(np.clip(noisy, 0, 255)), "Step 2 — Noisy Image")
    print(f"  {C.GREEN}▸ Noisy image displayed.{C.RESET}")
    wait()
    return noisy

def step_average1(noisy):
    section("STEP 3 — Average Blur: Cross Kernel (Kernel_Average1)")
    bullet("A convolution filter works by sliding a small matrix (kernel) over the image.")
    bullet("Each output pixel is the weighted sum of its neighbours.")
    print()
    print(f"  {C.WHITE}Formula (convolution):{C.RESET}")
    print(f"  {C.CYAN}  g_ij  =  Σ_kl  f_kl · h_(i−k+1, j−l+1){C.RESET}")
    print()

    K = np.array([[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]], dtype=float) / 5
    print_matrix(K, "Kernel_Average1  (cross shape, ÷5)", C.CYAN)
    print()
    bullet("Shape: 3×3   |   Non-zero entries: 5   |   Sum: 1.0")
    bullet("Only the centre + 4 direct neighbours are used (up/down/left/right).")
    bullet("Dividing by 5 ensures the kernel weights sum to 1 → brightness preserved.")
    print()
    print(f"  {C.YELLOW}Q1:{C.RESET} Size of ImJPG_Average1 = same as original (mode='same' preserves shape)")
    print(f"  {C.YELLOW}Q3:{C.RESET} This blurs LESS than Average2 — only 5 pixels vs 9.")

    result = apply_rgb(noisy, K)
    wait()
    show_image(result, "Step 3 — Blur: Cross Kernel (Average1)")
    print(f"  {C.GREEN}▸ Cross-blur image displayed.{C.RESET}")
    wait()
    return result

def step_average2(noisy):
    section("STEP 4 — Average Blur: Box Kernel (Kernel_Average2)")

    K = np.array([[1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1]], dtype=float) / 9
    print_matrix(K, "Kernel_Average2  (box / all-ones, ÷9)", C.CYAN)
    print()
    bullet("Shape: 3×3   |   Non-zero entries: 9   |   Sum: 1.0")
    bullet("All 9 pixels in the 3×3 window are averaged equally.")
    bullet("Dividing by 9 normalises the kernel — brightness is preserved.")
    print()
    print(f"  {C.YELLOW}Q1:{C.RESET} Size of ImJPG_Average2 = same as original.")
    print(f"  {C.YELLOW}Q3:{C.RESET} This blurs MORE than Average1.")
    print(f"  {C.DIM}  Reason: it averages ALL 9 neighbours equally, losing more")
    print(f"  {C.DIM}  high-frequency detail. Average1 only uses 5, keeping the")
    print(f"  {C.DIM}  centre pixel more dominant.{C.RESET}")

    result = apply_rgb(noisy, K)
    wait()
    show_image(result, "Step 4 — Blur: Box Kernel (Average2)")
    print(f"  {C.GREEN}▸ Box-blur image displayed.{C.RESET}")
    wait()
    return result

def step_gauss(noisy):
    section("STEP 5 — Gaussian Blur")
    bullet("Gaussian blur assigns HIGHER weight to the centre pixel.")
    bullet("This gives a smoother, more natural-looking result than box blur.")
    print()

    K = np.array([[0, 1, 0],
                  [1, 4, 1],
                  [0, 1, 0]], dtype=float) / 8
    print_matrix(K, "Kernel_Gauss  (centre-weighted, ÷8)", C.CYAN)
    print()
    bullet("Centre weight: 4/8 = 0.5   |   Each side: 1/8 = 0.125")
    bullet("Sum of all weights = 1.0  → brightness preserved.")
    bullet("Approximates a 2D Gaussian (bell curve) distribution.")
    print()
    bullet("Compare with Average2:")
    print(f"  {C.DIM}  Average2: all weights = 1/9 ≈ 0.111  (uniform)")
    print(f"  {C.DIM}  Gauss:    centre = 0.5, edges = 0.125 (tapered){C.RESET}")

    result = apply_rgb(noisy, K)
    wait()
    show_image(result, "Step 5 — Gaussian Blur")
    print(f"  {C.GREEN}▸ Gaussian blur image displayed.{C.RESET}")
    wait()
    return result

def step_gauss2(gauss1):
    section("STEP 6 — Gaussian Blur × 2  (Layered Convolution)")
    bullet("We apply the SAME Gaussian kernel a second time to the already-blurred image.")
    bullet("This is called 'layering' or 'chaining' convolutions.")
    print()

    K = np.array([[0, 1, 0],
                  [1, 4, 1],
                  [0, 1, 0]], dtype=float) / 8

    K2 = convolve2d(K, K, mode='full')
    print_matrix(K, "Kernel_Gauss  (applied again)", C.CYAN)
    print()
    print(f"  {C.YELLOW}Q4:{C.RESET} A single equivalent kernel = Kernel_Gauss ★ Kernel_Gauss")
    print_matrix(K2, "Equivalent single kernel  (Gauss ★ Gauss)", C.MAGENTA)
    print()
    print(f"  {C.YELLOW}Q5:{C.RESET} Size of this equivalent kernel = {K2.shape}")
    print(f"  {C.DIM}  Full conv of two 3×3 matrices:")
    print(f"  {C.DIM}  (3 + 3 − 1) × (3 + 3 − 1)  =  5 × 5{C.RESET}")

    result = apply_rgb(gauss1.astype(np.float64), K)
    wait()
    show_image(result, "Step 6 — Gaussian Blur × 2")
    print(f"  {C.GREEN}▸ Double Gaussian image displayed.{C.RESET}")
    wait()
    return result

def step_large(ImJPG):
    section("STEP 7 — Large Kernel Blur (5×5)")
    bullet("A larger kernel covers a wider area → stronger, wider blur in one pass.")
    print()

    K = np.array([[0, 1, 2, 1, 0],
                  [1, 4, 8, 4, 1],
                  [2, 8,16, 8, 2],
                  [1, 4, 8, 4, 1],
                  [0, 1, 2, 1, 0]], dtype=float) / 80
    print_matrix(K, "Kernel_Large  (5×5 Gaussian-like, ÷80)", C.CYAN)
    print()
    bullet("Centre weight: 16/80 = 0.200  |  Sum of all entries = 1.0")
    bullet("Applied to the ORIGINAL image (not the noisy one).")
    print()
    print(f"  {C.YELLOW}Q6:{C.RESET} Kernel_Large blurs MORE than double Gaussian.")
    print(f"  {C.DIM}  - Kernel_Large: single 5×5 pass, wide spatial reach")
    print(f"  {C.DIM}  - Double Gauss: two sequential 3×3 passes, narrower reach")
    print(f"  {C.DIM}  Larger kernel window = more pixel information averaged = more blur.{C.RESET}")

    result = apply_rgb(ImJPG.astype(np.float64), K)
    wait()
    show_image(result, "Step 7 — Large Kernel Blur")
    print(f"  {C.GREEN}▸ Large kernel blur displayed.{C.RESET}")
    wait()
    return result

def step_sharp1(ImJPG):
    section("STEP 8 — Sharpening: Kernel_Sharp1")
    bullet("Sharpening enhances edges by amplifying the difference between")
    bullet("a pixel and its neighbours.")
    print()
    print(f"  {C.WHITE}Idea:{C.RESET}")
    print(f"  {C.DIM}  Sharp = Original + α × (Original − Blurred)")
    print(f"  {C.DIM}  = Identity filter + edge-enhancement term{C.RESET}")
    print()

    K = np.array([[ 0, -1,  0],
                  [-1,  5, -1],
                  [ 0, -1,  0]])
    print_matrix(K.astype(float), "Kernel_Sharp1", C.CYAN)
    print()
    bullet("Centre: +5  |  Neighbours: −1  |  Sum = 5 − 4 = 1")
    bullet("The +5 centre amplifies the pixel; the −1 neighbours subtract surroundings.")
    bullet("Net effect: edges (regions of high contrast) are enhanced.")
    bullet("Applied to GRAYSCALE image.")

    gray = np.mean(ImJPG, axis=2).astype(np.uint8)
    result = apply_gray(gray, K)
    wait()
    show_image(result, "Step 8 — Sharpen: Kernel_Sharp1", cmap='gray')
    print(f"  {C.GREEN}▸ Sharp1 image displayed.{C.RESET}")
    wait()
    return result

def step_sharp2(ImJPG):
    section("STEP 9 — Sharpening: Kernel_Sharp2")
    bullet("A stronger sharpening filter — uses ALL 8 neighbours instead of 4.")
    print()

    K = np.array([[-1, -1, -1],
                  [-1,  9, -1],
                  [-1, -1, -1]])
    print_matrix(K.astype(float), "Kernel_Sharp2", C.CYAN)
    print()
    bullet("Centre: +9  |  All 8 neighbours: −1  |  Sum = 9 − 8 = 1")
    bullet("Stronger sharpening than Sharp1 — diagonals also contribute.")
    print()
    print(f"  {C.DIM}  Compare:")
    print(f"  {C.DIM}  Sharp1: 4 neighbours (cross shape) → moderate sharpening")
    print(f"  {C.DIM}  Sharp2: 8 neighbours (full 3×3)   → stronger sharpening{C.RESET}")

    gray = np.mean(ImJPG, axis=2).astype(np.uint8)
    result = apply_gray(gray, K)
    wait()
    show_image(result, "Step 9 — Sharpen: Kernel_Sharp2", cmap='gray')
    print(f"  {C.GREEN}▸ Sharp2 image displayed.{C.RESET}")
    wait()
    return result

def step_sobel1(ImJPG):
    section("STEP 10 — Edge Detection: Sobel Horizontal")
    bullet("Sobel filters compute the discrete DERIVATIVE of the image.")
    bullet("Large derivative = rapid change in brightness = EDGE.")
    print()

    K = np.array([[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]])
    print_matrix(K.astype(float), "Kernel_Sobel1  (horizontal gradient)", C.CYAN)
    print()
    bullet("Left column: −1, −2, −1  |  Right column: +1, +2, +1  |  Centre: 0")
    bullet("Computes difference between LEFT and RIGHT sides of each pixel.")
    bullet("Strong response at VERTICAL edges (where left/right values differ).")
    bullet("Column of zeros → no response in flat horizontal regions.")

    gray = np.mean(ImJPG, axis=2).astype(np.uint8)
    result = apply_gray(gray, K)
    wait()
    show_image(result, "Step 10 — Sobel Horizontal Edges", cmap='gray')
    print(f"  {C.GREEN}▸ Sobel horizontal edges displayed.{C.RESET}")
    wait()
    return result

def step_sobel2(ImJPG):
    section("STEP 11 — Edge Detection: Sobel Vertical")

    K = np.array([[-1, -2, -1],
                  [ 0,  0,  0],
                  [ 1,  2,  1]])
    print_matrix(K.astype(float), "Kernel_Sobel2  (vertical gradient)", C.CYAN)
    print()
    bullet("Top row: −1, −2, −1  |  Bottom row: +1, +2, +1  |  Centre: 0")
    bullet("Computes difference between TOP and BOTTOM sides of each pixel.")
    bullet("Strong response at HORIZONTAL edges.")
    print()
    print(f"  {C.DIM}  Note: Sobel2 = transpose of Sobel1{C.RESET}")
    print(f"  {C.CYAN}  Kernel_Sobel2 = Kernel_Sobel1ᵀ{C.RESET}")

    gray = np.mean(ImJPG, axis=2).astype(np.uint8)
    result = apply_gray(gray, K)
    wait()
    show_image(result, "Step 11 — Sobel Vertical Edges", cmap='gray')
    print(f"  {C.GREEN}▸ Sobel vertical edges displayed.{C.RESET}")
    wait()
    return result

def step_sobel_combined(ImJPG):
    section("STEP 12 — Edge Detection: Sobel Combined")
    bullet("We add both Sobel responses to get ALL edges (horizontal + vertical).")
    print()
    print(f"  {C.WHITE}Formula:{C.RESET}")
    print(f"  {C.CYAN}  ImJPG_SobelCombined  =  ImJPG_Sobel1  +  ImJPG_Sobel2{C.RESET}")
    print()

    K1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    K2 = np.array([[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]])
    print_matrix(K1.astype(float), "Kernel_Sobel1  (horizontal gradient)", C.BLUE)
    print_matrix(K2.astype(float), "Kernel_Sobel2  (vertical gradient)", C.MAGENTA)
    print()
    bullet("Bright pixels in the result = edges detected in EITHER direction.")

    gray = np.mean(ImJPG, axis=2).astype(np.uint8)
    s1 = convolve2d(gray, K1, mode='same', boundary='symm')
    s2 = convolve2d(gray, K2, mode='same', boundary='symm')
    result = np.uint8(np.clip(s1 + s2, 0, 255))
    wait()
    show_image(result, "Step 12 — Sobel Combined Edges", cmap='gray')
    print(f"  {C.GREEN}▸ Combined edge map displayed.{C.RESET}")
    wait()
    return result

def step_laplace(ImJPG):
    section("STEP 13 — Edge Detection: Laplacian")
    bullet("The Laplacian computes the SECOND derivative — it finds regions")
    bullet("of rapid curvature in ALL directions at once (isotropic).")
    print()

    K = np.array([[ 0, -1,  0],
                  [-1,  4, -1],
                  [ 0, -1,  0]])
    print_matrix(K.astype(float), "Kernel_Laplace  (second derivative)", C.CYAN)
    print()
    bullet("Centre: +4  |  Neighbours: −1  |  Sum = 0")
    bullet("Sum = 0 means: flat regions give zero response (only edges lit up).")
    bullet("Unlike Sobel, it is ISOTROPIC — responds equally in all directions.")
    print()
    print(f"  {C.DIM}  Compare with Sobel:")
    print(f"  {C.DIM}  Sobel:     1st derivative  →  two separate directional passes")
    print(f"  {C.DIM}  Laplacian: 2nd derivative  →  single isotropic pass{C.RESET}")

    result = apply_rgb(ImJPG.astype(np.float64), K)
    wait()
    show_image(result, "Step 13 — Laplacian Edge Detection")
    print(f"  {C.GREEN}▸ Laplacian edge map displayed.{C.RESET}")
    wait()
    return result

def step_qa():
    section("STEP 14 — Q&A Answers Summary")
    print()

    qa = [
        ("Q1", "Size of ImJPG_Average1 and ImJPG_Average2?",
         "Same as original ImJPG. Using mode='same' in convolve2d\n"
         "     preserves the spatial dimensions of the image."),
        ("Q2", "Size of original ImJPG?",
         "Use ImJPG.shape → returns (m, n, 3)  e.g. (512, 512, 3)"),
        ("Q3", "Which filter blurs more — Average1 or Average2?",
         "Average2 (box filter) blurs MORE.\n"
         "     Average1 uses 5 pixels (cross), Average2 uses 9 (full 3×3).\n"
         "     More neighbours averaged = more high-frequency detail lost."),
        ("Q4", "Single kernel equivalent to applying Gaussian twice?",
         "Convolve the kernel with itself:\n"
         "     Kernel_Gauss_Twice = convolve2d(K_Gauss, K_Gauss, mode='full')"),
        ("Q5", "Size of the double-Gaussian equivalent kernel?",
         "5×5   because  (3+3−1) × (3+3−1) = 5×5\n"
         "     Full convolution of two 3×3 matrices always gives (2N−1)×(2N−1)."),
        ("Q6", "Kernel_Large vs double Gaussian — which blurs more?",
         "Kernel_Large blurs MORE.\n"
         "     It operates over a wider 5×5 spatial window in one pass.\n"
         "     Double Gaussian applies two narrow 3×3 passes sequentially."),
    ]

    for q, question, answer in qa:
        hr("─", 62, C.DIM)
        print(f"  {C.YELLOW}{C.BOLD}{q}:{C.RESET}  {C.WHITE}{question}{C.RESET}")
        print(f"     {C.GREEN}→ {answer}{C.RESET}")
    hr("─", 62, C.DIM)
    wait()

# ═══════════════════════════════════════════════════════════════════════
# STEPS 15–18  (Linear Algebra Extension)
# ═══════════════════════════════════════════════════════════════════════

def step_linear_transform(ImJPG):
    section("STEP 15 — Linear Transformation: Linearity of Convolution")
    bullet("Convolution is a LINEAR operation. This means it satisfies:")
    print()
    print(f"  {C.WHITE}Superposition Principle:{C.RESET}")
    print(f"  {C.CYAN}  T(A + B)  =  T(A) + T(B)           (Additivity)")
    print(f"  {C.CYAN}  T(α · A)  =  α · T(A)              (Homogeneity / Scaling){C.RESET}")
    print()
    bullet("We verify additivity using two small 2×2 matrices and a kernel.")
    print()

    A = np.array([[1, 2],
                  [3, 4]], dtype=float)
    B = np.array([[5, 6],
                  [7, 8]], dtype=float)
    K = np.array([[1,  0],
                  [0, -1]], dtype=float)

    TA  = convolve2d(A,     K, mode='same')
    TB  = convolve2d(B,     K, mode='same')
    TAB = convolve2d(A + B, K, mode='same')
    TA_plus_TB = TA + TB

    print_matrix(A,   "Matrix A", C.BLUE)
    print_matrix(B,   "Matrix B", C.MAGENTA)
    print_matrix(K,   "Kernel K  (flip vertical)", C.CYAN)
    print()
    print_matrix(TA,           "T(A)  =  convolve(A, K)", C.GREEN)
    print_matrix(TB,           "T(B)  =  convolve(B, K)", C.GREEN)
    print_matrix(TAB,          "T(A+B)  =  convolve(A+B, K)", C.YELLOW)
    print_matrix(TA_plus_TB,   "T(A) + T(B)", C.YELLOW)
    print()

    if np.allclose(TAB, TA_plus_TB):
        print(f"  {C.GREEN}{C.BOLD}  ✔  Verified: T(A+B) = T(A) + T(B)  — Additivity holds!{C.RESET}")
    else:
        print(f"  {C.RED}  ✘  Mismatch detected (numerical issue?){C.RESET}")
    print()
    bullet("This property is fundamental in signal processing and image analysis.")
    bullet("It allows us to decompose complex inputs and process them separately.")
    wait()
    # 🔹 APPLY ON ACTUAL IMAGE (ADD THIS)

    bullet("Now applying linearity on the actual image.")

    A_img = ImJPG * 0.8
    B_img = ImJPG * 0.2

    K_img = np.array([[0,1,0],
                  [1,4,1],
                  [0,1,0]]) / 8

    TA_img = apply_rgb(A_img, K_img)
    TB_img = apply_rgb(B_img, K_img)
    TAB_img = apply_rgb(A_img + B_img, K_img)

    combined = np.uint8(np.clip(TA_img.astype(float) + TB_img.astype(float), 0, 255))

    wait()

    show_image(TAB_img, "T(A+B) — Image")
    wait()

    show_image(combined, "T(A) + T(B) — Image")
    wait()

def step_eigen(ImJPG):
    section("STEP 16 — Eigenvalues: Spectral Analysis of Kernel")
    bullet("Eigenvalues reveal how a matrix (kernel) stretches or compresses space.")
    bullet("For a square kernel K, if  K·v = λ·v  then λ is an eigenvalue.")
    print()
    print(f"  {C.WHITE}Why do eigenvalues matter for image kernels?{C.RESET}")
    print(f"  {C.DIM}  • Large |λ|  → strong amplification of that frequency component")
    print(f"  {C.DIM}  • |λ| < 1    → attenuation (blurring suppresses high frequencies)")
    print(f"  {C.DIM}  • λ = 0      → that direction is completely zeroed out{C.RESET}")
    print()

    K = np.ones((5,5)) / 25

    print_matrix(K, "Kernel_Gauss  (used for eigenvalue analysis)", C.CYAN)
    print()

    vals, vecs = np.linalg.eig(K)

    # Sort by descending magnitude for clarity
    idx = np.argsort(np.abs(vals))[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    print_vector(vals.real, "Eigenvalues  λ₁, λ₂, λ₃  (real parts)", C.YELLOW)
    print()
    bullet("Interpretation of these eigenvalues:")
    for i, lam in enumerate(vals.real):
        tag = "dominant" if i == 0 else ("mid" if i == 1 else "smallest")
        print(f"  {C.DIM}  λ{i+1} = {lam:.6f}   ({tag}){C.RESET}")
    print()
    bullet("All eigenvalues < 1 → confirms blurring: high-frequency components are attenuated.")
    bullet("Largest eigenvalue ≈ sum of kernel = 1.0 (DC component / brightness preserved).")
    wait()
     # 🔹 APPLY KERNEL ON IMAGE (ADD THIS)

    bullet("Now applying this kernel on the image to observe effect.")

    gray = np.mean(ImJPG, axis=2)

    result = convolve2d(gray, K, mode='same', boundary='symm')
    result = np.uint8(np.clip(result, 0, 255))

    wait()

    show_image(gray.astype(np.uint8), "Original Image (Grayscale)", cmap='gray')
    wait()

    show_image(result, "After Applying Kernel (Eigen Effect)", cmap='gray')
    wait()

def step_svd(ImJPG):
    section("STEP 17 — SVD Compression: Low-Rank Approximation")
    bullet("SVD (Singular Value Decomposition) decomposes a matrix M as:")
    print()
    print(f"  {C.CYAN}  M  =  U · Σ · Vᵀ{C.RESET}")
    print(f"  {C.DIM}  U: left singular vectors  (m×m orthogonal)")
    print(f"  {C.DIM}  Σ: diagonal matrix of singular values (descending)")
    print(f"  {C.DIM}  Vᵀ: right singular vectors transposed (n×n orthogonal){C.RESET}")
    print()
    bullet("By keeping only the top k singular values, we get a compressed approximation.")
    print()
    print(f"  {C.WHITE}Formula for rank-k approximation:{C.RESET}")
    print(f"  {C.CYAN}  M_k  =  U[:, :k] · diag(Σ[:k]) · Vᵀ[:k, :]{C.RESET}")
    print()

    gray = np.mean(ImJPG, axis=2)
    m, n = gray.shape
    U, S, Vt = np.linalg.svd(gray, full_matrices=True)

    info("Image dimensions:", f"{m} × {n}   ({m*n:,} values)")
    info("Singular values:", f"{len(S)} total  (descending magnitude)")
    print()

    # Show singular value energy distribution
    energy_10  = np.sum(S[:10]**2)  / np.sum(S**2) * 100
    energy_50  = np.sum(S[:50]**2)  / np.sum(S**2) * 100
    energy_100 = np.sum(S[:100]**2) / np.sum(S**2) * 100

    bullet(f"Top  10 singular values capture {energy_10:.1f}% of image energy")
    bullet(f"Top  50 singular values capture {energy_50:.1f}% of image energy")
    bullet(f"Top 100 singular values capture {energy_100:.1f}% of image energy")
    print()

    k = 10
    compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    compressed_img = np.uint8(np.clip(compressed, 0, 255))

    ratio = (k * (m + n + 1)) / (m * n) * 100
    print(f"  {C.YELLOW}  Compression ratio at k={k}: stores {ratio:.1f}% of original data{C.RESET}")
    print()

    wait()

    # Show both side by side
    plt.close('all')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#111111')
    for ax in axes:
        ax.set_facecolor('#111111')
        ax.axis('off')
    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title("Original Grayscale", color='white', fontsize=12)
    axes[1].imshow(compressed_img, cmap='gray')
    axes[1].set_title(f"SVD Compressed  (k={k}, ~{ratio:.0f}% data)", color='white', fontsize=12)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    print(f"  {C.GREEN}▸ Original vs SVD compressed image displayed.{C.RESET}")
    wait()

def step_inner():
    section("STEP 18 — Inner Product: Orthogonality Check")
    bullet("The inner product (dot product) of two vectors measures their 'similarity'.")
    bullet("If the dot product = 0, the vectors are ORTHOGONAL (perpendicular).")
    print()
    print(f"  {C.WHITE}Formula:{C.RESET}")
    print(f"  {C.CYAN}  ⟨v₁, v₂⟩  =  v₁ · v₂  =  Σᵢ  v₁ᵢ · v₂ᵢ{C.RESET}")
    print()
    bullet("In image processing, orthogonality means two basis signals don't interfere.")
    bullet("SVD singular vectors are orthogonal — that's why SVD compression works cleanly.")
    print()

    v1 = np.array([1.0,  2.0, 3.0])
    v2 = np.array([3.0, -2.0, 1.0])

    print_vector(v1, "v₁  =  [1, 2, 3]", C.BLUE)
    print_vector(v2, "v₂  =  [3, −2, 1]", C.MAGENTA)
    print()

    dot = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cosine  = dot / (norm_v1 * norm_v2)

    info("Dot product:",      f"v₁ · v₂  =  {dot:.4f}")
    info("‖v₁‖ (norm):",     f"{norm_v1:.4f}")
    info("‖v₂‖ (norm):",     f"{norm_v2:.4f}")
    info("cos θ:",            f"{cosine:.4f}")
    print()

    if np.isclose(dot, 0):
        print(f"  {C.GREEN}{C.BOLD}  ✔  dot = 0  →  v₁ and v₂ are ORTHOGONAL{C.RESET}")
    else:
        print(f"  {C.YELLOW}  →  dot = {dot:.4f}  (not zero → NOT orthogonal){C.RESET}")
        angle_deg = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        print(f"  {C.DIM}     Angle between vectors: {angle_deg:.2f}°{C.RESET}")
    print()

    # Bonus: show an orthogonal pair
    hr("─", 62, C.DIM)
    print(f"\n  {C.WHITE}Bonus — Orthogonal example:{C.RESET}")
    u1 = np.array([1.0, 0.0, 0.0])
    u2 = np.array([0.0, 1.0, 0.0])
    print_vector(u1, "u₁  =  [1, 0, 0]  (x-axis)", C.GREEN)
    print_vector(u2, "u₂  =  [0, 1, 0]  (y-axis)", C.GREEN)
    print(f"\n  {C.GREEN}{C.BOLD}  ✔  u₁ · u₂  =  {np.dot(u1, u2):.1f}  →  perfectly orthogonal{C.RESET}")
    print()
    bullet("Orthogonal bases are the foundation of Fourier transforms, wavelets, and SVD.")
    wait()

# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def main():
    print()
    hr("═", 62, C.MAGENTA)
    print(f"{C.BOLD}{C.MAGENTA}  IMAGE CONVOLUTION — LINEAR ALGEBRA MINI PROJECT{C.RESET}")
    print(f"  {C.DIM}UE24MA241B  |  PES University{C.RESET}")
    hr("═", 62, C.MAGENTA)
    print(f"\n  {C.WHITE}This program guides you step by step through the pipeline.")
    print(f"  Each step prints the kernel matrix, explains it,")
    print(f"  then opens an image window showing the result.{C.RESET}\n")
    wait("  Press ENTER to open the main menu...")

    ImJPG    = None
    noisy    = None
    gauss    = None
    computed = set()

    STEP_FNS = {
        1:  lambda: step_load(),
        2:  lambda: step_noise(ImJPG),
        3:  lambda: step_average1(noisy),
        4:  lambda: step_average2(noisy),
        5:  lambda: step_gauss(noisy),
        6:  lambda: step_gauss2(gauss),
        7:  lambda: step_large(ImJPG),
        8:  lambda: step_sharp1(ImJPG),
        9:  lambda: step_sharp2(ImJPG),
        10: lambda: step_sobel1(ImJPG),
        11: lambda: step_sobel2(ImJPG),
        12: lambda: step_sobel_combined(ImJPG),
        13: lambda: step_laplace(ImJPG),
        14: lambda: step_qa(),
        15: lambda: step_linear_transform(ImJPG),
        16: lambda: step_eigen(ImJPG),
        17: lambda: step_svd(ImJPG)
        
    }

    while True:
        print_menu(computed)
        choice = input(f"  {C.CYAN}Enter step number (0 to exit): {C.RESET}").strip()

        try:
            c = int(choice)
        except ValueError:
            print(f"  {C.RED}Invalid input — please enter a number.{C.RESET}")
            wait()
            continue

        if c == 0:
            print(f"\n  {C.GREEN}Goodbye!{C.RESET}\n")
            plt.close('all')
            break

        if c not in STEP_FNS:
            print(f"  {C.RED}Step {c} not found. Choose from the menu.{C.RESET}")
            wait()
            continue

        # Guard: steps 2–18 need step 1 (image)
        if c >= 2 and ImJPG is None:
            print(f"  {C.RED}Please run Step 1 first to load the image.{C.RESET}")
            wait()
            continue

        # Guard: steps 3–6 need step 2 (noisy image)
        if c in (3, 4, 5, 6) and noisy is None:
            print(f"  {C.RED}Please run Step 2 first to add noise.{C.RESET}")
            wait()
            continue

        # Guard: step 6 needs step 5 (gaussian output)
        if c == 6 and gauss is None:
            print(f"  {C.RED}Please run Step 5 first (Gaussian Blur).{C.RESET}")
            wait()
            continue

        # Run the step, capturing outputs where needed
        if c == 1:
            ImJPG = STEP_FNS[1]()
        elif c == 2:
            noisy = STEP_FNS[2]()
        elif c == 5:
            gauss = STEP_FNS[5]()
        else:
            STEP_FNS[c]()

        computed.add(c)

if __name__ == "__main__":
    main()