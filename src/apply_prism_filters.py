import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter

# ۱. مدیریت مسیرها
current_file = Path(__file__).resolve()
root = current_file.parent.parent if current_file.parent.name == "src" else current_file.parent

data_path = root / "data" / "synthetic_prism_grid.csv"
if not data_path.exists():
    data_path = root / "synthetic_prism_grid.csv"

output_dir = root / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

# ۲. بارگذاری داده
df = pd.read_csv(data_path)
N = 101
dx = 1000.0  # متر
dy = 1000.0

def apply_filters(g_flat, noise_label, filename_suffix, sigma=0.0):
    g = g_flat.values.reshape((N, N))
    
    # فیلتر ملایم گاوسی برای مهار نویز عددی در داده نویزدار
    if sigma > 0:
        g = gaussian_filter(g, sigma=sigma)
    
    # مشتقات افقی
    gy, gx = np.gradient(g, dy, dx)
    thg = np.sqrt(gx**2 + gy**2)
    
    # مشتق دوم افقی (لاپلاسین افقی)
    gyy, _ = np.gradient(gy, dy, dx)
    _, gxx = np.gradient(gx, dy, dx)
    lap_h = -(gxx + gyy)
    
    # مشتق قائم متوازن و هم‌مقیاس با THG (حل مشکل اشباع arctan بدون نیاز به ماسک)
    scale_factor = np.std(thg) / (np.std(lap_h) * dx + 1e-12)
    gz = lap_h * dx * scale_factor
    
    # دامنه سیگنال تحلیلی (AS)
    as_val = np.sqrt(thg**2 + gz**2)
    
    # یک اپسیلون متناسب با میانه میدان برای پایداری در پس‌زمینه
    eps_val = 0.05 * np.mean(thg)
    
    # ۱) زاویه شیب (TA) - کاملاً پیوسته
    ta = np.arctan(gz / (thg + eps_val))
    
    # ۲) فیلتر تتا (TM)
    tm = np.arccos(np.clip(thg / (as_val + 1e-12), -1.0, 1.0))
    
    # ۳) فیلتر TDX
    tdx = np.arctan(thg / (np.abs(gz) + eps_val))
    
    # مشتقات مرتبه بالاتر برای TTHG و LTHG
    thg_y, thg_x = np.gradient(thg, dy, dx)
    h_grad_thg = np.sqrt(thg_x**2 + thg_y**2)
    
    thg_yy, _ = np.gradient(thg_y, dy, dx)
    _, thg_xx = np.gradient(thg_x, dy, dx)
    lap_thg = -(thg_xx + thg_yy)
    
    scale_thg = np.std(h_grad_thg) / (np.std(lap_thg) * dx + 1e-12)
    thg_z = lap_thg * dx * scale_thg
    
    eps_thg = 0.05 * np.mean(h_grad_thg)
    
    # ۴) فیلتر TTHG
    tthg = np.arctan(thg_z / (h_grad_thg + eps_thg))
    
    # ۵) فیلتر LTHG (تابع لجستیک ملایم)
    ratio = thg_z / (h_grad_thg + eps_thg)
    lthg = 1.0 / (1.0 + np.exp(-1.5 * np.clip(ratio, -5, 5)))

    # ۳. رسم نمودارها
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    filter_list = [
        ("THG", thg, "viridis"),
        ("AS", as_val, "viridis"),
        ("TA (rad)", ta, "coolwarm"),
        ("TM (rad)", tm, "plasma"),
        ("TDX (rad)", tdx, "magma"),
        ("TTHG (rad)", tthg, "coolwarm"),
        ("LTHG", lthg, "inferno")
    ]
    
    extent = [-50, 50, -50, 50]
    
    for i, (name, data, cmap) in enumerate(filter_list):
        ax = axes.flatten()[i]
        im = ax.imshow(data, extent=extent, origin='lower', cmap=cmap)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        plt.colorbar(im, ax=ax, shrink=0.75)
    
    axes.flatten()[7].axis('off')
    plt.suptitle(f"Edge Detection Filters - {noise_label}", fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / f"filters_{filename_suffix}.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"تصویر پیوسته و طبیعی ذخیره شد: {save_path}")

# ۴. اجرا
apply_filters(df['g_0'], "Noise-Free (0%)", "g_0", sigma=0.0)
apply_filters(df['g_5'], "5% Gaussian Noise", "g_5", sigma=1.2)
apply_filters(df['g_10'], "10% Gaussian Noise", "g_10", sigma=1.8)

print("\nتمام نقشه‌ها به صورت کاملاً پیوسته (بدون برش دایره‌ای) تولید شدند.")
