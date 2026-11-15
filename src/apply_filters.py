from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# ==========================================
# ۱. بارگذاری داده‌های مصنوعی از فایل CSV
# ==========================================
root = Path(__file__).resolve().parent.parent
data_path = root / "data" / "synthetic_rectangle_profile.csv"
df = pd.read_csv(data_path)

x = df['x_m'].values
g_0 = df['g_0_mGal'].values
g_5 = df['g_5_mGal'].values
g_10 = df['g_10_mGal'].values

# ==========================================
# ۲. تابع محاسبه ۷ فیلتر برای یک پروفیل گرانی
# ==========================================
def compute_all_filters(g, x_coords):
    # مشتقات افقی و قائم
    Fx = np.gradient(g, x_coords)
    Fz = np.imag(hilbert(Fx))
    
    # فیلترهای پایه و زاویه‌ای
    THG = np.abs(Fx)
    AS = np.sqrt(Fx**2 + Fz**2)
    TA = np.arctan2(Fz, THG + 1e-10)
    TM = np.arccos(np.clip(THG / (AS + 1e-10), -1.0, 1.0))
    TDX = np.arctan2(THG, np.abs(Fz) + 1e-10)
    
    # مشتقات ثانویه روی THG برای TTHG و LTHG
    THG_x = np.gradient(THG, x_coords)
    THG_z = np.imag(hilbert(THG_x))
    
    TTHG = np.arctan2(THG_z, np.abs(THG_x) + 1e-10)
    LTHG = (1.0 + np.exp(-1.0 * (THG_z / (np.abs(THG_x) + 1e-10))))**(-2.0)  # K = 2
    
    return {
        'THG': THG,
        'AS': AS,
        'TA': TA,
        'TM': TM,
        'TDX': TDX,
        'TTHG': TTHG,
        'LTHG': LTHG
    }

# محاسبه فیلترها برای هر سه سطح نویز
res_0 = compute_all_filters(g_0, x)
res_5 = compute_all_filters(g_5, x)
res_10 = compute_all_filters(g_10, x)

# ==========================================
# ۳. رسم نمودار مقایسه‌ای همه‌جانبه (4 در 2)
# ==========================================
fig, axes = plt.subplots(4, 2, figsize=(15, 16), sharex=True)
axes = axes.flatten()

# پنل ۱: نمایش داده‌های خام ورودی
axes[0].plot(x, g_0, color='black', linewidth=1.8, label='Noise 0%')
axes[0].plot(x, g_5, color='royalblue', linewidth=1.2, alpha=0.85, label='Noise 5%')
axes[0].plot(x, g_10, color='crimson', linewidth=1.0, alpha=0.75, label='Noise 10%')
axes[0].set_title("Input Gravity Anomaly (mGal)", fontsize=11, fontweight='bold')
axes[0].set_ylabel("Anomaly (mGal)")
axes[0].grid(alpha=0.3)
axes[0].legend(loc='upper right')
axes[0].axvline(-10000, color='gray', linestyle='--', linewidth=1.2)
axes[0].axvline(10000, color='gray', linestyle='--', linewidth=1.2)

# پنل‌های ۲ تا ۸: نمایش ۷ فیلتر لبه‌یاب
filter_keys = ['THG', 'AS', 'TA', 'TM', 'TDX', 'TTHG', 'LTHG']

for i, key in enumerate(filter_keys):
    ax = axes[i + 1]
    
    ax.plot(x, res_0[key], color='black', linewidth=1.8, label='0%')
    ax.plot(x, res_5[key], color='royalblue', linewidth=1.2, alpha=0.85, label='5%')
    ax.plot(x, res_10[key], color='crimson', linewidth=1.0, alpha=0.75, label='10%')
    
    ax.set_title(f"Filter: {key}", fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    
    # لبه‌های واقعی در -10000 و +10000 متر
    ax.axvline(-10000, color='gray', linestyle='--', linewidth=1.2)
    ax.axvline(10000, color='gray', linestyle='--', linewidth=1.2)

# برچسب محور افقی برای دو نمودار ردیف آخر
axes[6].set_xlabel("Position x (m)", fontsize=10)
axes[7].set_xlabel("Position x (m)", fontsize=10)

plt.tight_layout()

# ذخیره تصویر با کیفیت بالا
figures_dir = root / "figures"
figures_dir.mkdir(exist_ok=True)
output_fig = figures_dir / "filters_noise_comparison.png"
fig.savefig(output_fig, dpi=300)
plt.show()

print(f"نمودار مقایسه فیلترها در مسیر زیر ذخیره شد:\n{output_fig}")
