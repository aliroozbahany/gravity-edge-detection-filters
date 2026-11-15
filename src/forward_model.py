from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# ۱. تعریف پارامترها (کاملاً در سیستم SI: متر و کیلوگرم)
# ==========================================
x0 = 0.0          # مرکز افقی مستطیل (متر)
width = 20000.0   # عرض مستطیل (متر)
z1 = 2000.0       # عمق سقف مستطیل (متر)
z2 = 6000.0       # عمق کف مستطیل (متر)
density = 500.0   # اختلاف چگالی با سنگ اطراف (kg/m^3)

# ایستگاه‌های اندازه‌گیری روی پروفیل (از منفی ۵۰ کیلومتر تا مثبت ۵۰ کیلومتر با گام ۱۰۰۰ متر)
x = np.arange(-50000.0, 51000.0, 1000.0)

# ثابت جهانی گرانش
G = 6.67430e-11

# ==========================================
# ۲. فوروارد مدلینگ گرانی مستطیل دوبعدی
# ==========================================
left = x0 - width / 2.0
right = x0 + width / 2.0

def primitive(u):
    # پادمشتق تحلیلی اثر گرانی منشور دوبعدی
    return (
        u * np.log((u**2 + z2**2) / (u**2 + z1**2))
        + 2.0 * z2 * np.arctan(u / z2)
        - 2.0 * z1 * np.arctan(u / z1)
    )

# محاسبه گرانی خالص بدون نویز و تبدیل از m/s^2 به mGal (ضریب 1e5)
g_clean = G * density * (primitive(right - x) - primitive(left - x)) * 1e5

# ==========================================
# ۳. تولید ۳ سطح نویز (۰، ۵ و ۱۰ درصد)
# ==========================================
# درصد نویز به نسبت انحراف معیار سیگنال خالص تعریف می‌شود
rng = np.random.default_rng(42)
noise_random = rng.normal(size=len(x))
signal_std = np.std(g_clean)

g_0 = g_clean
g_5 = g_clean + 0.05 * signal_std * noise_random
g_10 = g_clean + 0.10 * signal_std * noise_random

# ==========================================
# ۴. مسیردهی و ساخت پوشه‌های data و figures
# ==========================================
root = Path(__file__).resolve().parent.parent
data_dir = root / "data"
figures_dir = root / "figures"
data_dir.mkdir(exist_ok=True)
figures_dir.mkdir(exist_ok=True)

# ذخیره خروجی در فایل CSV
csv_path = data_dir / "synthetic_rectangle_profile.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["x_m", "g_0_mGal", "g_5_mGal", "g_10_mGal"])
    writer.writerows(zip(x, g_0, g_5, g_10))

# ==========================================
# ۵. رسم و ذخیره نمودار آنومالی
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(x, g_0, color="black", linewidth=2, label="Noise 0%")
plt.plot(x, g_5, linewidth=1, alpha=0.8, label="Noise 5%")
plt.plot(x, g_10, linewidth=1, alpha=0.8, label="Noise 10%")

# خط‌چین‌های مشخص‌کننده لبه‌های واقعی مستطیل
plt.axvline(left, color="gray", linestyle="--", label="True Edges")
plt.axvline(right, color="gray", linestyle="--")

plt.xlabel("Position x (m)")
plt.ylabel("Gravity Anomaly (mGal)")
plt.title("Buried 2D Rectangle Gravity Anomaly (Synthetic Profile)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

figure_path = figures_dir / "gravity_rectangle_profile.png"
plt.savefig(figure_path, dpi=300)
plt.show()

print(f"فایل CSV در این مسیر ذخیره شد: {csv_path}")
print(f"تصویر نمودار در این مسیر ذخیره شد: {figure_path}")
