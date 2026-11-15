import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# تنظیم مسیر ریشه پروژه (E:\Doctorate\Projects\gravity-edge-detection-filters)
# ==========================================
# چون اسکریپت داخل src قرار دارد، parent.parent به پوشه اصلی پروژه اشاره می‌کند
root = Path(__file__).resolve().parent.parent

data_dir = root / "data"
figures_dir = root / "figures"

data_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

# ==========================================
# شبکه محاسباتی (متر)
# ==========================================
x = np.linspace(-50000, 50000, 101)
y = np.linspace(-50000, 50000, 101)
X, Y = np.meshgrid(x, y)

# ==========================================
# مشخصات منشور مستطیلی (متر و SI)
# ==========================================
x1, x2 = -10000.0, 10000.0
y1, y2 = -10000.0, 10000.0
z1, z2 = 2000.0, 6000.0   # عمق مثبت به سمت پایین
rho = 500.0               # کنتراست چگالی (kg/m^3)
G = 6.67430e-11

# ==========================================
# فرمول تحلیلی و دقیق گرانش منشور (Nagy et al., 2000)
# ==========================================
def gravity_prism(xp, yp, x1, x2, y1, y2, z1, z2, rho):
    xs = [x1, x2]
    ys = [y1, y2]
    zs = [z1, z2]
    
    g_total = np.zeros_like(xp, dtype=np.float64)
    eps = 1e-12
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # علامت متناظر با کران بالا (۱) یا پایین (۰) انتگرال
                sign = (-1.0) ** (i + j + k)
                
                dx = xs[i] - xp
                dy = ys[j] - yp
                dz = zs[k]  # عمق مشاهده‌گر روی سطح زمین z=0 است
                
                r = np.sqrt(dx**2 + dy**2 + dz**2)
                
                # کرنل تحلیلی مولفه قائم گرانی
                term1 = dx * np.log(np.maximum(dy + r, eps))
                term2 = dy * np.log(np.maximum(dx + r, eps))
                term3 = dz * np.arctan2(dx * dy, np.maximum(dz * r, eps))
                
                g_total += sign * (term1 + term2 - term3)
                
    # تبدیل شتاب به میلی‌گال (1 m/s^2 = 1e5 mGal)
    return g_total * G * rho * 1e5

print("در حال محاسبه میدان گرانی مصنوعی منشور...")
g_base = gravity_prism(X, Y, x1, x2, y1, y2, z1, z2, rho)

# ==========================================
# افزودن نویز گاوسی
# ==========================================
np.random.seed(42)
std_signal = np.std(g_base)
g_0 = g_base
g_5 = g_base + np.random.normal(0, 0.05 * std_signal, g_base.shape)
g_10 = g_base + np.random.normal(0, 0.10 * std_signal, g_base.shape)

# ==========================================
# ذخیره در فایل CSV
# ==========================================
csv_path = data_dir / "synthetic_prism_grid.csv"
df = pd.DataFrame({
    'x': X.flatten(),
    'y': Y.flatten(),
    'g_0': g_0.flatten(),
    'g_5': g_5.flatten(),
    'g_10': g_10.flatten()
})
df.to_csv(csv_path, index=False)
print(f"✓ داده‌ها با موفقیت ذخیره شدند در:\n  {csv_path.resolve()}")

# ==========================================
# رسم و ذخیره نقشه گرانی
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
datasets = [g_0, g_5, g_10]
titles = ['Noise 0%', 'Noise 5%', 'Noise 10%']

for i, ax in enumerate(axes):
    c = ax.contourf(X, Y, datasets[i], levels=25, cmap='viridis')
    ax.set_title(titles[i], fontsize=13)
    ax.set_xlabel("x (m)")
    if i == 0:
        ax.set_ylabel("y (m)")
    fig.colorbar(c, ax=ax, shrink=0.8, label="mGal")

plt.tight_layout()
fig_path = figures_dir / "gravity_grid_anomaly.png"
plt.savefig(fig_path, dpi=300)
plt.close(fig)
print(f"✓ نقشه گرانی ذخیره شد در:\n  {fig_path.resolve()}")
