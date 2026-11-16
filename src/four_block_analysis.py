from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ۱) تنظیمات اولیه و ثابت‌ها
G = 6.67430e-11 

# تعریف منشورها (استفاده از .0 برای اطمینان از Float بودن)
blocks = np.array(
    [
        [220.0, 500.0, 230.0, 270.0, 220.0, 280.0, 800.0],
        [170.0, 320.0, 390.0, 530.0, 150.0, 330.0, 500.0],
        [200.0, 320.0, 650.0, 780.0, 80.0, 200.0, 300.0],
        [700.0, 830.0, 460.0, 600.0, 120.0, 290.0, 650.0],
    ], dtype=float
)

x = np.arange(0, 1001, 25)
y = np.arange(0, 1001, 25)
X, Y = np.meshgrid(x, y)

h = 1.0 

# ۲) فرمول تحلیلی
def gravity_prism(Xo, Yo, b, G_const):
    x1, x2, y1, y2, d1, d2, rho = b
    def F(u, v, d):
        r = np.sqrt(u**2 + v**2 + d**2)
        term1 = u * np.log(v + r + 1e-14)
        term2 = v * np.log(u + r + 1e-14)
        term3 = -d * np.arctan2(u * v, d * r + 1e-14)
        return -(term1 + term2 + term3)

    gz_sum = np.zeros_like(Xo, dtype=float)
    for ix, xc in enumerate([x1, x2]):
        sx = -1 if ix == 0 else 1
        for iy, yc in enumerate([y1, y2]):
            sy = -1 if iy == 0 else 1
            for iz, d in enumerate([d1, d2]):
                sz = -1 if iz == 0 else 1
                gz_sum += sx * sy * sz * F(xc - Xo, yc - Yo, d)
    return G_const * rho * gz_sum

def compute_model(blk_list, X_grid, Y_grid, G_const):
    gz_total = np.zeros_like(X_grid, dtype=float)
    for k in range(blk_list.shape[0]):
        gz_total += gravity_prism(X_grid, Y_grid, blk_list[k], G_const)
    return gz_total

# ۳) محاسبه مشتقات (بخش اصلاح شده برای رفع خطا)
dx = x[1] - x[0]
dy = y[1] - y[0]

gz = compute_model(blocks, X, Y, G) * 1e5 

# استفاده از astype(float) برای اطمینان ۱۰۰٪ از عدم بروز خطای Casting
bNear = blocks.astype(float).copy()
bNear[:, 4:6] -= h
bFar = blocks.astype(float).copy()
bFar[:, 4:6] += h

gz_near = compute_model(bNear, X, Y, G) * 1e5
gz_far = compute_model(bFar, X, Y, G) * 1e5
FVD = (gz_near - gz_far) / (2 * h)

# ۴) فیلترها
dgdy, dgdx = np.gradient(gz, dy, dx)
THDR = np.hypot(dgdx, dgdy)
ASA = np.hypot(THDR, FVD)
ThetaMap = np.degrees(np.arctan2(THDR, np.abs(FVD) + 1e-12))

# ۵) WEGF
n_pow = 0.5
eps_val = 1e-6
edge_term = THDR / (ASA + eps_val)
edge_term = edge_term / (np.max(edge_term) + 1e-14)
WEGF = 1.0 * (edge_term**n_pow)

# ۶) ذخیره فایل‌ها
def write_grid(filename, X_grid, Y_grid, Z_grid):
    data = np.column_stack((X_grid.T.flatten(), Y_grid.T.flatten(), Z_grid.T.flatten()))
    np.savetxt(filename, data, fmt='%.0f %.0f %.12g')

write_grid("four_block_gravity_simple.dat", X, Y, gz)
write_grid("four_block_fvd.dat", X, Y, FVD)
write_grid("four_block_thdr.dat", X, Y, THDR)
write_grid("four_block_asa.dat", X, Y, ASA)
write_grid("four_block_theta.dat", X, Y, ThetaMap)
write_grid("four_block_wegf.dat", X, Y, WEGF)

# ۷) رسم
fig, axes = plt.subplots(2, 4, figsize=(18, 9), facecolor="w")
axes = axes.flatten()

# نمایش بلوک‌ها
ax = axes[0]
ax.set_aspect("equal")
for k, b in enumerate(blocks):
    ax.add_patch(plt.Rectangle((b[0], b[2]), b[1]-b[0], b[3]-b[2], color=plt.cm.tab10(k), alpha=0.3))
    ax.text((b[0]+b[1])/2, (b[2]+b[3])/2, str(k+1), ha='center', va='center', fontweight='bold')
ax.set_xlim(0, 1000); ax.set_ylim(0, 1000); ax.set_title("Block footprints"); ax.grid(True)

fields = [gz, FVD, THDR, ASA, ThetaMap, WEGF]
titles = ["Gravity (mGal)", "FVD (mGal/m)", "THDR (mGal/m)", "ASA (mGal/m)", "Theta (deg)", "WEGF"]
for j, (field, title) in enumerate(zip(fields, titles)):
    ax = axes[j+1]
    im = ax.imshow(field, extent=[0, 1000, 0, 1000], origin='lower', cmap='viridis')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

axes[7].axis('off')
plt.tight_layout()
plt.savefig("four_block_results.png", dpi=300)
plt.show()

print(f"Done! Gravity range: {gz.min():.4f} to {gz.max():.4f} mGal")
