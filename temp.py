import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# npz 파일 로드
data = np.load("slicer\MedSAM2\MedSAM2\TotalSegmentator_mask_result.npz")
mask = data["segs"]
print("Mask shape:", mask.shape, "Unique:", np.unique(mask))

# 초기 slice index
z = mask.shape[0] // 2

# figure 만들기
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# 초기 이미지
im = ax.imshow(mask[z], cmap="nipy_spectral")
ax.set_title(f"Segmentation mask (z={z})")
cbar = plt.colorbar(im, ax=ax)

# 슬라이더 추가
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, 'Slice', 0, mask.shape[0]-1, valinit=z, valfmt='%d')

# 슬라이더 이벤트
def update(val):
    z_new = int(slider.val)
    im.set_data(mask[z_new])
    ax.set_title(f"Segmentation mask (z={z_new})")
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()
