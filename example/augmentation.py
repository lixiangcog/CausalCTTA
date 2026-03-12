import cv2
import numpy as np
import albumentations as A
from albumentations.core.composition import Compose
import matplotlib.pyplot as plt

# ------------------------------------
# OCT 数据增强类（使用 albumentations）
# ------------------------------------
class OCTAugmentor:
    def __init__(self):
        self.transform = Compose([
            A.Rotate(limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.GaussNoise(var_limit=(5, 25), p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.ElasticTransform(alpha=20, sigma=5, alpha_affine=5, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5)
        ])

    def __call__(self, img):
        return self.transform(image=img)["image"]


# -------------------------
# 示例：生成 4×4 图像对比
# -------------------------
if __name__ == "__main__":
    # 读取原图
    img = cv2.imread("OPTIC/Fundus/Drishti_GS/test/image/gdrishtiGS_001.png", cv2.IMREAD_GRAYSCALE)

    aug = OCTAugmentor()

    # 生成 15 张增强
    augmented_imgs = [aug(img) for _ in range(15)]
    all_imgs = [img] + augmented_imgs

    # ------------------------------------
    # 可视化：4 × 4 网格显示
    # ------------------------------------
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    for ax, im in zip(axes.ravel(), all_imgs):
        ax.imshow(im, cmap='gray')
        ax.axis("off")

    plt.tight_layout()
    plt.show()
