# import numpy as np
# import torch
# from skimage.filters import frangi
# from skimage.morphology import skeletonize

# # -------------------------------------------------
# # ① 骨架提取
# # -------------------------------------------------
# def extract_skeleton(image, frangi_scale_range=(1, 3), threshold=0.1):
#     """
#     image: 2D numpy array, [H,W], float/uint8
#     return: binary skeleton, uint8 {0,1}
#     """
#     # Frangi vesselness
#     img_norm = (image - image.min()) / (image.max() - image.min() + 1e-5)
#     vessel = frangi(img_norm, scale_range=frangi_scale_range)

#     # Threshold & thinning
#     binary = (vessel > threshold).astype(np.uint8)
#     skeleton = skeletonize(binary).astype(np.uint8)
#     return skeleton


# # -------------------------------------------------
# # ② Chamfer 距离
# # -------------------------------------------------
# def compute_chamfer_distance(pred_skel, gt_skel):
#     """
#     pred_skel, gt_skel: binary numpy arrays, shape [H,W]
#     return: torch.Tensor(float)  Chamfer distance
#     """
#     pts_pred = torch.tensor(np.argwhere(pred_skel > 0), dtype=torch.float32)
#     pts_gt   = torch.tensor(np.argwhere(gt_skel  > 0), dtype=torch.float32)

#     if pts_pred.numel() == 0 or pts_gt.numel() == 0:
#         return torch.tensor(0.0, dtype=torch.float32)

#     dists = torch.cdist(pts_pred, pts_gt)            # [N_pred, N_gt]
#     loss  = dists.min(1)[0].mean() + dists.min(0)[0].mean()
#     return loss


# # -------------------------------------------------
# # ③ batch 级骨架正则 Loss
# # -------------------------------------------------
# ## 平均而没返回每个bach
# # def skeleton_loss_batch(pred_imgs, gt_imgs, threshold=0.1):
# #     """
# #     pred_imgs, gt_imgs: torch.Tensor [B,1,H,W]  (0‑1 range)
# #     threshold: Frangi vesselness 阈值
# #     return: float  (batch 平均 Chamfer distance)
# #     """
# #     B = pred_imgs.size(0)
# #     total = 0.0

# #     for i in range(B):
# #         pred_np = pred_imgs[i, 0].detach().cpu().numpy()
# #         gt_np   = gt_imgs[i, 0].detach().cpu().numpy()

# #         sk_pred = extract_skeleton(pred_np, threshold=threshold)
# #         sk_gt   = extract_skeleton(gt_np,   threshold=threshold)

# #         total += compute_chamfer_distance(sk_pred, sk_gt).item()

# #     return total / B

# #版本3
# def skeleton_loss_batch(pred_imgs, gt_imgs, threshold=0.1):
#     B = pred_imgs.size(0)
#     total = 0.0

#     for i in range(B):
#         pred_np = pred_imgs[i, 0].detach().cpu().numpy()
#         gt_np   = gt_imgs[i, 0].detach().cpu().numpy()

#         sk_pred = extract_skeleton(pred_np, threshold=threshold)
#         sk_gt   = extract_skeleton(gt_np, threshold=threshold)

#         total += compute_chamfer_distance(sk_pred, sk_gt).item()

#     return torch.tensor(total / B, dtype=pred_imgs.dtype, device=pred_imgs.device)


# # def skeleton_loss_batch(pred_imgs, gt_imgs, threshold=0.1):
# #     """
# #     pred_imgs, gt_imgs: torch.Tensor [B,1,H,W]  (0‑1 range)
# #     return: torch.Tensor [B]，每个样本一个 Chamfer 距离
# #     """
# #     B = pred_imgs.size(0)
# #     losses = []

# #     for i in range(B):
# #         pred_np = pred_imgs[i, 0].detach().cpu().numpy()
# #         gt_np   = gt_imgs[i, 0].detach().cpu().numpy()

# #         sk_pred = extract_skeleton(pred_np, threshold=threshold)
# #         sk_gt   = extract_skeleton(gt_np, threshold=threshold)

# #         dist = compute_chamfer_distance(sk_pred, sk_gt).item()
# #         losses.append(dist)

# #     return torch.tensor(losses, dtype=torch.float32)  # [B]



# # -------------------------------------------------
# # demo
# # -------------------------------------------------
# # if __name__ == "__main__":
# #     # 随机示例：两张 256×256
# #     pred = torch.rand(2, 1, 256, 256)
# #     gt   = torch.rand(2, 1, 256, 256)
# #     loss = skeleton_loss_batch(pred, gt, threshold=0.15)
# #     print(f"Skeleton‑aware loss = {loss:.4f}")

# if __name__ == "__main__":
#     from PIL import Image

#     # ✅ 手动填写两张图路径
#     pred_path = "/disk/ssy/pyproj/bishe/Xray-Diffsuion-main/logs/ldm/pl_test_ldm-2025-07-08/19-21-16/ztest_image_s48_i4_origin_origin_x.jpg"
#     gt_path = "/disk/ssy/pyproj/bishe/Xray-Diffsuion-main/logs/ldm/pl_test_ldm-2025-07-08/19-21-16/ztest_image_s48_i4_origin_ae_rec.jpg"

#     def load_and_preprocess(image_path, resize=(256, 256)):
#         """
#         加载图片，转灰度，缩放并归一化成 [H,W] numpy float32
#         """
#         img = Image.open(image_path).convert("L")  # 转灰度
#         img = img.resize(resize)
#         img_np = np.array(img).astype(np.float32) / 255.0
#         return img_np

#     # 加载图像为 numpy 数组
#     pred_np = load_and_preprocess(pred_path)
#     gt_np   = load_and_preprocess(gt_path)

#     # 转成 torch tensor，添加 batch 和 channel 维度：[1, 1, H, W]
#     pred_tensor = torch.tensor(pred_np).unsqueeze(0).unsqueeze(0)
#     gt_tensor   = torch.tensor(gt_np).unsqueeze(0).unsqueeze(0)

#     # 调用你原有的骨架loss函数
#     loss = skeleton_loss_batch(pred_tensor, gt_tensor, threshold=0.15)
#     print(f"Skeleton-aware loss = {loss:.4f}")



import torch
import torch.nn.functional as F
from kornia.filters import sobel
from kornia.contrib import distance_transform


def soft_skeletonize(x, thresh=0.5, iters=10):
    for _ in range(iters):
        min_pool = -F.max_pool2d(-x, 3, 1, 1)
        diff = F.relu(x - min_pool)
        x = F.relu(x - diff)
    return (x > thresh).float()


def chamfer_distance_map(skel_a, skel_b):
    dist_a = distance_transform(1.0 - skel_a)
    dist_b = distance_transform(1.0 - skel_b)
    loss_ab = (skel_a * dist_b).mean(dim=(1, 2, 3))
    loss_ba = (skel_b * dist_a).mean(dim=(1, 2, 3))
    return loss_ab + loss_ba


# def skeleton_loss_batch(pred_imgs, gt_imgs, threshold=0.3):
#     skel_pred = soft_skeletonize(pred_imgs, thresh=threshold)
#     skel_gt   = soft_skeletonize(gt_imgs,   thresh=threshold)
#     losses = chamfer_distance_map(skel_pred, skel_gt)
#     return losses  # [B]

def skeleton_loss_batch(pred_imgs, gt_imgs, threshold=0.3):
    # L1 loss
    l1_loss = F.l1_loss(pred_imgs, gt_imgs, reduction='mean')  # 标量

    # Skeleton loss
    skel_pred = soft_skeletonize(pred_imgs, thresh=threshold)
    skel_gt   = soft_skeletonize(gt_imgs,   thresh=threshold)
    skel_loss = chamfer_distance_map(skel_pred, skel_gt).mean()  # → 标量

    # Combine
    #total_loss = skel_loss / 100.0
    total_loss = l1_loss * 0.8 + 0.2 * skel_loss / 1000.0

    print("L1:", l1_loss.item(), "Skel:", skel_loss.item(), "total:", total_loss.item())
    # l1_loss * 0.8 + 
    return total_loss


if __name__ == "__main__":
    from PIL import Image
    import torchvision.transforms as T

    def load_image(path, size=(256, 256)):
        """加载图像并转换为 [1,1,H,W] 的归一化Tensor"""
        image = Image.open(path).convert("L")  # 灰度
        transform = T.Compose([
            T.Resize(size),
            T.ToTensor(),  # -> [1,H,W], 值在[0,1]
        ])
        tensor = transform(image).unsqueeze(0)  # -> [1,1,H,W]
        return tensor

    # 👉 路径直接写在这里
    # pred_path = "./pred.png"
    # gt_path = "./gt.png"
    pred_path = "/disk/ssy/pyproj/bishe/Xray-Diffsuion-main/logs/ldm/pl_test_ldm-2025-07-08/19-21-16/ztest_image_s48_i4_origin_origin_x.jpg"
    gt_path = "/disk/ssy/pyproj/bishe/Xray-Diffsuion-main/logs/ldm/pl_test_ldm-2025-07-08/19-21-16/ztest_image_s48_i4_origin_rec.jpg"

    
    pred_img = load_image(pred_path)
    gt_img = load_image(gt_path)

    pred_img = pred_img.to(torch.float32)
    gt_img = gt_img.to(torch.float32)

    with torch.no_grad():
        loss = skeleton_loss_batch(pred_img, gt_img)
        print(f"Skeleton-aware Chamfer Loss: {loss.item():.6f}")
