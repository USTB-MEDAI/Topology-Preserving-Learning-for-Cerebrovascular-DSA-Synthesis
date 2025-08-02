# import cv2
# import numpy as np
# import os

# # 记录鼠标点击的位置和画图状态
# drawing = False
# ix, iy = -1, -1
# layer = None  # 用于绘制的图层

# # 鼠标回调函数
# def draw_circle(event, x, y, flags, param):
#     global ix, iy, drawing, layer

#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix, iy = x, y
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing == True:
#             cv2.line(layer, (ix, iy), (x, y), (255, 255, 255), 2)
#             ix, iy = x, y
#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         cv2.line(layer, (ix, iy), (x, y), (255, 255, 255), 2)

# # 处理每张图像并进行绘制
# def process_image(image_path, output_path):
#     global layer

#     # 读取原图
#     img = cv2.imread(image_path)
#     # 创建一个新的图层，完全透明
#     layer = np.zeros_like(img)

#     # 设置鼠标回调函数
#     cv2.namedWindow("Image")
#     cv2.setMouseCallback("Image", draw_circle)

#     while True:
#         # 显示原图，并叠加绘制图层
#         temp_img = img.copy()
#         temp_img = cv2.addWeighted(temp_img, 1, layer, 0.5, 0)
#         cv2.imshow("Image", temp_img)

#         # 等待用户按键，按 'q' 键退出，按 's' 键保存
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):  # 按 'q' 键退出
#             break
#         elif key == ord('s'):  # 按 's' 键保存当前绘制的图像
#             cv2.imwrite(output_path, layer)
#             print(f"Saved: {output_path}")
#             break

#     cv2.destroyAllWindows()

# # 批量处理文件夹中的图像
# def batch_process(input_dir, output_dir):
#     os.makedirs(output_dir, exist_ok=True)

#     for fname in os.listdir(input_dir):
#         if fname.endswith((".png", ".jpg", ".jpeg")):
#             input_path = os.path.join(input_dir, fname)
#             output_path = os.path.join(output_dir, fname)
#             process_image(input_path, output_path)

# # 示例用法
# if __name__ == "__main__":
#     input_dir = "/disk/ssy/data/DIAS/myDIAS/littletest/handtest"  # 替换为输入图片文件夹的路径
#     output_dir = "/disk/ssy/data/DIAS/myDIAS/hand/test_makehand"  # 替换为输出图片文件夹的路径
#     batch_process(input_dir, output_dir)
