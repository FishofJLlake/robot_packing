import cv2
import os
import glob

def make_video():
    output_dir = 'output'
    video_path = os.path.join(output_dir, 'packing_process.mp4')
    
    # 查找所有 step 图片并排序
    image_files = sorted(glob.glob(os.path.join(output_dir, 'step_*.png')))
    
    if not image_files:
        print("未找到任何图片 (step_*.png)，请先运行 visualize_demo.py。")
        return
        
    print(f"找到 {len(image_files)} 张图片，开始生成视频 (1Hz)...")
    
    # 读取第一张图片获取宽高
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"无法读取图片 {image_files[0]}")
        return
        
    height, width, layers = first_image.shape
    
    # 初始化视频写入器 (1 fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 或使用 'XVID' 配合 .avi
    video = cv2.VideoWriter(video_path, fourcc, 1.0, (width, height))
    
    # 将图片写入视频
    for file in image_files:
        img = cv2.imread(file)
        if img is not None:
            video.write(img)
            print(f"  已添加: {os.path.basename(file)}")
            
    # 额外将总览图也加入到视频末尾并定格几秒？(可选，先只加step)
    
    video.release()
    print(f"\n✅ 视频已生成: {video_path}")

if __name__ == "__main__":
    make_video()
