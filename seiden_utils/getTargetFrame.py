import cv2
import os
from tqdm import tqdm

def read_file_to_array(filename):
    """
    Reads the content of a file into a Python array.

    Args:
        filename: The name of the file to read.

    Returns:
        A Python array containing the content of the file.
    """

    with open(filename, "r") as file:
        # Read the entire file content
        content = file.read()

        # Split the content into lines
        lines = content.splitlines()

        # Create an empty array to store the numbers
        numbers = []

        # Iterate through each line and extract numbers
        for line in lines:
            # Split the line into individual numbers (handling spaces)
            line_numbers = [int(num) for num in line.split() if num.isdigit()]
            numbers.extend(line_numbers)

    return numbers

def extract_frames(video_path, frame_numbers, output_folder):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Unable to open video file")
        return

    # 确保输出文件夹存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 逐个提取指定帧并保存为图像文件
    for frame_number in tqdm(frame_numbers, desc="Extracting frames", unit="frame"):
        # 设置视频帧位置到指定的帧编号
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # 逐帧读取视频，直到找到目标帧
        success, frame = cap.read()

        # 检查是否成功读取帧
        if not success:
            print(f"Error: Unable to read frame {frame_number} from video")
            continue

        # 构造输出图像文件路径
        output_path = os.path.join(output_folder, f"frame_{frame_number}.jpg")

        # 将帧保存为图像文件
        cv2.imwrite(output_path, frame)

    # 关闭视频文件
    cap.release()

# # 视频文件路径
# video_path = "D:/Projects/PyhtonProjects/thesis/video_data/cherry_1h/video.mp4"
#
# # 帧编号序列（从0开始）
# frame_numbers = read_file_to_array("test.txt")
#
# # 输出文件夹路径
# output_folder = "D:/Projects/PyhtonProjects/thesis/OutPutResults"
#
# # 提取并保存指定帧
# extract_frames(video_path, frame_numbers, output_folder)
