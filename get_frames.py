import os
import subprocess
import re
import pathlib

# uses ffmpeg to extract frames from all videos in given directory
def video_to_frames(data_dir):
    vid_names = os.listdir(data_dir)
    frame_dir = os.path.join(os.path.dirname(data_dir), "frames")
    extension = pathlib.Path(vid_names[0]).suffix
    for vid in vid_names:
        if extension in vid:
            print("MAKING THIS VIDEO INTO MANY PICTURES (FRAMES): ", vid)
            vid_name = os.path.splitext(vid)[0]
            if not os.path.exists(os.path.dirname(vid_name)):
                os.makedirs(os.path.join(frame_dir, vid_name), exist_ok=True)
                vid_path = os.path.join(data_dir, vid)
                output = os.path.join(frame_dir, vid_name, vid_name+"_%03d.jpg")
                subprocess.call('ffmpeg -i {video} -r 30 -q:v 1 {out_name}'.format(video=vid_path, out_name=output), shell=True)

    print("Finished extracting frames.")

# generates csv file with frame paths
def get_frames_csv(data_dir, model_name):
    vid_names = os.listdir(data_dir)
    extension = pathlib.Path(vid_names[0]).suffix
    frame_dir = os.path.join(os.path.dirname(data_dir), "frames")
    if "Kinetics" in model_name:
        frame_csv_path = os.path.join(data_dir, "test.csv")
        frames_csv = open(frame_csv_path, "w")
        frame_folders = os.listdir(frame_dir)
        for folder in frame_folders:
            video = folder + extension
            if video in vid_names:
                frames_csv.write(folder + extension + " 0\n")
        frames_csv.close()
        print("Finished making test.csv for video list.")
    else:
        frame_csv_path = os.path.join(data_dir, "val.csv")
        frames_csv = open(frame_csv_path, "w")
        frame_folders = os.listdir(frame_dir)
        frames_csv.write("original_vido_id video_id frame_id path labels\n")
        for folder in frame_folders:
            video = folder + extension
            if video in vid_names:
                frames = os.listdir(os.path.join(frame_dir, folder))
                frames.sort(key=lambda f: int(re.sub('\D', '', f)))
                frame_id = 0
                for frame in frames:
                    frames_csv.write(folder + " 10 " + str(frame_id) + " " + os.path.join(folder, frame) + " \"\"\n")
                    frame_id += 1
                frame_id = 0
        frames_csv.close()
        print("Finished making val.csv for frame list.")
