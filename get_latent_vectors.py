# imports

from sklearn import model_selection
from tools_slowfast import run_net
import argparse
import os
import yaml
from moviepy.editor import VideoFileClip

from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

import get_frames

def edit_config_file(model_name, vid_dir_path):
    num_clips = 0
    videos = {}
    for vid in os.listdir(vid_dir_path):
        if ".mp4" in vid:
            clip = VideoFileClip(os.path.join(vid_dir_path, vid))
            num_clips += 30*clip.duration
            videos[vid] = int(clip.duration)
    num_clips = int(num_clips) + 10

    config_file_path = os.path.join("configs", model_name+".yaml")
    print(config_file_path, "\n")

    with open(config_file_path) as cfg_file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)

    cfg["TRAIN"]["ENABLE"] = False
    if "SSV2" in model_name:
        cfg["TRAIN"]["CHECKPOINT_TYPE"] = "pytorch"
    cfg["TEST"]["ENABLE"] = True
    cfg["TEST"]["CHECKPOINT_FILE_PATH"] = os.path.basename(model_name) + ".pkl"
    cfg["TEST"]["BATCH_SIZE"] = num_clips
    cfg["TEST"]["VIDEOS"] = [videos]
    cfg["DEMO"]["ENABLE"] = False
    cfg["DEMO"]["LABEL_FILE_PATH"] = "validation_labels.json"
    cfg["DATA"]["PATH_TO_DATA_DIR"] = vid_dir_path
    cfg["NUM_GPUS"] = 0
    if "Kinetics" in model_name:
        cfg["DATA"]["PATH_PREFIX"] = vid_dir_path
    else:
        cfg["DATA"]["PATH_PREFIX"] = os.path.join(os.path.dirname(vid_dir_path), "frames")
    with open(config_file_path, "w") as cfg_file:
        yaml.dump(cfg, cfg_file)

# function to get latent vectors (add description)
def get_latent_vectors():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid_dir_path", help="path to directory with input videos")
    parser.add_argument("--model_config_name", help="see PySlowFast Model Zoo for config files for each model")
    args = parser.parse_args()

    vid_dir_path = args.vid_dir_path
    model_name = args.model_config_name

    print(vid_dir_path, model_name)

    # call frames script to create frames
    get_frames.video_to_frames(vid_dir_path)
    get_frames.get_frames_csv(vid_dir_path, model_name)

    edit_config_file(model_name, vid_dir_path)

    # run run_net.py
    run_net.main(model_name)
    # video_model_builder.py and head_helper.py are modified to save latent vectors

def main():
    get_latent_vectors()

if __name__ == "__main__":
    main()