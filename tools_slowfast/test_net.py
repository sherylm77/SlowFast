#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

from unittest import TestLoader
import numpy as np
import os
import pickle
import torch

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters import AVAMeter, TestMeter

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    # test_meter.iter_tic()

    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        # test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = (
                ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            )
            metadata = (
                metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
            )

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            # test_meter.iter_toc()
            # # Update and log stats.
            # test_meter.update_stats(preds, ori_boxes, metadata)
            # test_meter.log_iter_stats(None, cur_iter)
        else:
            # Perform the forward pass.
            # for input in inputs:
            #     pred = model(input)

            vid_ids = test_loader.dataset.video_ids
            frames = test_loader.dataset.frames
            frames = [x/30 for x in frames]
            second_frames = [x for x in frames]
            frames.insert(0, 0) 
            frames.pop(len(frames)-1)
            intervals = np.vstack((frames, second_frames)).T
            input_part1 = inputs[0]
            input_part2 = inputs[1]
            preds = []

            vid_part1 = torch.squeeze(input_part1)
            vid_part2 = torch.squeeze(input_part2)
            stop_flag = False

            if input_part1.shape[2] == 0:
                print("CANT DO VIDEO")
                return test_meter

            for i in range(vid_part2.shape[1]):
                if stop_flag:
                    break
                start = i // 4
                if start + 1 >= vid_part1.shape[1]:
                    stop_flag = True

                vid_1_r = torch.unsqueeze(vid_part1[0][start:start+1], 0)
                vid_1_g = torch.unsqueeze(vid_part1[1][start:start+1], 0)
                vid_1_b = torch.unsqueeze(vid_part1[2][start:start+1], 0)
                vide1 = torch.concat([vid_1_r, vid_1_g, vid_1_b], 0)
                v1 = torch.unsqueeze(vide1, 0)

                if i+4 >= vid_part2.shape[1] - 1:
                    end = vid_part2.shape[1]
                    start = end-4
                    stop_flag = True
                else: 
                    start = i
                    end = i+4

                vid_r = torch.unsqueeze(vid_part2[0][start:end], 0)
                vid_g = torch.unsqueeze(vid_part2[1][start:end], 0)
                vid_b = torch.unsqueeze(vid_part2[2][start:end], 0)
                vide2 = torch.concat([vid_r, vid_g, vid_b], 0)
                v2 = torch.unsqueeze(vide2, 0)
                vid = [v1, v2]
                pred = model(vid, vid_ids[0])
                preds.append(pred)

            output_vec = preds[0]
            if len(preds) != 1:
                output_vec = torch.unsqueeze(output_vec, 0)
                for i in range(1, len(preds)):
                    vec = torch.unsqueeze(preds[i], 0)
                    output_vec = torch.concat([output_vec, vec], 0)

            output_vecs_dir = os.path.join(os.path.dirname(cfg.DATA.PATH_TO_DATA_DIR), "output_vecs")
            if not os.path.exists(output_vecs_dir):
                os.makedirs(output_vecs_dir)
            # pk_file = os.path.join(output_vecs_dir, vid_ids[0] + ".pk")
            # vid_dict = {vid_ids[0]: {'features': output_vec, 'intervals': intervals} }
            np.save(output_vecs_dir + "/output_latent_vec_" + vid_ids[0], output_vec)
            # with open(pk_file, 'wb') as f:
            #     pickle.dump(vid_dict, f)
            print("Saved latent vector for video", vid_ids[0])
        break
            # preds = model(inputs)
            # Gather all the predictions across all the devices to perform ensemble.
    #         if cfg.NUM_GPUS > 1:
    #             preds, labels, video_idx = du.all_gather(
    #                 [preds, labels, video_idx]
    #             )
    #         if cfg.NUM_GPUS:
    #             preds = preds.cpu()
    #             labels = labels.cpu()
    #             video_idx = video_idx.cpu()

    #         test_meter.iter_toc()
    #         # Update and log stats.
    #         test_meter.update_stats(
    #             preds.detach(), labels.detach(), video_idx.detach()
    #         )
    #         test_meter.log_iter_stats(cur_iter)

    #     test_meter.iter_tic()

    # # Log epoch stats and print the final testing results.
    # if not cfg.DETECTION.ENABLE:
    #     all_preds = test_meter.video_preds.clone().detach()
    #     all_labels = test_meter.video_labels
    #     if cfg.NUM_GPUS:
    #         all_preds = all_preds.cpu()
    #         all_labels = all_labels.cpu()
    #     if writer is not None:
    #         writer.plot_eval(preds=all_preds, labels=all_labels)

    #     if cfg.TEST.SAVE_RESULTS_PATH != "":
    #         save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

    #         if du.is_root_proc():
    #             with pathmgr.open(save_path, "wb") as f:
    #                 pickle.dump([all_preds, all_labels], f)

    #         logger.info(
    #             "Successfully saved prediction results to {}".format(save_path)
    #         )

    # test_meter.finalize_metrics()
    return test_meter


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    # if du.is_master_proc() and cfg.LOG_MODEL_INFO:
    #     misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    for vid in cfg.TEST.VIDEOS[0]:
        test_loader = loader.construct_loader(cfg, "test", vid)
        logger.info("Testing model for {} iterations".format(len(test_loader)))
        print("video:", vid)

        if cfg.DETECTION.ENABLE:
            assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
            test_meter = AVAMeter(len(test_loader), cfg, mode="test")
        else:
            # assert (
            #     test_loader.dataset.num_videos
            #     % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
            #     == 0
            # )
            # Create meters for multi-view testing.
            test_meter = TestMeter(
                test_loader.dataset.num_videos
                // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
                cfg.MODEL.NUM_CLASSES,
                len(test_loader),
                cfg.DATA.MULTI_LABEL,
                cfg.DATA.ENSEMBLE_METHOD,
            )

        # Set up writer for logging to Tensorboard format.
        if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
            cfg.NUM_GPUS * cfg.NUM_SHARDS
        ):
            writer = tb.TensorboardWriter(cfg)
        else:
            writer = None

        # # Perform multi-view test on the entire dataset.
        global vid_ids
        vid_ids = test_loader.dataset.video_ids
        test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
        if writer is not None:
            writer.close()
