"""
Copyright [2022-2023] Victor C Hall

Licensed under the GNU Affero General Public License;
You may not use this code except in compliance with the License.
You may obtain a copy of the License at

    https://www.gnu.org/licenses/agpl-3.0.en.html

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import pprint
import sys
import math
import signal
import argparse
import logging
import time
import gc
import random
import traceback
import shutil

import torch.nn.functional as F
from torch.cuda.amp import autocast

from colorama import Fore, Style
import numpy as np
import torch
import datetime
import json
from tqdm.auto import tqdm

from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DDPMScheduler
#from diffusers.models import AttentionBlock
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer
#from accelerate import Accelerator
from accelerate.utils import set_seed

import wandb
import webbrowser
from torch.utils.tensorboard import SummaryWriter
from data.data_loader import DataLoaderMultiAspect

from data.every_dream import EveryDreamBatch, build_torch_dataloader
from data.every_dream_validation import EveryDreamValidator
from data.image_train_item import ImageTrainItem, DEFAULT_BATCH_ID
from utils.huggingface_downloader import try_download_model_from_hf
from utils.convert_diff_to_ckpt import convert as converter
from utils.isolate_rng import isolate_rng
from utils.check_git import check_git
from optimizer.optimizers import EveryDreamOptimizer

if torch.cuda.is_available():
    from utils.gpu import GPU
import data.aspects as aspects
import data.resolver as resolver
from utils.sample_generator import SampleGenerator

_SIGTERM_EXIT_CODE = 130
_VERY_LARGE_NUMBER = 1e9

def get_hf_ckpt_cache_path(ckpt_path):
    return os.path.join("ckpt_cache", os.path.basename(ckpt_path))

def convert_to_hf(ckpt_path):

    hf_cache = get_hf_ckpt_cache_path(ckpt_path)
    from utils.unet_utils import get_attn_yaml

    if os.path.isfile(ckpt_path):
        if not os.path.exists(hf_cache):
            os.makedirs(hf_cache)
            logging.info(f"Converting {ckpt_path} to Diffusers format")
            try:
                import utils.convert_original_stable_diffusion_to_diffusers as convert
                convert.convert(ckpt_path, f"ckpt_cache/{ckpt_path}")
            except:
                logging.info("Please manually convert the checkpoint to Diffusers format (one time setup), see readme.")
                exit()
        else:
            logging.info(f"Found cached checkpoint at {hf_cache}")

        is_sd1attn, yaml = get_attn_yaml(hf_cache)
        return hf_cache, is_sd1attn, yaml
    elif os.path.isdir(hf_cache):
        is_sd1attn, yaml = get_attn_yaml(hf_cache)
        return hf_cache, is_sd1attn, yaml
    else:
        is_sd1attn, yaml = get_attn_yaml(ckpt_path)
        return ckpt_path, is_sd1attn, yaml

def setup_local_logger(args):
    """
    configures logger with file and console logging, logs args, and returns the datestamp
    """
    log_path = args.logdir

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    json_config = json.dumps(vars(args), indent=2)
    datetimestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    with open(os.path.join(log_path, f"{args.project_name}-{datetimestamp}_cfg.json"), "w") as f:
        f.write(f"{json_config}")

    logfilename = os.path.join(log_path, f"{args.project_name}-{datetimestamp}.log")
    print(f" logging to {logfilename}")
    logging.basicConfig(filename=logfilename,
                        level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        datefmt="%m/%d/%Y %I:%M:%S %p",
                       )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.addFilter(lambda msg: "Palette images with Transparency expressed in bytes" not in msg.getMessage())
    logging.getLogger().addHandler(console_handler)
    import warnings
    warnings.filterwarnings("ignore", message="UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images")
    #from PIL import Image

    return datetimestamp

# def save_optimizer(optimizer: torch.optim.Optimizer, path: str):
#     """
#     Saves the optimizer state
#     """
#     torch.save(optimizer.state_dict(), path)

# def load_optimizer(optimizer: torch.optim.Optimizer, path: str):
#     """
#     Loads the optimizer state
#     """
#     optimizer.load_state_dict(torch.load(path))

def get_gpu_memory(nvsmi):
    """
    returns the gpu memory usage
    """
    gpu_query = nvsmi.DeviceQuery('memory.used, memory.total')
    gpu_used_mem = int(gpu_query['gpu'][0]['fb_memory_usage']['used'])
    gpu_total_mem = int(gpu_query['gpu'][0]['fb_memory_usage']['total'])
    return gpu_used_mem, gpu_total_mem

def append_epoch_log(global_step: int, epoch_pbar, gpu, log_writer, **logs):
    """
    updates the vram usage for the epoch
    """
    if gpu is not None:
        gpu_used_mem, gpu_total_mem = gpu.get_gpu_memory()
        log_writer.add_scalar("performance/vram", gpu_used_mem, global_step)
        epoch_mem_color = Style.RESET_ALL
        if gpu_used_mem > 0.93 * gpu_total_mem:
            epoch_mem_color = Fore.LIGHTRED_EX
        elif gpu_used_mem > 0.85 * gpu_total_mem:
            epoch_mem_color = Fore.LIGHTYELLOW_EX
        elif gpu_used_mem > 0.7 * gpu_total_mem:
            epoch_mem_color = Fore.LIGHTGREEN_EX
        elif gpu_used_mem < 0.5 * gpu_total_mem:
            epoch_mem_color = Fore.LIGHTBLUE_EX

        if logs is not None:
            epoch_pbar.set_postfix(**logs, vram=f"{epoch_mem_color}{gpu_used_mem}/{gpu_total_mem} MB{Style.RESET_ALL} gs:{global_step}")

def set_args_12gb(args):
    logging.info(" Setting args to 12GB mode")
    if not args.gradient_checkpointing:
        logging.info("  - Overiding gradient checkpointing to True")
        args.gradient_checkpointing = True
    if args.batch_size > 2:
        logging.info("  - Overiding batch size to max 2")
        args.batch_size = 2
        args.grad_accum = 1
    if args.resolution > 512:
        logging.info("  - Overiding resolution to max 512")
        args.resolution = 512

def find_last_checkpoint(logdir):
    """
    Finds the last checkpoint in the logdir, recursively
    """
    last_ckpt = None
    last_date = None

    for root, dirs, files in os.walk(logdir):
        for file in files:
            if os.path.basename(file) == "model_index.json":
                curr_date = os.path.getmtime(os.path.join(root,file))

                if last_date is None or curr_date > last_date:
                    last_date = curr_date
                    last_ckpt = root

    assert last_ckpt, f"Could not find last checkpoint in logdir: {logdir}"
    assert "errored" not in last_ckpt, f"Found last checkpoint: {last_ckpt}, but it was errored, cancelling"

    print(f"    {Fore.LIGHTCYAN_EX}Found last checkpoint: {last_ckpt}, resuming{Style.RESET_ALL}")

    return last_ckpt

def setup_args(args):
    """
    Sets defaults for missing args (possible if missing from json config)
    Forces some args to be set based on others for compatibility reasons
    """
    if args.disable_amp:
        logging.warning(f"{Fore.LIGHTYELLOW_EX} Disabling AMP, not recommended.{Style.RESET_ALL}")
        args.amp = False
    else:
        args.amp = True

    if args.disable_unet_training and args.disable_textenc_training:
        raise ValueError("Both unet and textenc are disabled, nothing to train")

    if args.resume_ckpt == "findlast":
        logging.info(f"{Fore.LIGHTCYAN_EX} Finding last checkpoint in logdir: {args.logdir}{Style.RESET_ALL}")
        # find the last checkpoint in the logdir
        args.resume_ckpt = find_last_checkpoint(args.logdir)

    if args.lowvram:
        set_args_12gb(args)

    if not args.shuffle_tags:
        args.shuffle_tags = False

    args.clip_skip = max(min(4, args.clip_skip), 0)

    if args.useadam8bit:
        logging.warning(f"{Fore.LIGHTYELLOW_EX} Useadam8bit arg is deprecated, use optimizer.json instead, which defaults to useadam8bit anyway{Style.RESET_ALL}")

    if args.ckpt_every_n_minutes is None and args.save_every_n_epochs is None:
        logging.info(f"{Fore.LIGHTCYAN_EX} No checkpoint saving specified, defaulting to every 20 minutes.{Style.RESET_ALL}")
        args.ckpt_every_n_minutes = 20

    if args.ckpt_every_n_minutes is None or args.ckpt_every_n_minutes < 1:
        args.ckpt_every_n_minutes = _VERY_LARGE_NUMBER

    if args.save_every_n_epochs is None or args.save_every_n_epochs < 1:
        args.save_every_n_epochs = _VERY_LARGE_NUMBER

    if args.save_every_n_epochs < _VERY_LARGE_NUMBER and args.ckpt_every_n_minutes < _VERY_LARGE_NUMBER:
        logging.warning(f"{Fore.LIGHTYELLOW_EX}** Both save_every_n_epochs and ckpt_every_n_minutes are set, this will potentially spam a lot of checkpoints{Style.RESET_ALL}")
        logging.warning(f"{Fore.LIGHTYELLOW_EX}** save_every_n_epochs: {args.save_every_n_epochs}, ckpt_every_n_minutes: {args.ckpt_every_n_minutes}{Style.RESET_ALL}")

    if args.cond_dropout > 0.26:
        logging.warning(f"{Fore.LIGHTYELLOW_EX}** cond_dropout is set fairly high: {args.cond_dropout}, make sure this was intended{Style.RESET_ALL}")

    if args.grad_accum > 1:
        logging.info(f"{Fore.CYAN} Batch size: {args.batch_size}, grad accum: {args.grad_accum}, 'effective' batch size: {args.batch_size * args.grad_accum}{Style.RESET_ALL}")

    total_batch_size = args.batch_size * args.grad_accum

    if args.save_ckpt_dir is not None and not os.path.exists(args.save_ckpt_dir):
        os.makedirs(args.save_ckpt_dir)

    if args.rated_dataset:
        args.rated_dataset_target_dropout_percent = min(max(args.rated_dataset_target_dropout_percent, 0), 100)

        logging.info(logging.info(f"{Fore.CYAN} * Activating rated images learning with a target rate of {args.rated_dataset_target_dropout_percent}% {Style.RESET_ALL}"))

    args.aspects = aspects.get_aspect_buckets(args.resolution)

    return args


def report_image_train_item_problems(log_folder: str, items: list[ImageTrainItem], batch_size) -> None:
    undersized_items = [item for item in items if item.is_undersized]
    if len(undersized_items) > 0:
        underized_log_path = os.path.join(log_folder, "undersized_images.txt")
        logging.warning(f"{Fore.LIGHTRED_EX} ** Some images are smaller than the target size, consider using larger images{Style.RESET_ALL}")
        logging.warning(f"{Fore.LIGHTRED_EX} ** Check {underized_log_path} for more information.{Style.RESET_ALL}")
        with open(underized_log_path, "w", encoding='utf-8') as undersized_images_file:
            undersized_images_file.write(f" The following images are smaller than the target size, consider removing or sourcing a larger copy:")
            for undersized_item in undersized_items:
                message = f" *** {undersized_item.pathname} with size: {undersized_item.image_size} is smaller than target size: {undersized_item.target_wh}\n"
                undersized_images_file.write(message)


    # warn on underfilled aspect ratio buckets

    # Intuition: if there are too few images to fill a batch, duplicates will be appended.
    # this is not a problem for large image counts but can seriously distort training if there
    # are just a handful of images for a given aspect ratio.

    # at a dupe ratio of 0.5, all images in this bucket have effective multiplier 1.5,
    # at a dupe ratio 1.0, all images in this bucket have effective multiplier 2.0
    warn_bucket_dupe_ratio = 0.5

    def make_bucket_key(item):
        return (item.batch_id, int(item.target_wh[0]), int(item.target_wh[1]))

    ar_buckets = set(make_bucket_key(i) for i in items)
    for ar_bucket in ar_buckets:
        count = len([i for i in items if make_bucket_key(i) == ar_bucket])
        runt_size = batch_size - (count % batch_size)
        bucket_dupe_ratio = runt_size / count
        if bucket_dupe_ratio > warn_bucket_dupe_ratio:
            aspect_ratio_rational = aspects.get_rational_aspect_ratio((ar_bucket[1], ar_bucket[2]))
            aspect_ratio_description = f"{aspect_ratio_rational[0]}:{aspect_ratio_rational[1]}"
            batch_id_description = "" if ar_bucket[0] == DEFAULT_BATCH_ID else f" for batch id '{ar_bucket[0]}'"
            effective_multiplier = round(1 + bucket_dupe_ratio, 1)
            logging.warning(f" * {Fore.LIGHTRED_EX}Aspect ratio bucket {ar_bucket} has only {count} "
                            f"images{Style.RESET_ALL}. At batch size {batch_size} this makes for an effective multiplier "
                            f"of {effective_multiplier}, which may cause problems. Consider adding {runt_size} or "
                            f"more images with aspect ratio {aspect_ratio_description}{batch_id_description}, or reducing your batch_size.")

def resolve_image_train_items(args: argparse.Namespace) -> list[ImageTrainItem]:
    logging.info(f"* DLMA resolution {args.resolution}, buckets: {args.aspects}")
    logging.info(" Preloading images...")

    resolved_items = resolver.resolve(args.data_root, args)
    image_paths = set(map(lambda item: item.pathname, resolved_items))

    # Remove erroneous items
    for item in resolved_items:
        if item.error is not None:
            logging.error(f"{Fore.LIGHTRED_EX} *** Error opening {Fore.LIGHTYELLOW_EX}{item.pathname}{Fore.LIGHTRED_EX} to get metadata. File may be corrupt and will be skipped.{Style.RESET_ALL}")
            logging.error(f" *** exception: {item.error}")
    image_train_items = [item for item in resolved_items if item.error is None]
    print (f" * Found {len(image_paths)} files in '{args.data_root}'")

    return image_train_items

def write_batch_schedule(args: argparse.Namespace, log_folder: str, train_batch: EveryDreamBatch, epoch: int):
    if args.write_schedule:
        with open(f"{log_folder}/ep{epoch}_batch_schedule.txt", "w", encoding='utf-8') as f:
            for i in range(len(train_batch.image_train_items)):
                try:
                    item = train_batch.image_train_items[i]
                    f.write(f"step:{int(i / train_batch.batch_size):05}, wh:{item.target_wh}, r:{item.runt_size}, path:{item.pathname}\n")
                except Exception as e:
                    logging.error(f" * Error writing to batch schedule for file path: {item.pathname}")


def read_sample_prompts(sample_prompts_file_path: str):
    sample_prompts = []
    with open(sample_prompts_file_path, "r") as f:
        for line in f:
            sample_prompts.append(line.strip())
    return sample_prompts

def log_args(log_writer, args):
    arglog = "args:\n"
    for arg, value in sorted(vars(args).items()):
        arglog += f"{arg}={value}, "
    log_writer.add_text("config", arglog)

@torch.no_grad()
def __save_model(save_path, unet, text_encoder, tokenizer, scheduler, vae, ed_optimizer, save_ckpt_dir, yaml_name,
                    save_full_precision=False, save_optimizer_flag=False, save_ckpt=True):
    """
    Save the model to disk
    """
    global global_step
    if global_step is None or global_step == 0:
        logging.warning("  No model to save, something likely blew up on startup, not saving")
        return
    logging.info(f" * Saving diffusers model to {save_path}")
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None, # save vram
        requires_safety_checker=None, # avoid nag
        feature_extractor=None, # must be none of no safety checker
    )
    pipeline.save_pretrained(save_path)
    sd_ckpt_path = f"{os.path.basename(save_path)}.ckpt"

    if save_ckpt:
        if save_ckpt_dir is not None:
            sd_ckpt_full = os.path.join(save_ckpt_dir, sd_ckpt_path)
        else:
            sd_ckpt_full = os.path.join(os.curdir, sd_ckpt_path)
            save_ckpt_dir = os.curdir

        half = not save_full_precision

        logging.info(f" * Saving SD model to {sd_ckpt_full}")
        converter(model_path=save_path, checkpoint_path=sd_ckpt_full, half=half)

        if yaml_name and yaml_name != "v1-inference.yaml":
            yaml_save_path = f"{os.path.join(save_ckpt_dir, os.path.basename(save_path))}.yaml"
            logging.info(f" * Saving yaml to {yaml_save_path}")
            shutil.copyfile(yaml_name, yaml_save_path)

    if save_optimizer_flag:
        logging.info(f" Saving optimizer state to {save_path}")
        ed_optimizer.save(save_path)
def create_sigterm_handler(log_folder, unet, text_encoder, tokenizer, noise_scheduler, vae, ed_optimizer, args):
    def sigterm_handler(signum, frame):
        """
        handles sigterm
        """
        is_main_thread = (torch.utils.data.get_worker_info() == None)
        if is_main_thread:
            global interrupted
            if not interrupted:
                interrupted=True
                global global_step
                #TODO: save model on ctrl-c
                interrupted_checkpoint_path = os.path.join(f"{log_folder}/ckpts/interrupted-gs{global_step}")
                print()
                logging.error(f"{Fore.LIGHTRED_EX} ************************************************************************{Style.RESET_ALL}")
                logging.error(f"{Fore.LIGHTRED_EX} CTRL-C received, attempting to save model to {interrupted_checkpoint_path}{Style.RESET_ALL}")
                logging.error(f"{Fore.LIGHTRED_EX} ************************************************************************{Style.RESET_ALL}")
                time.sleep(2) # give opportunity to ctrl-C again to cancel save
                __save_model(interrupted_checkpoint_path, unet, text_encoder, tokenizer, noise_scheduler, vae, ed_optimizer, args.save_ckpt_dir, args.save_full_precision, args.save_optimizer, save_ckpt=not args.no_save_ckpt)
            exit(_SIGTERM_EXIT_CODE)
        else:
            # non-main threads (i.e. dataloader workers) should exit cleanly
            exit(0)
def get_model_prediction_and_target(image, tokens, vae, text_encoder, noise_scheduler, unet, zero_frequency_noise_ratio=0.0):
    with torch.no_grad():
        with autocast(enabled=args.amp):
            pixel_values = image.to(memory_format=torch.contiguous_format).to(unet.device)
            latents = vae.encode(pixel_values, return_dict=False)
        del pixel_values
        latents = latents[0].sample() * 0.18215

        if zero_frequency_noise_ratio > 0.0:
            # see https://www.crosslabs.org//blog/diffusion-with-offset-noise
            zero_frequency_noise = zero_frequency_noise_ratio * torch.randn(latents.shape[0], latents.shape[1], 1, 1, device=latents.device)
            noise = torch.randn_like(latents) + zero_frequency_noise
        else:
            noise = torch.randn_like(latents)

        bsz = latents.shape[0]

        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        cuda_caption = tokens.to(text_encoder.device)

    encoder_hidden_states = text_encoder(cuda_caption, output_hidden_states=True)

    if args.clip_skip > 0:
        encoder_hidden_states = text_encoder.text_model.final_layer_norm(
            encoder_hidden_states.hidden_states[-args.clip_skip])
    else:
        encoder_hidden_states = encoder_hidden_states.last_hidden_state

    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type in ["v_prediction", "v-prediction"]:
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
    del noise, latents, cuda_caption

    with autocast(enabled=args.amp):
        #print(f"types: {type(noisy_latents)} {type(timesteps)} {type(encoder_hidden_states)}")
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

    return model_pred, target

def generate_samples(global_step: int, sample_generator, unet, text_encoder, tokenizer, vae, reference_scheduler, device, batch):
    with isolate_rng():
        prev_sample_steps = sample_generator.sample_steps
        sample_generator.reload_config()
        if prev_sample_steps != sample_generator.sample_steps:
            next_sample_step = math.ceil((global_step + 1) / sample_generator.sample_steps) * sample_generator.sample_steps
            print(f" * SampleGenerator config changed, now generating images samples every " +
                    f"{sample_generator.sample_steps} training steps (next={next_sample_step})")
        sample_generator.update_random_captions(batch["captions"])
        inference_pipe = sample_generator.create_inference_pipe(unet=unet,
                                                                text_encoder=text_encoder,
                                                                tokenizer=tokenizer,
                                                                vae=vae,
                                                                diffusers_scheduler_config=reference_scheduler.config
                                                                ).to(device)
        sample_generator.generate_samples(inference_pipe, global_step)

        del inference_pipe
    gc.collect()
    torch.cuda.empty_cache()

def make_save_path(epoch, global_step, prepend=""):
    return os.path.join(f"{log_folder}/ckpts/{prepend}{args.project_name}-ep{epoch:02}-gs{global_step:05}")

def log_epoch(args, log_folder, log_writer, train_batch, epoch_pbar, epoch_times, epoch, loss_epoch, epoch_start_time, steps_pbar):
    steps_pbar.close()

    elapsed_epoch_time = (time.time() - epoch_start_time) / 60
    epoch_times.append(dict(epoch=epoch, time=elapsed_epoch_time))
    log_writer.add_scalar("performance/minutes per epoch", elapsed_epoch_time, global_step)

    epoch_pbar.update(1)
    if epoch < args.max_epochs - 1:
        train_batch.shuffle(epoch_n=epoch, max_epochs = args.max_epochs)
        write_batch_schedule(args, log_folder, train_batch, epoch + 1)

    loss_local = sum(loss_epoch) / len(loss_epoch)
    log_writer.add_scalar(tag="loss/epoch", scalar_value=loss_local, global_step=global_step)

def complete_training(args, yaml, text_encoder, vae, unet, noise_scheduler, tokenizer, ed_optimizer, epoch_times, training_start_time):
    epoch = args.max_epochs
    save_path = make_save_path(epoch, global_step, prepend=("" if args.no_prepend_last else "last-"))
    __save_model(save_path, unet, text_encoder, tokenizer, noise_scheduler, vae, ed_optimizer, args.save_ckpt_dir, yaml, args.save_full_precision, args.save_optimizer, save_ckpt=not args.no_save_ckpt)

    total_elapsed_time = time.time() - training_start_time
    logging.info(f"{Fore.CYAN}Training complete{Style.RESET_ALL}")
    logging.info(f"Total training time took {total_elapsed_time/60:.2f} minutes, total steps: {global_step}")
    logging.info(f"Average epoch time: {np.mean([t['time'] for t in epoch_times]):.2f} minutes")
    logging.info(f"{Fore.LIGHTWHITE_EX} ***************************{Style.RESET_ALL}")
    logging.info(f"{Fore.LIGHTWHITE_EX} **** Finished training ****{Style.RESET_ALL}")
    logging.info(f"{Fore.LIGHTWHITE_EX} ***************************{Style.RESET_ALL}")
    return epoch

def log_step(args, gpu, log_writer, ed_optimizer, epoch_pbar, loss_log_step, images_per_sec_log_step, images_per_sec):
    if (global_step + 1) % args.log_step == 0:
        loss_local = sum(loss_log_step) / len(loss_log_step)
        lr_unet = ed_optimizer.get_unet_lr()
        lr_textenc = ed_optimizer.get_textenc_lr()
        loss_log_step = []
                    
        log_writer.add_scalar(tag="hyperparameter/lr unet", scalar_value=lr_unet, global_step=global_step)
        log_writer.add_scalar(tag="hyperparameter/lr text encoder", scalar_value=lr_textenc, global_step=global_step)
        log_writer.add_scalar(tag="loss/log_step", scalar_value=loss_local, global_step=global_step)

        sum_img = sum(images_per_sec_log_step)
        avg = sum_img / len(images_per_sec_log_step)
        images_per_sec_log_step = []
        if args.amp:
            log_writer.add_scalar(tag="hyperparameter/grad scale", scalar_value=ed_optimizer.get_scale(), global_step=global_step)
        log_writer.add_scalar(tag="performance/images per second", scalar_value=avg, global_step=global_step)

        logs = {"loss/log_step": loss_local, "lr_unet": lr_unet, "lr_te": lr_textenc, "img/s": images_per_sec}
        append_epoch_log(global_step=global_step, epoch_pbar=epoch_pbar, gpu=gpu, log_writer=log_writer, **logs)
        torch.cuda.empty_cache()



class Trainer:
    def __init__(self, args, log_time, log_folder, optimizer_config):
        self.args = args
        self.log_time = log_time
        self.log_folder = log_folder
        self.optimizer_config = optimizer_config
        self.optimizer_config_path = args.optimizer_config_path
        self.validator = None
        self.train_dataloader = None

    def main(self):
        """
        Main entry point
        """
        args, log_time, seed, device, gpu, log_folder = init(self.args)

        yaml, text_encoder, vae, unet, reference_scheduler, noise_scheduler, tokenizer, optimizer_config = load_models(self.args, device)

        log_writer = setup_logging(self.args, log_time, log_folder, optimizer_config)

        validator, train_batch = create_image_loaders(self.args, seed, log_folder, tokenizer, log_writer)

        torch.cuda.benchmark = False

        epoch_len = math.ceil(len(train_batch) / self.args.batch_size)

        ed_optimizer = EveryDreamOptimizer(self.args,
                                        optimizer_config,
                                        text_encoder,
                                        unet,
                                        epoch_len)

        log_self.args(log_writer, self.args)

        sample_generator = SampleGenerator(log_folder=log_folder, log_writer=log_writer,
                                        default_resolution=self.args.resolution, default_seed=self.args.seed,
                                        config_file_path=self.args.sample_prompts,
                                        batch_size=max(1,self.args.batch_size//2),
                                        default_sample_steps=self.args.sample_steps,
                                        use_xformers=is_xformers_available() and not self.args.disable_xformers,
                                        use_penultimate_clip_layer=(self.args.clip_skip >= 2)
                                        )

        """
        Train the model

        """
        train_dataloader, epoch_pbar, epoch_times, training_start_time, last_epoch_saved_time, loss_log_step = perform_train_setup(self.args, gpu, log_folder, text_encoder, vae, unet, noise_scheduler, tokenizer, log_writer, train_batch, epoch_len, ed_optimizer)

        # Pre-train validation to establish a starting point on the loss graph
        if validator:
            validator.do_validation(global_step=0,
                                    get_model_prediction_and_target_callable=get_model_prediction_and_target)

        # the sample generator might be configured to generate samples before step 0
        if sample_generator.generate_pretrain_samples:
            _, batch = next(enumerate(train_dataloader))
            generate_samples(global_step=0, sample_generator=sample_generator, unet=unet, text_encoder=text_encoder, tokenizer=tokenizer, vae=vae, reference_scheduler=reference_scheduler, device=device, batch=batch)

        try:
            write_batch_schedule(self.args, log_folder, train_batch, epoch = 0)

            for epoch in range(self.args.max_epochs):
                loss_epoch, epoch_start_time, steps_pbar = do_epoch(self.args, gpu, yaml, text_encoder, vae, unet, noise_scheduler, tokenizer, log_writer, validator, train_batch, ed_optimizer, sample_generator, train_dataloader, epoch_pbar, last_epoch_saved_time, loss_log_step, epoch)
                    # end of step

                log_epoch(self.args, log_folder, log_writer, train_batch, epoch_pbar, epoch_times, epoch, loss_epoch, epoch_start_time, steps_pbar)

                gc.collect()
                # end of epoch

            # end of training
            epoch = complete_training(self.args, yaml, text_encoder, vae, unet, noise_scheduler, tokenizer, ed_optimizer, epoch_times, training_start_time)

        except Exception as ex:
            logging.error(f"{Fore.LIGHTYELLOW_EX}Something went wrong, attempting to save model{Style.RESET_ALL}")
            save_path = make_save_path(epoch, global_step, prepend="errored-")
            __save_model(save_path, unet, text_encoder, tokenizer, noise_scheduler, vae, ed_optimizer, self.args.save_ckpt_dir, yaml, self.args.save_full_precision, self.args.save_optimizer, save_ckpt=not self.args.no_save_ckpt)
            raise ex
    def do_epoch(self, train_batch):
        loss_epoch = []
        epoch_start_time = time.time()
        images_per_sec_log_step = []

        epoch_len = math.ceil(len(train_batch) / self.args.batch_size)
        steps_pbar = tqdm(range(epoch_len), position=1, leave=False, dynamic_ncols=True)
        steps_pbar.set_description(f"{Fore.LIGHTCYAN_EX}Steps{Style.RESET_ALL}")

        validation_steps = (
                    [] if self.validator is None
                    else self.validator.get_validation_step_indices(epoch, len(train_dataloader))
                )

        for step, batch in enumerate(self.train_dataloader):
            images_per_sec = self.do_step(self.args, text_encoder, vae, unet, noise_scheduler, ed_optimizer, loss_log_step, batch, loss_epoch, images_per_sec_log_step, steps_pbar, step)

            log_step(self.args, gpu, log_writer, ed_optimizer, epoch_pbar, loss_log_step, images_per_sec_log_step, images_per_sec)

            if validator and step in validation_steps:
                validator.do_validation(global_step, get_model_prediction_and_target)

            if (global_step + 1) % sample_generator.sample_steps == 0:
                generate_samples(global_step=global_step, batch=batch)

            min_since_last_ckpt =  (time.time() - last_epoch_saved_time) / 60

            if self.args.ckpt_every_n_minutes is not None and (min_since_last_ckpt > self.args.ckpt_every_n_minutes):
                last_epoch_saved_time = time.time()
                logging.info(f"Saving model, {self.args.ckpt_every_n_minutes} mins at step {global_step}")
                save_path = make_save_path(epoch, global_step)
                __save_model(save_path, unet, text_encoder, tokenizer, noise_scheduler, vae, ed_optimizer, self.args.save_ckpt_dir, yaml, self.args.save_full_precision, self.args.save_optimizer, save_ckpt=not self.args.no_save_ckpt)

            if epoch > 0 and epoch % self.args.save_every_n_epochs == 0 and step == 0 and epoch < self.args.max_epochs - 1 and epoch >= self.args.save_ckpts_from_n_epochs:
                logging.info(f" Saving model, {self.args.save_every_n_epochs} epochs at step {global_step}")
                save_path = make_save_path(epoch, global_step)
                __save_model(save_path, unet, text_encoder, tokenizer, noise_scheduler, vae, ed_optimizer, self.args.save_ckpt_dir, yaml, self.args.save_full_precision, self.args.save_optimizer, save_ckpt=not self.args.no_save_ckpt)

            del batch
            global_step += 1
        return loss_epoch,epoch_start_time,steps_pbar
    def train(self):
        data_loader = DataLoaderMultiAspect(
            image_train_items=self.args.image_train_items,
            seed=self.args.seed,
            batch_size=self.args.batch_size,
            grad_accum=self.args.grad_accum,
        )

        self.train_batch = EveryDreamBatch(
            data_loader=data_loader,
            debug_level=1,
            conditional_dropout=self.args.cond_dropout,
            tokenizer=self.tokenizer,
            seed=self.args.seed,
            shuffle_tags=self.args.shuffle_tags,
            rated_dataset=self.args.rated_dataset,
            rated_dataset_dropout_target=(
                1.0
                - (self.args.rated_dataset_target_dropout_percent / 100.0)
            ),
        )

        self.validator = None
        if self.args.validation:
            self.validator = EveryDreamBatch(
                data_loader=self.args.validation,
                debug_level=1,
                conditional_dropout=self.args.cond_dropout,
                tokenizer=self.tokenizer,
                seed=self.args.seed,
                shuffle_tags=self.args.shuffle_tags,
                rated_dataset=self.args.rated_dataset,
                rated_dataset_dropout_target=(
                    1.0
                    - (self.args.rated_dataset_target_dropout_percent / 100.0)
                ),
            )

        return self.validator, self.train_batch

    def setup_logging(self):
        if self.args.wandb:
            wandb.tensorboard.patch(
                root_logdir=self.log_folder,
                pytorch=False,
                tensorboard_x=False,
                save=False,
            )
            wandb_run = wandb.init(
                project=self.args.project_name,
                config={
                    "main_cfg": vars(self.args),
                    "optimizer_cfg": self.optimizer_config,
                },
                name=self.args.run_name,
            )
            try:
                if webbrowser.get():
                    webbrowser.open(wandb_run.url, new=2)
            except Exception:
                pass

        self.log_writer = SummaryWriter(
            log_dir=self.log_folder,
            flush_secs=20,
            comment=(
                self.args.run_name
                if self.args.run_name is not None
                else self.log_time
            ),
        )

        return self.log_writer

    def load_models(self, device):
        optimizer_state_path = None
        try:
            # check for a local file
            hf_cache_path = get_hf_ckpt_cache_path(self.args.resume_ckpt)
            if (
                os.path.exists(hf_cache_path)
                or os.path.exists(self.args.resume_ckpt)
            ):
                model_root_folder, is_sd1attn, yaml = convert_to_hf(
                    self.args.resume_ckpt
                )
                self.text_encoder = CLIPTextModel.from_pretrained(
                    model_root_folder, subfolder="text_encoder"
                )
                self.vae = AutoencoderKL.from_pretrained(
                    model_root_folder, subfolder="vae"
                )
                self.unet = UNet2DConditionModel.from_pretrained(
                    model_root_folder, subfolder="unet"
                )

                optimizer_state_path = os.path.join(
                    self.args.resume_ckpt, "optimizer.pt"
                )
                if not os.path.exists(optimizer_state_path):
                    optimizer_state_path = None
            else:
                # try to download from HF using resume_ckpt as a repo id
                downloaded = try_download_model_from_hf(
                    repo_id=self.args.resume_ckpt
                )
                if downloaded is None:
                    raise ValueError(
                        f"No local file/folder for {self.args.resume_ckpt}, and no matching huggingface.co repo could be downloaded"
                    )
                pipe, model_root_folder, is_sd1attn, yaml = downloaded
                self.text_encoder = pipe.text_encoder
                self.vae = pipe.vae
                self.unet = pipe.unet
                del pipe

            _args = {
                "subfolder": "scheduler",
                "rescale_betas_zero_snr": self.args.enable_zero_terminal_snr,
                "timestep_spacing": (
                    "trailing"
                    if self.args.enable_trailing_timesteps
                    else "leading"
                ),
                "prediction_type": (
                    "v_prediction" if self.args.v_prediction else None
                ),
            }
            self.reference_scheduler = DDIMScheduler.from_pretrained(
                self.args.model_root_folder, **_args
            )
            self.noise_scheduler = DDPMScheduler.from_pretrained(
                self.args.model_root_folder,
                subfolder="scheduler",
                trained_betas=self.reference_scheduler.betas.detach().clone(),
            )

            self.tokenizer = CLIPTokenizer.from_pretrained(
                model_root_folder, subfolder="tokenizer", use_fast=False
            )

        except Exception as e:
            traceback.print_exc()
            logging.error(" * Failed to load checkpoint *")
            raise

        if self.args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            self.text_encoder.gradient_checkpointing_enable()

        if not self.args.disable_xformers:
            if (self.args.amp and is_sd1attn) or (not is_sd1attn):
                try:
                    self.unet.enable_xformers_memory_efficient_attention()
                    logging.info("Enabled xformers")
                except Exception as ex:
                    logging.warning(
                        "failed to load xformers, using attention slicing instead"
                    )
                    self.unet.set_attention_slice("auto")
                    pass
            elif (not self.args.amp and is_sd1attn):
                logging.info(
                    "AMP is disabled but model is SD1.X, using attention slicing instead of xformers"
                )
                self.unet.set_attention_slice("auto")
        else:
            logging.info(
                "xformers disabled via arg, using attention slicing instead"
            )
            self.unet.set_attention_slice("auto")

        self.vae = self.vae.to(
            device, dtype=torch.float16 if self.args.amp else torch.float32
        )
        self.unet = self.unet.to(device, dtype=torch.float32)
        if (
            self.args.disable_textenc_training
            and self.args.amp
        ):
            self.text_encoder = self.text_encoder.to(
                device, dtype=torch.float16
            )
        else:
            self.text_encoder = self.text_encoder.to(
                device, dtype=torch.float32
            )

        try:
            torch.compile(self.unet)
            torch.compile(self.text_encoder)
            torch.compile(self.vae)
            logging.info("Successfully compiled models")
        except Exception as ex:
            logging.warning(
                f"Failed to compile model, continuing anyway, ex: {ex}"
            )
            pass

        
        self.optimizer_config_path = (
            self.args.optimizer_config
            if self.args.optimizer_config
            else "optimizer.json"
        )
        if os.path.exists(
            os.path.join(os.curdir, self.optimizer_config_path)
        ):
            with open(
                os.path.join(os.curdir, self.optimizer_config_path), "r"
            ) as f:
                self.optimizer_config = json.load(f)
    def do_step(self, args, text_encoder, vae, unet, noise_scheduler, ed_optimizer, loss_log_step, batch, loss_epoch, images_per_sec_log_step, steps_pbar, step):
        step_start_time = time.time()

        model_pred, target = get_model_prediction_and_target(batch["image"], batch["tokens"], vae, text_encoder, noise_scheduler, unet, args.zero_frequency_noise_ratio)

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        del target, model_pred

        if batch["runt_size"] > 0:
            loss_scale = batch["runt_size"] / args.batch_size
            loss = loss * loss_scale

        ed_optimizer.step(loss, step, global_step)

        loss_step = loss.detach().item()

        steps_pbar.set_postfix({"loss/step": loss_step}, {"gs": global_step})
        steps_pbar.update(1)

        images_per_sec = args.batch_size / (time.time() - step_start_time)
        images_per_sec_log_step.append(images_per_sec)

        loss_log_step.append(loss_step)
        loss_epoch.append(loss_step)
        return images_per_sec

    def perform_train_setup(self, args, gpu, log_folder, text_encoder, vae, unet, noise_scheduler, tokenizer, log_writer, train_batch, epoch_len, ed_optimizer):
        print(f" {Fore.LIGHTGREEN_EX}** Welcome to EveryDream trainer 2.0!**{Style.RESET_ALL}")
        print(f" (C) 2022-2023 Victor C Hall  This program is licensed under AGPL 3.0 https://www.gnu.org/licenses/agpl-3.0.en.html")
        print()
        print("** Trainer Starting **")

        global interrupted
        interrupted = False

        
        signal.signal(signal.SIGINT, create_sigterm_handler(log_folder, unet, text_encoder, tokenizer, noise_scheduler, vae, ed_optimizer, args))

        if not os.path.exists(f"{log_folder}/samples/"):
            os.makedirs(f"{log_folder}/samples/")

        if gpu is not None:
            gpu_used_mem, gpu_total_mem = gpu.get_gpu_memory()
            logging.info(f" Pretraining GPU Memory: {gpu_used_mem} / {gpu_total_mem} MB")
        logging.info(f" saving ckpts every {args.ckpt_every_n_minutes} minutes")
        logging.info(f" saving ckpts every {args.save_every_n_epochs } epochs")

        train_dataloader = build_torch_dataloader(train_batch, batch_size=args.batch_size)

        unet.train() if not args.disable_unet_training else unet.eval()
        text_encoder.train() if not args.disable_textenc_training else text_encoder.eval()

        logging.info(f" unet device: {unet.device}, precision: {unet.dtype}, training: {unet.training}")
        logging.info(f" text_encoder device: {text_encoder.device}, precision: {text_encoder.dtype}, training: {text_encoder.training}")
        logging.info(f" vae device: {vae.device}, precision: {vae.dtype}, training: {vae.training}")
        logging.info(f" scheduler: {noise_scheduler.__class__}")

        logging.info(f" {Fore.GREEN}Project name: {Style.RESET_ALL}{Fore.LIGHTGREEN_EX}{args.project_name}{Style.RESET_ALL}")
        logging.info(f" {Fore.GREEN}grad_accum: {Style.RESET_ALL}{Fore.LIGHTGREEN_EX}{args.grad_accum}{Style.RESET_ALL}"),
        logging.info(f" {Fore.GREEN}batch_size: {Style.RESET_ALL}{Fore.LIGHTGREEN_EX}{args.batch_size}{Style.RESET_ALL}")
        logging.info(f" {Fore.GREEN}epoch_len: {Fore.LIGHTGREEN_EX}{epoch_len}{Style.RESET_ALL}")

        epoch_pbar = tqdm(range(args.max_epochs), position=0, leave=True, dynamic_ncols=True)
        epoch_pbar.set_description(f"{Fore.LIGHTCYAN_EX}Epochs{Style.RESET_ALL}")
        epoch_times = []

        global global_step
        global_step = 0
        training_start_time = time.time()
        last_epoch_saved_time = training_start_time

        append_epoch_log(global_step=global_step, epoch_pbar=epoch_pbar, gpu=gpu, log_writer=log_writer)

        loss_log_step = []

        assert len(train_batch) > 0, "train_batch is empty, check that your data_root is correct"
        return train_dataloader,epoch_pbar,epoch_times,training_start_time,last_epoch_saved_time,loss_log_step

if __name__ == "__main__":
    check_git()
    supported_resolutions = aspects.get_supported_resolutions()
    argparser = argparse.ArgumentParser(description="EveryDream2 Training options")
    argparser.add_argument("--config", type=str, required=False, default=None, help="JSON config file to load options from")
    args, argv = argparser.parse_known_args()

    if args.config is not None:
        print(f"Loading training config from {args.config}.")
        with open(args.config, 'rt') as f:
            args.__dict__.update(json.load(f))
            if len(argv) > 0:
                print(f"Config .json loaded but there are additional CLI arguments -- these will override values in {args.config}.")
    else:
        print("No config file specified, using command line args")

    argparser = argparse.ArgumentParser(description="EveryDream2 Training options")
    argparser.add_argument("--amp", action="store_true",  default=True, help="deprecated, use --disable_amp if you wish to disable AMP")
    argparser.add_argument("--batch_size", type=int, default=2, help="Batch size (def: 2)")
    argparser.add_argument("--ckpt_every_n_minutes", type=int, default=None, help="Save checkpoint every n minutes, def: 20")
    argparser.add_argument("--clip_grad_norm", type=float, default=None, help="Clip gradient norm (def: disabled) (ex: 1.5), useful if loss=nan?")
    argparser.add_argument("--clip_skip", type=int, default=0, help="Train using penultimate layer (def: 0) (2 is 'penultimate')", choices=[0, 1, 2, 3, 4])
    argparser.add_argument("--cond_dropout", type=float, default=0.04, help="Conditional drop out as decimal 0.0-1.0, see docs for more info (def: 0.04)")
    argparser.add_argument("--data_root", type=str, default="input", help="folder where your training images are")
    argparser.add_argument("--disable_amp", action="store_true", default=False, help="disables training of text encoder (def: False)")
    argparser.add_argument("--disable_textenc_training", action="store_true", default=False, help="disables training of text encoder (def: False)")
    argparser.add_argument("--disable_unet_training", action="store_true", default=False, help="disables training of unet (def: False) NOT RECOMMENDED")
    argparser.add_argument("--disable_xformers", action="store_true", default=False, help="disable xformers, may reduce performance (def: False)")
    argparser.add_argument("--flip_p", type=float, default=0.0, help="probability of flipping image horizontally (def: 0.0) use 0.0 to 1.0, ex 0.5, not good for specific faces!")
    argparser.add_argument("--gpuid", type=int, default=0, help="id of gpu to use for training, (def: 0) (ex: 1 to use GPU_ID 1)")
    argparser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="enable gradient checkpointing to reduce VRAM use, may reduce performance (def: False)")
    argparser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation factor (def: 1), (ex, 2)")
    argparser.add_argument("--logdir", type=str, default="logs", help="folder to save logs to (def: logs)")
    argparser.add_argument("--log_step", type=int, default=25, help="How often to log training stats, def: 25, recommend default!")
    argparser.add_argument("--lowvram", action="store_true", default=False, help="automatically overrides various args to support 12GB gpu")
    argparser.add_argument("--lr", type=float, default=None, help="Learning rate, if using scheduler is maximum LR at top of curve")
    argparser.add_argument("--lr_decay_steps", type=int, default=0, help="Steps to reach minimum LR, default: automatically set")
    argparser.add_argument("--lr_scheduler", type=str, default="constant", help="LR scheduler, (default: constant)", choices=["constant", "linear", "cosine", "polynomial"])
    argparser.add_argument("--lr_warmup_steps", type=int, default=None, help="Steps to reach max LR during warmup (def: 0.02 of lr_decay_steps), non-functional for constant")
    argparser.add_argument("--max_epochs", type=int, default=300, help="Maximum number of epochs to train for")
    argparser.add_argument("--no_prepend_last", action="store_true", help="Do not prepend 'last-' to the final checkpoint filename")
    argparser.add_argument("--no_save_ckpt", action="store_true", help="Save only diffusers files, no .ckpts" )
    argparser.add_argument("--optimizer_config", default="optimizer.json", help="Path to a JSON configuration file for the optimizer.  Default is 'optimizer.json'")
    argparser.add_argument("--project_name", type=str, default="myproj", help="Project name for logs and checkpoints, ex. 'tedbennett', 'superduperV1'")
    argparser.add_argument("--resolution", type=int, default=512, help="resolution to train", choices=supported_resolutions)
    argparser.add_argument("--resume_ckpt", type=str, required=not ('resume_ckpt' in args), default="sd_v1-5_vae.ckpt", help="The checkpoint to resume from, either a local .ckpt file, a converted Diffusers format folder, or a Huggingface.co repo id such as stabilityai/stable-diffusion-2-1 ")
    argparser.add_argument("--run_name", type=str, required=False, default=None, help="Run name for wandb (child of project name), and comment for tensorboard, (def: None)")
    argparser.add_argument("--sample_prompts", type=str, default="sample_prompts.txt", help="Text file with prompts to generate test samples from, or JSON file with sample generator settings (default: sample_prompts.txt)")
    argparser.add_argument("--sample_steps", type=int, default=250, help="Number of steps between samples (def: 250)")
    argparser.add_argument("--save_ckpt_dir", type=str, default=None, help="folder to save checkpoints to (def: root training folder)")
    argparser.add_argument("--save_every_n_epochs", type=int, default=None, help="Save checkpoint every n epochs, def: 0 (disabled)")
    argparser.add_argument("--save_ckpts_from_n_epochs", type=int, default=0, help="Only saves checkpoints starting an N epochs, def: 0 (disabled)")
    argparser.add_argument("--save_full_precision", action="store_true", default=False, help="save ckpts at full FP32")
    argparser.add_argument("--save_optimizer", action="store_true", default=False, help="saves optimizer state with ckpt, useful for resuming training later")
    argparser.add_argument("--seed", type=int, default=555, help="seed used for samples and shuffling, use -1 for random")
    argparser.add_argument("--shuffle_tags", action="store_true", default=False, help="randomly shuffles CSV tags in captions, for booru datasets")
    argparser.add_argument("--useadam8bit", action="store_true", default=False, help="deprecated, use --optimizer_config and optimizer.json instead")
    argparser.add_argument("--wandb", action="store_true", default=False, help="enable wandb logging instead of tensorboard, requires env var WANDB_API_KEY")
    argparser.add_argument("--validation_config", default=None, help="Path to a JSON configuration file for the validator.  Default is no validation.")
    argparser.add_argument("--write_schedule", action="store_true", default=False, help="write schedule of images and their batches to file (def: False)")
    argparser.add_argument("--rated_dataset", action="store_true", default=False, help="enable rated image set training, to less often train on lower rated images through the epochs")
    argparser.add_argument("--rated_dataset_target_dropout_percent", type=int, default=50, help="how many images (in percent) should be included in the last epoch (Default 50)")
    argparser.add_argument("--zero_frequency_noise_ratio", type=float, default=0.02, help="adds zero frequency noise, for improving contrast (def: 0.0) use 0.0 to 0.15, set to -1 to use zero terminal SNR noising beta schedule instead")
    # create an argument group for experimental arguments
    experimental_group = argparser.add_argument_group('experimental')

    # add the experimental arguments to the group
    experimental_group.add_argument("--v_prediction", action="store_true", default=False, help="enable v prediction loss (def: False)")
    experimental_group.add_argument("--enable_trailing_timesteps", action="store_true", default=False, help="enable trailing timesteps (def: False)")
    experimental_group.add_argument("--enable_zero_terminal_snr", action="store_true", default=False, help="enable zero terminal SNR (def: False)")
    experimental_group.add_argument("--sweep", action="store_true", default=False, help="enable sweep mode, disables saving and runs until terminated on the same dataset.(def: False)")
    # load CLI args to overwrite existing config args
    args = argparser.parse_args(args=argv, namespace=args)
    
    main(args)
