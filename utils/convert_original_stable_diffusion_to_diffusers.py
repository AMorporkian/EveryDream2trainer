import diffusers
import torch    
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path", default=None, type=str, required=True, help="Path to the checkpoint to convert."
    )
    parser.add_argument(
        "--original_config_file",
        default=None,
        type=str,
        help="The YAML config file corresponding to the original architecture.",
    )
    parser.add_argument(
        "--num_in_channels",
        default=None,
        type=int,
        help="The number of input channels. If `None` number of input channels will be automatically inferred.",
    )
    parser.add_argument(
        "--scheduler_type",
        default="pndm",
        type=str,
        help="Type of scheduler to use. Should be one of ['pndm', 'lms', 'ddim', 'euler', 'euler-ancest', 'dpm']",
    )
    parser.add_argument(
        "--pipeline_type",
        default=None,
        type=str,
        help="The pipeline type. If `None` pipeline will be automatically inferred.",
    )
    parser.add_argument(
        "--image_size",
        default=None,
        type=int,
        help=(
            "The image size that the model was trained on. Use 512 for Stable Diffusion v1.X and Stable Siffusion v2"
            " Base. Use 768 for Stable Diffusion v2."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        default=None,
        type=str,
        help=(
            "The prediction type that the model was trained on. Use 'epsilon' for Stable Diffusion v1.X and Stable"
            " Diffusion v2 Base. Use 'v-prediction' for Stable Diffusion v2."
        ),
    )
    parser.add_argument(
        "--extract_ema",
        action="store_true",
       help=(
            "Only relevant for checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights"
            " or not. Defaults to `False`. Add `--extract_ema` to extract the EMA weights. EMA weights usually yield"
            " higher quality images for inference. Non-EMA weights are usually better to continue fine-tuning."
        ),
    )
    parser.add_argument(
        "--upcast_attn",
        default=False,
        type=bool,
        help=(
            "Whether the attention computation should always be upcasted. This is necessary when running stable"
            " diffusion 2.1."
        ),
    )
    parser.add_argument("--dump_path", default=None, type=str, required=False, help="Path to the output model.")
    args = parser.parse_args()
    ckpt_path = args.checkpoint_path
    ckpt_base_path = ckpt_path.split(".")[0]
    dump_path = args.dump_path
    if dump_path is None:
        dump_path = ckpt_base_path + "_diffusers"
    name = ckpt_base_path.split("/")[-1]
    pipe = diffusers.StableDiffusionPipeline.from_ckpt(args.checkpoint_path)
    pipe.save_pretrained(dump_path)
    print(" * Converted model to diffusers for training: ", dump_path)