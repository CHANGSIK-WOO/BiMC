from yacs.config import CfgNode as CN
from utils.util import set_gpu, set_seed
import argparse

def print_args(cfg):
    print("************")
    print("** Config **")
    print("************")
    print(cfg)
    print("************")


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """

    # Device setting
    cfg.DEVICE = CN()
    cfg.DEVICE.DEVICE_NAME = ''
    cfg.DEVICE.GPU_ID = ''

    cfg.METHOD = ''
    cfg.SEED = -1

    # For dataset config
    cfg.DATASET = CN()
    cfg.DATASET.NAME = ''
    cfg.DATASET.ROOT = ''
    cfg.DATASET.GPT_PATH = ''
    cfg.DATASET.NUM_CLASSES   = -1
    cfg.DATASET.NUM_INIT_CLS  = -1
    cfg.DATASET.NUM_INC_CLS   = -1
    cfg.DATASET.NUM_BASE_SHOT = -1
    cfg.DATASET.NUM_INC_SHOT  = -1
    cfg.DATASET.BETA = -1.0
    cfg.DATASET.ENSEMBLE_ALPHA = -1.0
    
    # For data
    cfg.DATALOADER = CN()
    cfg.DATALOADER.TRAIN = CN()
    cfg.DATALOADER.TRAIN.BATCH_SIZE_BASE = -1
    cfg.DATALOADER.TRAIN.BATCH_SIZE_INC = -1
    cfg.DATALOADER.TEST = CN()
    cfg.DATALOADER.TEST.BATCH_SIZE = -1
    cfg.DATALOADER.NUM_WORKERS = -1

    # For model
    cfg.MODEL = CN()
    cfg.MODEL.BACKBONE = CN()
    cfg.MODEL.BACKBONE.NAME = ''

    # For methods
    cfg.TRAINER = CN()
    cfg.TRAINER.BiMC = CN()
    cfg.TRAINER.BiMC.METHOD = 'bimc'  # bimc, bimc_ensemble, edge
    cfg.TRAINER.BiMC.PREC = ''
    cfg.TRAINER.BiMC.VISION_CALIBRATION = False
    cfg.TRAINER.BiMC.LAMBDA_I = -1.0
    cfg.TRAINER.BiMC.TAU = -1
    cfg.TRAINER.BiMC.TEXT_CALIBRATION = False
    cfg.TRAINER.BiMC.LAMBDA_T = -1.0
    cfg.TRAINER.BiMC.GAMMA_BASE = -1.0
    cfg.TRAINER.BiMC.GAMMA_INC = -1.0
    cfg.TRAINER.BiMC.USING_ENSEMBLE = False

    # EDGE-specific config
    cfg.TRAINER.BiMC.EDGE = CN()
    cfg.TRAINER.BiMC.EDGE.ENABLED = True
    cfg.TRAINER.BiMC.EDGE.SIGMA = 1.0
    cfg.TRAINER.BiMC.EDGE.GAMMA = 0.6
    cfg.TRAINER.BiMC.EDGE.KERNEL_TYPE = "laplacian"
    cfg.TRAINER.BiMC.EDGE.INFERENCE_EDGE = False
    cfg.TRAINER.BiMC.EDGE.SAVE_IMAGE = False
    cfg.TRAINER.BiMC.EDGE.SAVE_CLASSES = []

    # Meta-learning config
    cfg.TRAINER.BiMC.META = CN()
    cfg.TRAINER.BiMC.META.ENABLED = False
    cfg.TRAINER.BiMC.META.NUM_EPISODES = 100
    cfg.TRAINER.BiMC.META.INNER_LR = 0.01
    cfg.TRAINER.BiMC.META.OUTER_LR = 0.001
    cfg.TRAINER.BiMC.META.INNER_STEPS = 3
    cfg.TRAINER.BiMC.META.BASE_SUPPORT_CLASSES = 200
    cfg.TRAINER.BiMC.META.BASE_QUERY_CLASSES = 40
    cfg.TRAINER.BiMC.META.INC_SUPPORT_CLASSES = 30
    cfg.TRAINER.BiMC.META.INC_QUERY_CLASSES = 5
    cfg.TRAINER.BiMC.META.PROMPT_LENGTH = 4
    cfg.TRAINER.BiMC.META.PROMPT_DIM = 512
    cfg.TRAINER.BiMC.META.BATCH_SIZE = 16  # Small batch size for meta-learning



    

def setup_cfg(dataset_cfg_file, method_cfg_file):
    cfg = CN()
    extend_cfg(cfg)

    # 1. From the dataset config file
    cfg.merge_from_file(dataset_cfg_file)

    # 2. From the method config file
    cfg.merge_from_file(method_cfg_file)

    # add paths before freezing
    cfg.DATA_CFG_PATH = dataset_cfg_file
    cfg.TRAIN_CFG_PATH = method_cfg_file

    cfg.freeze()
    return cfg


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Run the pipeline")

    parser.add_argument('--data_cfg', type=str, help="Path to the data configuration file")
    parser.add_argument('--train_cfg', type=str, help="Path to the training configuration file")
    parser.add_argument('--hyperparam_sweep', action='store_true',
                        help="Run hyperparameter sweep for edge method")
    parser.add_argument('--meta', action='store_true',
                        help="Run meta-learning for prompt learning")
    parser.add_argument('--prompt_checkpoint', type=str, default=None,
                        help="Path to prompt checkpoint (skips meta-learning if provided)")

    args = parser.parse_args()

    data_cfg = args.data_cfg
    train_cfg = args.train_cfg

    cfg = setup_cfg(data_cfg, train_cfg)

    # Set the random seed and GPU ID
    set_seed(cfg.SEED)
    set_gpu(cfg.DEVICE.GPU_ID)

    # Import and run the trainer
    from engine.engine import Runner

    if args.hyperparam_sweep:
        # Hyperparameter grid for edge method experiments
        # 5x5 = 25 combinations
        sigma_values = [0.5, 1.0, 1.5, 2.0, 2.5]
        edge_mix_weight_values = [0.2, 0.4, 0.5, 0.6, 0.8]

        print("=" * 60)
        print("Starting Hyperparameter Sweep for Edge Method")
        print(f"Sigma values: {sigma_values}")
        print(f"Edge mix weight values: {edge_mix_weight_values}")
        print(f"Total combinations: {len(sigma_values) * len(edge_mix_weight_values)}")
        print("=" * 60)

        total_runs = len(sigma_values) * len(edge_mix_weight_values)
        current_run = 0

        for sigma in sigma_values:
            for edge_mix_weight in edge_mix_weight_values:
                current_run += 1
                hyperparam_dict = {
                    'sigma': sigma,
                    'edge_mix_weight': edge_mix_weight
                }

                print("\n" + "=" * 60)
                print(f"Run {current_run}/{total_runs}")
                print(f"Hyperparameters: sigma={sigma}, edge_mix_weight={edge_mix_weight}")
                print("=" * 60 + "\n")

                # Create a new engine for each run
                engine = Runner(cfg)
                engine.run(hyperparam_dict=hyperparam_dict)

                print(f"\nCompleted run {current_run}/{total_runs}")

        print("\n" + "=" * 60)
        print("Hyperparameter sweep completed!")
        print("=" * 60)
    else:
        # Single run without hyperparameter sweep
        engine = Runner(cfg)

        if args.meta:
            # Meta-learning or checkpoint-based mode
            if args.prompt_checkpoint:
                ckpt = args.prompt_checkpoint
            else:
                # Meta-learning mode: train prompts
                print("\n" + "=" * 60)
                print("Starting Meta-Learning for Learnable Prompts")
                print("=" * 60 + "\n")
                ckpt = engine.meta_run()
                print("\n" + "=" * 60)
                print("Meta-Learning Completed!")
                print("Now running evaluation with trained prompts...")
                print("=" * 60 + "\n")
            # Load prompts from checkpoint (skip meta-learning)
            print("\n" + "=" * 60)
            print("Loading Prompts from Checkpoint")
            print(f"Checkpoint: {ckpt}")
            print("=" * 60 + "\n")
            engine.load_prompt_checkpoint(ckpt)

            # After meta-learning or loading checkpoint, run evaluation with prompts
            engine.run(use_meta_prompts=True)
        else:
            # Normal run without meta-learning
            engine.run()


if __name__ == '__main__':
    main()