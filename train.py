import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #str(cvd)
import argparse
from config_loader.config import get_config
from trainer.trainer import train
from utils.parser_helper import str2bool
import wandb
# import Args
# import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('--config', default= 'cl4ac/config/base.yml')
parser.add_argument('--freeze_bert', type=str2bool, default=None)
parser.add_argument('--freeze_cnn', type=str2bool, default=None)
parser.add_argument('--batch', type=int, default=None)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--experiment_name', type=str, default=None)
parser.add_argument('--lr', type=float, default=None)

args = parser.parse_args()
config_path = args.config
freeze_bert = args.freeze_bert
freeze_cnn = args.freeze_cnn
batch_size = args.batch
lr = args.lr
experiment_name = args.experiment_name
device = args.device
config = get_config(config_path)
if batch_size is not None:
    config.training.batch_size = args.batch
if freeze_bert is not None:
    config.bert.freeze_bert = freeze_bert
if freeze_cnn is not None:
    config.encoder.freeze_cnn = freeze_cnn
if lr is not None:
    config.optimizer.lr = lr
if experiment_name is not None:
    config.experiment.name = experiment_name


# from params_proto.hyper import Sweep
# parser = argparse.ArgumentParser()
# parser.add_argument("sweep_file",
#                     type=str, help="sweep file")
# parser.add_argument("-l", "--line-number",
#                     type=int, help="line number of the sweep-file")
# args = parser.parse_args()

# # Obtain kwargs from Sweep
# sweep = Sweep(Args).load(args.sweep_file)
# kwargs = list(sweep)[args.line_number]

# # Set prefix
# job_name = kwargs['job_name']

# # Obtain / generate wandb_runid
# params_dir = os.path.join(Args.checkpoint_root, RUN.prefix)
# params_path = os.path.join(Args.checkpoint_root, RUN.prefix, 'wandb_id.txt')
# if os.path.exists(params_path):
#     print(f'Checkpoint file is found at {params_path}')
#     print('Resume training...')
#     # Read from wandb_id.txt
#     with open(params_path, "r") as f:
#         wandb_runid = f.read()
#     resume=True
# else:
#     print(f'Checkpoint file is not found. Start training from scratch!')
#     resume=False
#     wandb_runid = wandb.util.generate_id()
#     os.makedirs(params_dir)
#     with open(params_path, "x") as f:
#         f.write(wandb_runid)

# if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    # avail_gpus = [3]
    # cvd = avail_gpus[args.line_number % len(avail_gpus)]
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" #str(cvd)
# Args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sweep_basename = Path(args.sweep_file).stem
# sweep_basename = sweep_basename.split('--')[0]  # Remove separator!

project = f'audio captioning'
wandb.login()
wandb.init(
    # Set the project where this run will be logged
    project=project
    # group=sweep_basename,
    # config=vars(Args),
    # resume=resume,
    # id=wandb_runid
)

# Exit if completed
if wandb.run.summary.get('completed', False):
    print('The job seems to have been completed!!')
    wandb.finish()
    quit()

# Controlled exit before slurm kills the process
try:
    train(config, device)
    wandb.run.summary['completed'] = True
    wandb.finish()
except TimeoutError as e:
    wandb.mark_preempting()
    raise e

wandb.finish()