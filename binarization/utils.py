from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def make_specific_artifacts_dir(artifacts_dir: Path) -> Path:
    """Makes the directory <artifacts_dir>/<year_month_day>/<hour_minute_second>/"""
    str_now = datetime.now().strftime(r"%Y_%m_%d-%H_%M_%S")
    str_now_parent, str_now_child = str_now.split('-')
    specific_artifacts_dir = Path(artifacts_dir, str_now_parent, str_now_child)
    specific_artifacts_dir.mkdir(parents=True, exist_ok=True)
    return specific_artifacts_dir


def create_tensorboard_logger(specific_artifacts_dir: Path) -> SummaryWriter:
    """Creates a Tensorboard logger, caching in <specific_artifacts_dir>/runs/"""
    runs_dir = specific_artifacts_dir / 'runs'
    runs_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=runs_dir)


# TODO: not sure if it is worth to spend too much time here, maybe have a
# look at Yolact's implementation of a class that manages paths
# def get_last_checkpoint_dir(artifacts_dir: Path) -> Path:
#     """Retrives the last created directory for saving model checkpoints"""
#     parents = sorted(artifacts_dir.iterdir())
#     i = 1
#     children = sorted(parents[-i].iterdir())
#     while not list((children[-i] / 'checkpoints').iterdir()):
#         i += 1
#         children = sorted(parents[-i].iterdir())
#     last_child = children[-i]
#     return last_child / 'checkpoints'
# def get_last_checkpoint(artifacts_dir: Path) -> Path:
#     """Retrieves the last saved model checkpoint"""
#     last_checkpoint_dir = get_last_checkpoint_dir(artifacts_dir)
#     return sorted(last_checkpoint_dir.iterdir())[-1]
