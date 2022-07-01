import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path='conf', config_name='config')
def future_main(cfg: DictConfig) -> None:
    print(cfg.paths.train_dir)


if __name__ == '__main__':
    future_main()
