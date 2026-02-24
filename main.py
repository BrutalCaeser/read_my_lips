import torch
import hydra
from pipelines.pipeline import InferencePipeline
from chaplin import Chaplin


@hydra.main(version_base=None, config_path="hydra_configs", config_name="default")
def main(cfg):
    chaplin = Chaplin()

    # load the model
    if torch.cuda.is_available() and cfg.gpu_idx >= 0:
        device = torch.device(f"cuda:{cfg.gpu_idx}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    chaplin.vsr_model = InferencePipeline(
        cfg.config_filename, device=device, detector=cfg.detector, face_track=True)

    print("\n\033[48;5;22m\033[97m\033[1m MODEL LOADED SUCCESSFULLY! \033[0m\n")

    # start the webcam video capture
    chaplin.start_webcam()


if __name__ == '__main__':
    main()
