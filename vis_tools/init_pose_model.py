import mmcv
from models import build_posenet
from mmcv.runner import load_checkpoint


def init_pose_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a pose model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): CUDA or CPU device

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_posenet(config.model)
    if checkpoint is not None:
        # load model checkpoint
        load_checkpoint(model, checkpoint, map_location='cpu')
    # save the config in the model for convenience
    model.cfg = config
    model.to(device)
    model.eval()
    return model
