"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import os, sys, time, shutil, tempfile, datetime, pathlib, subprocess
from pathlib import Path
import numpy as np
from tqdm import trange, tqdm
from urllib.parse import urlparse
import torch

import logging

models_logger = logging.getLogger(__name__)

from . import transforms, dynamics, utils, plot
from .resnet_torch import CPnet
from .core import assign_device, check_mkl, run_net, run_3D

_MODEL_URL = "https://www.cellpose.org/models"
#Try to get the value named CELLPOSE_LOCAL_MODELS_PATH from the environment variable, which, if set, will serve as the local storage path for Cellpose models.
_MODEL_DIR_ENV = os.environ.get("CELLPOSE_LOCAL_MODELS_PATH")
#Default local model storage path, under .cellpose_.models in the user home directory
_MODEL_DIR_DEFAULT = pathlib.Path.home().joinpath(".cellpose_", "models")
#Determine the final path of the model
MODEL_DIR = pathlib.Path(_MODEL_DIR_ENV) if _MODEL_DIR_ENV else _MODEL_DIR_DEFAULT
#The model weights do not match and can only be used with cyto3 and cyto
MODEL_NAMES = [
    "cyto3", "nuclei", "cyto2_cp3", "tissuenet_cp3", "livecell_cp3", "yeast_PhC_cp3",
    "yeast_BF_cp3", "bact_phase_cp3", "bact_fluor_cp3", "deepbacs_cp3", "cyto2", "cyto",
    "transformer_cp3", "neurips_cellpose_default", "neurips_cellpose_transformer",
    "neurips_grayscale_cyto2"
]
#If you have other types of weights you can add them too
MODEL_LIST_PATH = os.fspath(MODEL_DIR.joinpath("gui_models.txt"))

normalize_default = {
    "lowhigh": None,
    "percentile": None,
    "normalize": True,
    "norm3D": False,
    "sharpen_radius": 0,
    "smooth_radius": 0,
    "tile_norm_blocksize": 0,
    "tile_norm_smooth3D": 1,
    "invert": False
}

def model_path(model_type, model_index=0):
    torch_str = "torch"
    if model_type == "cyto" or model_type == "cyto2" or model_type == "nuclei":
        basename = "%s%s_%d" % (model_type, torch_str, model_index)
    else:
        basename = model_type
    return cache_model_path(basename)
    
def size_model_path(model_type):
    if os.path.exists(model_type):
        return model_type + "_size.npy"
    else:
        torch_str = "torch"
        if model_type == "cyto" or model_type == "nuclei" or model_type == "cyto2":
            basename = "size_%s%s_0.npy" % (model_type, torch_str)
        else:
            basename = "size_%s.npy" % model_type
        return cache_model_path(basename)

#Download the pre-training file from the given address
def cache_model_path(basename):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{_MODEL_URL}/{basename}"
    cached_file = os.fspath(MODEL_DIR.joinpath(basename))
    if not os.path.exists(cached_file):
        models_logger.info('Downloading: "{}" to {}\n'.format(url, cached_file))
        utils.download_url_to_file(url, cached_file, progress=True)
    return cached_file


def get_user_models():
    model_strings = []
    if os.path.exists(MODEL_LIST_PATH):
        with open(MODEL_LIST_PATH, "r") as textfile:
            lines = [line.rstrip() for line in textfile]
            if len(lines) > 0:
                model_strings.extend(lines)
    return model_strings


class Cellpose():
    def __init__(self, gpu=False, model_type="cyto3", nchan=2, 
                device=None, backbone="default"):
        super(Cellpose, self).__init__()

        # assign device (GPU or CPU)
        sdevice, gpu = assign_device(use_torch=True, gpu=gpu)
        self.device = device if device is not None else sdevice
        self.gpu = gpu
        self.backbone = backbone

        model_type = "cyto3" if model_type is None else model_type

        self.diam_mean = 30.  #default for any cyto model
        nuclear = "nuclei" in model_type
        if nuclear:
            self.diam_mean = 17.

        if model_type in ["cyto", "nuclei", "cyto2", "cyto3"] and nchan!=2:
            nchan = 2
            models_logger.warning(f"cannot set nchan to other value for {model_type} model")
        self.nchan = nchan

        self.cp = CellposeModel(device=self.device, gpu=self.gpu, model_type=model_type,
                                diam_mean=self.diam_mean, nchan=self.nchan, 
                                backbone=self.backbone)
        self.cp.model_type = model_type

        # size model not used for bacterial model
        self.pretrained_size = size_model_path(model_type)
        self.sz = SizeModel(device=self.device, pretrained_size=self.pretrained_size,
                            cp_model=self.cp)
        self.sz.model_type = model_type

    def eval(self, x, batch_size=8, channels=[0,0], channel_axis=None, invert=False,
             normalize=True, diameter=30., do_3D=False, find_masks=True, **kwargs):

        tic0 = time.time()
        models_logger.info(f"channels set to {channels}")

        diam0 = diameter[0] if isinstance(diameter, (np.ndarray, list)) else diameter
        estimate_size = True if (diameter is None or diam0 == 0) else False

        if estimate_size and self.pretrained_size is not None and not do_3D and x[
                0].ndim < 4:
            tic = time.time()
            models_logger.info("~~~ ESTIMATING CELL DIAMETER(S) ~~~")
            diams, _ = self.sz.eval(x, channels=channels, channel_axis=channel_axis,
                                    batch_size=batch_size, normalize=normalize,
                                    invert=invert)
            diameter = None
            models_logger.info("estimated cell diameter(s) in %0.2f sec" %
                               (time.time() - tic))
            models_logger.info(">>> diameter(s) = ")
            if isinstance(diams, list) or isinstance(diams, np.ndarray):
                diam_string = "[" + "".join(["%0.2f, " % d for d in diams]) + "]"
            else:
                diam_string = "[ %0.2f ]" % diams
            models_logger.info(diam_string)
        elif estimate_size:
            if self.pretrained_size is None:
                reason = "no pretrained size model specified in model Cellpose"
            else:
                reason = "does not work on non-2D images"
            models_logger.warning(f"could not estimate diameter, {reason}")
            diams = self.diam_mean
        else:
            diams = diameter

        models_logger.info("~~~ FINDING MASKS ~~~")
        masks, flows, styles = self.cp.eval(x, channels=channels,
                                            channel_axis=channel_axis,
                                            batch_size=batch_size, normalize=normalize,
                                            invert=invert, diameter=diams, do_3D=do_3D,
                                            **kwargs)
        models_logger.info(">>>> TOTAL TIME %0.2f sec" % (time.time() - tic0))

        return masks, flows, styles, diams


class CellposeModel():

    def __init__(self, gpu=False, pretrained_model=False, model_type=None,
                 diam_mean=30., device=None, nchan=2, backbone="default"):

        self.diam_mean = diam_mean

        ### set model path
        default_model = "cyto3" if backbone=="default" else "transformer_cp3"
        builtin = False
        use_default = False
        model_strings = get_user_models()
        all_models = MODEL_NAMES.copy()
        all_models.extend(model_strings)

        # check if pretrained_model is builtin or custom user model saved in .cellpose_/models
        # if yes, then set to model_type
        if (pretrained_model and not Path(pretrained_model).exists() and 
            np.any([pretrained_model == s for s in all_models])):
            model_type = pretrained_model

        # check if model_type is builtin or custom user model saved in .cellpose_/models
        if model_type is not None and np.any([model_type == s for s in all_models]):
            if np.any([model_type == s for s in MODEL_NAMES]):
                builtin = True
            models_logger.info(f">> {model_type} << model set to be used")
            if model_type == "nuclei":
                self.diam_mean = 17.
            pretrained_model = model_path(model_type)
        # if model_type is not None and does not exist, use default model 
        elif model_type is not None:
            if Path(model_type).exists():
                pretrained_model = model_type
            else:
                models_logger.warning("model_type does not exist, using default model")
                use_default = True
        # if model_type is None...
        else:
            # if pretrained_model does not exist, use default model
            if pretrained_model and not Path(pretrained_model).exists():
                models_logger.warning("pretrained_model path does not exist, using default model")
                use_default = True
            
        builtin = True if use_default else builtin
        self.pretrained_model = model_path(default_model) if use_default else pretrained_model
        
        ### assign model device
        self.mkldnn = None
        if device is None:
            sdevice, gpu = assign_device(use_torch=True, gpu=gpu)
        self.device = device if device is not None else sdevice
        if device is not None:
            device_gpu = self.device.type == "cuda"
        self.gpu = gpu if device is None else device_gpu
        if not self.gpu:
            self.mkldnn = check_mkl(True)

        ### create neural network
        self.nchan = nchan
        self.nclasses = 3
        nbase = [32, 64, 128, 256]
        self.nbase = [nchan, *nbase]
        self.pretrained_model = pretrained_model
        if backbone=="default":
            self.net = CPnet(self.nbase, self.nclasses, sz=3, mkldnn=self.mkldnn,
                             max_pool=True, diam_mean=diam_mean).to(self.device)
        else:
            from .segformer import Transformer
            self.net = Transformer(encoder_weights="imagenet" if not self.pretrained_model else None,
                                     diam_mean=diam_mean).to(self.device)

        ### load model weights
        if self.pretrained_model:
            models_logger.info(f">>>> loading model {pretrained_model}")
            self.net.load_model(self.pretrained_model, device=self.device)
            if not builtin:
                self.diam_mean = self.net.diam_mean.data.cpu().numpy()[0]
            self.diam_labels = self.net.diam_labels.data.cpu().numpy()[0]
            models_logger.info(
                f">>>> model diam_mean = {self.diam_mean: .3f} (ROIs rescaled to this size during training)"
            )
            if not builtin:
                models_logger.info(
                    f">>>> model diam_labels = {self.diam_labels: .3f} (mean diameter of training ROIs)"
                )
        else:
            models_logger.info(f">>>> no model weights loaded")
            self.diam_labels = self.diam_mean

        self.net_type = f"cellpose_{backbone}"

    def eval(self, x, batch_size=8, resample=True, channels=None, channel_axis=None,
             z_axis=None, normalize=True, invert=False, rescale=None, diameter=None,
             flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False, anisotropy=None,
             stitch_threshold=0.0, min_size=15, niter=None, augment=False, tile=True,
             tile_overlap=0.1, bsize=224, interp=True, compute_masks=True,
             progress=None):

        if isinstance(x, list) or x.squeeze().ndim == 5:
            self.timing = []
            masks, styles, flows = [], [], []
            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            nimg = len(x)
            iterator = trange(nimg, file=tqdm_out,
                              mininterval=30) if nimg > 1 else range(nimg)
            for i in iterator:
                tic = time.time()
                maski, flowi, stylei = self.eval(
                    x[i], batch_size=batch_size,
                    channels=channels[i] if channels is not None and
                    ((len(channels) == len(x) and
                      (isinstance(channels[i], list) or
                       isinstance(channels[i], np.ndarray)) and len(channels[i]) == 2))
                    else channels, channel_axis=channel_axis, z_axis=z_axis,
                    normalize=normalize, invert=invert,
                    rescale=rescale[i] if isinstance(rescale, list) or
                    isinstance(rescale, np.ndarray) else rescale,
                    diameter=diameter[i] if isinstance(diameter, list) or
                    isinstance(diameter, np.ndarray) else diameter, do_3D=do_3D,
                    anisotropy=anisotropy, augment=augment, tile=tile,
                    tile_overlap=tile_overlap, bsize=bsize, resample=resample,
                    interp=interp, flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold, compute_masks=compute_masks,
                    min_size=min_size, stitch_threshold=stitch_threshold,
                    progress=progress, niter=niter)
                masks.append(maski)
                flows.append(flowi)
                styles.append(stylei)
                self.timing.append(time.time() - tic)
            return masks, flows, styles

        else:
            # reshape image
            x = transforms.convert_image(x, channels, channel_axis=channel_axis,
                                         z_axis=z_axis, do_3D=(do_3D or
                                                               stitch_threshold > 0),
                                         nchan=self.nchan)
            #增加一个维度 [1,2,512]
            if x.ndim < 4:
                x = x[np.newaxis, ...]

            self.batch_size = batch_size
            # print(diameter)
            if diameter is not None and diameter > 0:

                rescale = self.diam_mean / diameter

            elif rescale is None:
                diameter = self.diam_labels
                rescale = self.diam_mean / diameter


            masks, styles, dP, cellprob, p = self._run_cp(
                x, compute_masks=compute_masks, normalize=normalize, invert=invert,
                rescale=rescale, resample=resample, augment=augment, tile=tile,
                tile_overlap=tile_overlap, bsize=bsize, flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold, interp=interp, min_size=min_size,
                do_3D=do_3D, anisotropy=anisotropy, niter=niter,
                stitch_threshold=stitch_threshold)

            flows = [plot.dx_to_circ(dP), dP, cellprob, p]
            return masks, flows, styles

    def _run_cp(self, x, compute_masks=True, normalize=True, invert=False, niter=None,
                rescale=1.0, resample=True, augment=False, tile=True, tile_overlap=0.1,
                cellprob_threshold=0.0, bsize=224, flow_threshold=0.4, min_size=15,
                interp=True, anisotropy=1.0, do_3D=False, stitch_threshold=0.0):

        if isinstance(normalize, dict):
            normalize_params = {**normalize_default, **normalize}
        elif not isinstance(normalize, bool):
            raise ValueError("normalize parameter must be a bool or a dict")
        else:
            normalize_params = normalize_default
            normalize_params["normalize"] = normalize
        normalize_params["invert"] = invert

        tic = time.time()
        shape = x.shape #[1,2,512]
        nimg = shape[0]

        bd, tr = None, None

        # pre-normalize if 3D stack for stitching or do_3D
        do_normalization = True if normalize_params["normalize"] else False
        if nimg > 1 and do_normalization and (stitch_threshold or do_3D):
            # must normalize in 3D if do_3D is True
            normalize_params["norm3D"] = True if do_3D else normalize_params["norm3D"]
            x = np.asarray(x)
            x = transforms.normalize_img(x, **normalize_params)
            # do not normalize again
            do_normalization = False

        if do_3D:
            img = np.asarray(x)
            yf, styles = run_3D(self.net, img, rsz=rescale, anisotropy=anisotropy,
                                augment=augment, tile=tile, tile_overlap=tile_overlap)
            cellprob = yf[0][-1] + yf[1][-1] + yf[2][-1]
            dP = np.stack(
                (yf[1][0] + yf[2][0], yf[0][0] + yf[2][1], yf[0][1] + yf[1][1]),
                axis=0)  # (dZ, dY, dX)
            del yf
        else:
            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            iterator = trange(nimg, file=tqdm_out,
                              mininterval=30) if nimg > 1 else range(nimg)

            styles = np.zeros((nimg, self.nbase[-1]), np.float32)
            #-----------------------------------查看style的shape------------------------
            # print('styles.shape={}'.format(styles.shape))   styles.shape=(1, 256)
            if resample:
                dP = np.zeros((2, nimg, shape[1], shape[2]), np.float32)
                cellprob = np.zeros((nimg, shape[1], shape[2]), np.float32)
            else:
                #不进行重采样
                # print('不进行重采样')
                dP = np.zeros(
                    (2, nimg, int(shape[1] * rescale), int(shape[2] * rescale)),
                    np.float32)
                cellprob = np.zeros(
                    (nimg, int(shape[1] * rescale), int(shape[2] * rescale)),
                    np.float32)
            for i in iterator:
                # print('走这条路')
                img = np.asarray(x[i])
                if do_normalization:
                    img = transforms.normalize_img(img, **normalize_params)
                if rescale != 1.0:
                    img = transforms.resize_image(img, rsz=rescale)
                yf, style = run_net(self.net, img, bsize=bsize, augment=augment,
                                        tile=tile, tile_overlap=tile_overlap)
                if resample:
                    yf = transforms.resize_image(yf, shape[1], shape[2])

                cellprob[i] = yf[:, :, 2]
                dP[:, i] = yf[:, :, :2].transpose((2, 0, 1))
                if self.nclasses == 4:
                    if i == 0:
                        bd = np.zeros_like(cellprob)
                    bd[i] = yf[:, :, 3]
                styles[i][:len(style)] = style
            del yf, style
        styles = styles.squeeze()
        # print('styles.shape={}'.format(styles.shape))
        net_time = time.time() - tic
        if nimg > 1:
            models_logger.info("network run in %2.2fs" % (net_time))

        if compute_masks:
            tic = time.time()
            niter0 = 200 if (do_3D and not resample) else (1 / rescale * 200)
            niter = niter0 if niter is None or niter==0 else niter
            if do_3D:
                masks, p = dynamics.resize_and_compute_masks(
                    dP, cellprob, niter=niter, cellprob_threshold=cellprob_threshold,
                    flow_threshold=flow_threshold, interp=interp, do_3D=do_3D,
                    min_size=min_size, resize=None,
                    device=self.device if self.gpu else None)
            else:
                masks, p = [], []
                resize = [shape[1], shape[2]] if (not resample and
                                                  rescale != 1) else None
                iterator = trange(nimg, file=tqdm_out,
                                  mininterval=30) if nimg > 1 else range(nimg)
                for i in iterator:
                    outputs = dynamics.resize_and_compute_masks(
                        dP[:, i],
                        cellprob[i],
                        niter=niter,
                        cellprob_threshold=cellprob_threshold,
                        flow_threshold=flow_threshold,
                        interp=interp,
                        resize=resize,
                        min_size=min_size if stitch_threshold == 0 or nimg == 1 else
                        -1,  # turn off for 3D stitching
                        device=self.device if self.gpu else None)
                    masks.append(outputs[0])
                    p.append(outputs[1])

                masks = np.array(masks)
                p = np.array(p)
                if stitch_threshold > 0 and nimg > 1:
                    models_logger.info(
                        f"stitching {nimg} planes using stitch_threshold={stitch_threshold:0.3f} to make 3D masks"
                    )
                    masks = utils.stitch3D(masks, stitch_threshold=stitch_threshold)
                    masks = utils.fill_holes_and_remove_small_masks(
                        masks, min_size=min_size)
                elif nimg > 1:
                    models_logger.warning("3D stack used, but stitch_threshold=0 and do_3D=False, so masks are made per plane only")


            flow_time = time.time() - tic
            if nimg > 1:
                models_logger.info("masks created in %2.2fs" % (flow_time))
            masks, dP, cellprob, p = masks.squeeze(), dP.squeeze(), cellprob.squeeze(
            ), p.squeeze()

        else:
            masks, p = np.zeros(0), np.zeros(0)  #pass back zeros if not compute_masks
        return masks, styles, dP, cellprob, p

class SizeModel():


    def __init__(self, cp_model, device=None, pretrained_size=None, **kwargs):
        super(SizeModel, self).__init__(**kwargs)


        self.pretrained_size = pretrained_size
        self.cp = cp_model
        self.device = self.cp.device
        self.diam_mean = self.cp.diam_mean# diam_mean：30
        if pretrained_size is not None:
            self.params = np.load(self.pretrained_size, allow_pickle=True).item()
            self.diam_mean = self.params["diam_mean"]
        if not hasattr(self.cp, "pretrained_model"):
            error_message = "no pretrained cellpose_ model specified, cannot compute size"
            models_logger.critical(error_message)
            raise ValueError(error_message)

    def eval(self, x, channels=None, channel_axis=None, normalize=True, invert=False,
             augment=False, tile=True, batch_size=8, progress=None):

        if isinstance(x, list):

            self.timing = []
            diams, diams_style = [], []
            nimg = len(x)
            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            iterator = trange(nimg, file=tqdm_out,
                              mininterval=30) if nimg > 1 else range(nimg)
            for i in iterator:
                tic = time.time()
                diam, diam_style = self.eval(
                    x[i], channels=channels[i] if
                    (channels is not None and len(channels) == len(x) and
                     (isinstance(channels[i], list) or
                      isinstance(channels[i], np.ndarray)) and
                     len(channels[i]) == 2) else channels, channel_axis=channel_axis,
                    normalize=normalize, invert=invert, augment=augment, tile=tile,
                    batch_size=batch_size, progress=progress)
                diams.append(diam)
                diams_style.append(diam_style)
                self.timing.append(time.time() - tic)

            return diams, diams_style

        if x.squeeze().ndim > 3:
            models_logger.warning("image is not 2D cannot compute diameter")
            return self.diam_mean, self.diam_mean

        styles = self.cp.eval(x, channels=channels, channel_axis=channel_axis,
                              normalize=normalize, invert=invert, augment=augment,
                              tile=tile, batch_size=batch_size, resample=False,
                              compute_masks=False)[-1]
        #diam_mean=30.0
        diam_style = self._size_estimation(np.array(styles))
        diam_style = self.diam_mean if (diam_style == 0 or
                                        np.isnan(diam_style)) else diam_style

        masks = self.cp.eval(
            x, compute_masks=True, channels=channels, channel_axis=channel_axis,
            normalize=normalize, invert=invert, augment=augment, tile=tile,
            batch_size=batch_size, resample=False,
            rescale=self.diam_mean / diam_style if self.diam_mean > 0 else 1,
            diameter=None, interp=False)[0]
        #Calculate masks equivalent diameter
        diam = utils.diameters(masks)[0]
        diam = self.diam_mean if (diam == 0 or np.isnan(diam)) else diam
        return diam, diam_style

    def _size_estimation(self, style):

        szest = np.exp(self.params["A"] @ (style - self.params["smean"]).T +
                       np.log(self.diam_mean) + self.params["ymean"])
        szest = np.maximum(5., szest)
        return szest
