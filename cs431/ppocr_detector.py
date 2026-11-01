"""PP-OCRv5_server_det exactly as PaddleOCR v3.3.3 tools/eval.py runs it, for one image at a time.

Mirrors tools/eval.py + tools/program.eval (and tools/infer_det.py for unlabeled images):
  config      = tools.program.load_config(<official yml>)       (no parameter overrides)
  transforms  = config Eval transforms minus label ops; KeepKeys -> [image, shape]
                (DecodeImage BGR, DetResizeForTest default = limit_side_len 736 / min,
                 NormalizeImage ImageNet, ToCHWImage)
  model       = build_model(Architecture), official .pdparams loaded like ppocr load_model
                (dtype cast) but STRICT: any missing key / shape mismatch is an error
  inference   = model.eval(); paddle.no_grad(); batch 1; FP32 (no AMP)
  postprocess = build_post_process(config PostProcess) = DBPostProcess(0.3, 0.6, 1000, 1.5)
DBPostProcess.__call__ returns only points; scores come from the SAME boxes_from_bitmap call
it uses internally, and the boxes must be identical, otherwise the image fails.
Nothing in the PaddleOCR clone is modified.
"""
import sys
from pathlib import Path

import numpy as np


class OfficialDetector:
    def __init__(self, repo, config_relpath, weights, device='gpu:0'):
        repo = Path(repo).resolve()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        import paddle
        from tools.program import load_config
        from ppocr.data import create_operators, transform
        from ppocr.modeling.architectures import build_model
        from ppocr.postprocess import build_post_process

        self.paddle, self._transform = paddle, transform
        paddle.set_device(device)
        config = load_config(str(repo/config_relpath))
        if config['Architecture']['model_type'] != 'det' or config['PostProcess']['name'] != 'DBPostProcess':
            raise RuntimeError('Unexpected config: expected DB text detector')
        global_config = config['Global']
        self.post = build_post_process(config['PostProcess'], global_config)
        if getattr(self.post, 'box_type', 'quad') != 'quad':
            raise RuntimeError('Expected quadrilateral DB post-processing')
        transforms = []
        for op in config['Eval']['dataset']['transforms']:
            name = next(iter(op))
            if 'Label' in name:
                continue
            if name == 'KeepKeys':
                op[name]['keep_keys'] = ['image', 'shape']
            transforms.append(op)
        self.ops = create_operators(transforms, global_config)
        self.model = build_model(config['Architecture'])
        self.load_report = self._load_strict(weights)
        self.model.eval()
        self.transforms = [next(iter(op)) for op in transforms]
        self.resize = {k: getattr(self.ops[self.transforms.index('DetResizeForTest')], k, None)
                       for k in ('limit_side_len', 'limit_type', 'resize_type')}

    def _load_strict(self, weights):
        params = self.paddle.load(str(weights))
        state = self.model.state_dict()
        missing = sorted(k for k in state if k not in params)
        shape = sorted(k for k in state if k in params and list(state[k].shape) != list(params[k].shape))
        if missing or shape:
            raise RuntimeError(f'Checkpoint does not match architecture: missing={missing[:5]} shape={shape[:5]}')
        loaded = {k: (params[k].astype(v.dtype) if params[k].dtype != v.dtype else params[k]) for k, v in state.items()}
        self.model.set_state_dict(loaded)
        return dict(model_keys=len(state), checkpoint_keys=len(params), missing_keys=0, shape_mismatch=0,
                    unused_checkpoint_keys=len(set(params) - set(state)))

    def predict(self, path):
        with open(path, 'rb') as f:
            data = {'image': f.read()}
        batch = self._transform(data, self.ops)
        if batch is None:
            raise RuntimeError('Official transforms rejected image: ' + str(path))
        image, shape = batch
        shape_list = np.expand_dims(shape, axis=0)
        with self.paddle.no_grad():
            preds = self.model(self.paddle.to_tensor(np.expand_dims(image, axis=0)))
        official = self.post(preds, shape_list)[0]['points']
        pred = preds['maps'].numpy()[:, 0, :, :][0]
        mask = pred > self.post.thresh
        if self.post.dilation_kernel is not None:
            import cv2
            mask = cv2.dilate(np.array(mask).astype(np.uint8), self.post.dilation_kernel)
        src_h, src_w = shape_list[0][0], shape_list[0][1]
        boxes, scores = self.post.boxes_from_bitmap(pred, mask, src_w, src_h)
        if not np.array_equal(np.asarray(boxes), np.asarray(official)):
            raise RuntimeError('Score recovery diverged from official DBPostProcess output')
        polygons = [[float(v) for v in np.asarray(box, dtype=float).reshape(-1)] for box in official]
        return polygons, [float(s) for s in scores]
