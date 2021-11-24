from megflow import Envelope, register
import megengine
import importlib
import json
import sys
import megengine.functional as F
import megengine.hub as hub
import megengine.module as M
import numpy as np
import megengine.data.transform as T

logging = megengine.logger.get_logger()


@register(inputs=['inp'], outputs=['out'])
class Hello:
    def __init__(self, name, args):
        self.name = name
        self.args = args

        logging.info("load model %s", args['arch'])

        logging.info("load model_file %s", args['model_file'])
        module = import_from_file(args['model_file'])
        self.model = module.__dict__[args['arch']]()

        if self.model is not None:
            logging.info("load checkpoint %s", args['weight_file'])
            checkpoint = megengine.load(args['weight_file'])
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            self.model.load_state_dict(state_dict)

    def exec(self):
        # print(self.name)
        envelope = self.inp.recv()
        if envelope is None:
            return

        img_data = envelope.msg['data']
        logging.info("img_data len %d", len(img_data))
        transform = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.Normalize(
                    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]
                ),  # BGR
                T.ToMode("CHW"),
            ]
        )

        def infer_func(processed_img):
            self.model.eval()
            logits = self.model(processed_img)
            probs = F.softmax(logits)
            return probs

        processed_img = transform.apply(img_data)[np.newaxis, :]
        processed_img = megengine.tensor(processed_img, dtype="float32")
        probs = infer_func(processed_img)
        top_probs, classes = F.topk(probs, k=2, descending=True)
        imagenet_class_index = {0: 'apple', 1: 'orange'}
        result = {"apple": 0, "orange": 0}

        for rank, (prob, classid) in enumerate(
                zip(top_probs.numpy().reshape(-1), classes.numpy().reshape(-1))
        ):
            print(
                "{}: class = {:20s} with probability = {:4.1f} %".format(
                    rank, imagenet_class_index[classid], 100 * prob
                )
            )
            result[imagenet_class_index[classid]] = 100 * prob
        print("result", result)
        self.out.send(envelope.repack(json.dumps(result)))


def import_from_file(cfg_file):
    module_spec = importlib.util.spec_from_file_location("config", "./config/model.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module
