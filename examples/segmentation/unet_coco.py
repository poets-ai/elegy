import os

# PYTHONPATH="../.." python unet_coco.py --unet_backbone=ResNet50  --dataset=/root/DATA --image_size=512 --batch_size=8 --base_lr=0.005 --crop_min_size=0.8 --dtype=float16 --loss_scale=1024. --output_dir=outputs/unet_r50_crop0.8_lr0.005_batch8
# specify the cuda location for XLA when working with conda environments
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=" + os.sep.join(
    os.__file__.split(os.sep)[:-3]
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from absl import flags, app
import jax, jax.numpy as jnp
import numpy as np
import elegy
import optax
import cloudpickle
import PIL.Image
from pycocotools.coco import COCO


print("JAX version:", jax.__version__)
print("Elegy version:", elegy.__version__)

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "unet_backbone",
    default=None,
    enum_values=[
        "ResNet18",
        "ResNet50",
    ],
    help="Which backbone to use as encoder for the U-Net",
)

flags.DEFINE_string(
    "output_dir",
    default=None,
    help="Directory to save model checkpoints and tensorboard log data",
)
flags.DEFINE_integer("epochs", default=20, help="Number of epochs to train")
flags.DEFINE_integer("batch_size", default=16, help="Input batch size")
flags.DEFINE_integer("image_size", default=256, help="Image size in pixels")
flags.DEFINE_string("dataset", default=None, help="Path to the COCO dataset")
flags.DEFINE_enum(
    "dtype",
    default="float32",
    enum_values=["float16", "float32"],
    help="Mixed precision or normal mode",
)
flags.DEFINE_float("base_lr", default=0.005, help="SGD optimizer base learning rate")
flags.DEFINE_float("momentum", default=0.9, help="SGD optimizer momentum")
flags.DEFINE_float("L2_reg", default=1e-4, help="L2 weight regularization")
flags.DEFINE_float(
    "loss_scale",
    default=1.0,
    help="Loss scale for numerical stability when dtype=float16",
)
flags.DEFINE_float(
    "crop_min_size", default=0.8, help="Minimum size of the random crop during training"
)

flags.mark_flag_as_required("unet_backbone")
flags.mark_flag_as_required("output_dir")
flags.mark_flag_as_required("dataset")


def generate_cropbox(imagesize, minsize=0.9, augment=False):
    if not augment:
        return np.concatenate([(0, 0), imagesize])
    boxsize = np.random.uniform(minsize, 1.0, size=2) * imagesize
    yx = np.random.uniform(imagesize - boxsize, size=2)
    return np.concatenate([yx, boxsize])


class Dataset(elegy.data.Dataset):
    def __init__(
        self, coco, coco_path, classes, image_size, crop_min_size=0.9, augment=False
    ):
        self.coco_path = coco_path
        self.coco = coco
        self.classes = classes
        self.img_ids = [coco.getImgIds(catIds=cat) for cat in classes]
        self.img_ids = list(set([x for X in self.img_ids for x in X]))
        self.augment = augment
        self.image_size = (image_size, image_size)
        self.crop_min_size = crop_min_size

    def __len__(self):
        # return min(1000, len(self.img_ids))
        return len(self.img_ids)

    def __getitem__(self, i):
        img_id = self.img_ids[i]
        imgname = self.coco.loadImgs(img_id)[0]["file_name"]
        imgpath = os.path.join(self.coco_path, imgname)
        image = PIL.Image.open(imgpath).convert("RGB")
        cropbox = generate_cropbox(
            image.size, minsize=self.crop_min_size, augment=self.augment
        )
        image = image.crop(cropbox).resize(self.image_size)
        image = np.asarray(image) / np.float32(255)

        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.classes)
        anns = self.coco.loadAnns(ann_ids)
        masks = np.asarray([self.coco.annToMask(a) * a["category_id"] for a in anns])
        mask = (
            np.max(masks, axis=0)
            if len(masks)
            else np.zeros(image.shape[:2], dtype=np.uint8)
        )

        mask = np.asarray(
            PIL.Image.fromarray(np.asarray(mask))
            .crop(cropbox)
            .resize(self.image_size, resample=PIL.Image.NEAREST)
        )
        if self.augment and np.random.random() < 0.5:
            image = image[:, ::-1]
            mask = mask[:, ::-1]

        sample_weight = 1
        if self.augment:
            # set sample weight to zero if number of non-background pixels is small
            # effectively ignore this image
            valid_px = (mask > 0).sum()
            sample_weight = 0 if valid_px / mask.size < 0.02 else 1

        return image, mask, np.array([sample_weight]).reshape(1, 1)


def main(argv):
    assert (
        len(argv) == 1
    ), "Please specify arguments via flags. Use --help for instructions"

    assert not os.path.exists(
        FLAGS.output_dir
    ), "Output directory already exists. Delete manually or specify a new one."
    os.makedirs(FLAGS.output_dir)

    coco_train = COCO(
        os.path.join(FLAGS.dataset, "annotations/instances_train2017.json")
    )
    coco_valid = COCO(os.path.join(FLAGS.dataset, "annotations/instances_val2017.json"))
    categories = coco_train.loadCats(coco_train.getCatIds())
    names = [cat["name"] for cat in categories]
    print("All COCO categories: \n{}\n".format(", ".join(sorted(names))))

    CLASSNAMES = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "boat",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "bottle",
        "chair",
        "couch",
        "potted plant",
        "dining table",
        "tv",
    ]
    CLASSES = coco_train.getCatIds(catNms=CLASSNAMES)

    ds_train = Dataset(
        coco_train,
        os.path.join(FLAGS.dataset, "train2017"),
        classes=CLASSES,
        image_size=FLAGS.image_size,
        crop_min_size=FLAGS.crop_min_size,
        augment=True,
    )
    ds_valid = Dataset(
        coco_valid,
        os.path.join(FLAGS.dataset, "val2017"),
        classes=CLASSES,
        image_size=FLAGS.image_size,
        augment=False,
    )

    print("Training Images: ", len(ds_train))
    print("Validation Images: ", len(ds_valid))

    loader_train = elegy.data.DataLoader(
        ds_train,
        batch_size=FLAGS.batch_size,
        n_workers=os.cpu_count(),
        shuffle=True,
        worker_type="process",
    )
    loader_valid = elegy.data.DataLoader(
        ds_valid,
        batch_size=FLAGS.batch_size,
        n_workers=os.cpu_count(),
        shuffle=False,
        worker_type="process",
    )

    print("Training Steps:", len(loader_train))
    print("Validation Steps:", len(loader_valid))

    if FLAGS.unet_backbone == "ResNet18":
        unet = elegy.nets.segmentation.UNet_R18(
            max(CLASSES) + 1, backbone_weights="imagenet", dtype=FLAGS.dtype
        )
    else:
        unet = elegy.nets.segmentation.UNet_R50(
            max(CLASSES) + 1, backbone_weights="imagenet", dtype=FLAGS.dtype
        )
    model = elegy.Model(
        unet,
        loss=elegy.losses.SparseCategoricalCrossentropy(
            from_logits=True, weight=FLAGS.loss_scale
        ),
        optimizer=elegy.Optimizer(
            optax.scale(1 / FLAGS.loss_scale),
            optax.additive_weight_decay(FLAGS.L2_reg),
            optax.sgd(1.0, momentum=FLAGS.momentum),  # learning rate set below
            lr_schedule=lambda step, epoch: optax.cosine_decay_schedule(
                1.0, decay_steps=FLAGS.epochs * len(loader_train)
            )(step)
            * FLAGS.base_lr,
        ),
        metrics=[
            elegy.metrics.MeanIoU(classes=jnp.array([0] + CLASSES)),
            elegy.metrics.SparseCategoricalAccuracy(),
        ],
    )
    model.summary(next(iter(loader_train))[0], depth=1)

    model.fit(
        loader_train,
        validation_data=loader_valid,
        epochs=FLAGS.epochs,
        verbose=4,
        callbacks=[
            elegy.callbacks.TensorBoard(FLAGS.output_dir),
            elegy.callbacks.LambdaCallback(
                on_epoch_end=lambda *args: open(
                    os.path.join(FLAGS.output_dir, "module.pkl"), "wb"
                ).write(cloudpickle.dumps(model.module))
            ),
        ],
    )


if __name__ == "__main__":
    app.run(main)
