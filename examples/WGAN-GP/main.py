import os, glob
import numpy as np
from absl import flags, app
import PIL

import elegy
from model import WGAN_GP



FLAGS = flags.FLAGS

flags.DEFINE_string(
    "output_dir",
    default=None,
    help="Directory to save model checkpoints and example generated images",
)
flags.DEFINE_integer("epochs", default=100, help="Number of epochs to train")
flags.DEFINE_integer("batch_size", default=64, help="Input batch size")

flags.DEFINE_string(
    "dataset", default=None, help="Search path to the dataset images e.g: path/to/*.png"
)

flags.mark_flag_as_required("dataset")
flags.mark_flag_as_required("output_dir")




class Dataset(elegy.data.Dataset):
    def __init__(self, path):
        self.files = glob.glob(os.path.expanduser(path))
        if len(self.files)==0:
            raise RuntimeError(f'Could not find any files in path "{path}"')
        print(f'Found {len(self.files)} files')
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        f          = self.files[i]
        img        = np.array(PIL.Image.open(f).resize((64,64))) / np.float32(255)
        img        = np.fliplr(img) if np.random.random()<0.5 else img
        return img


class SaveImagesCallback(elegy.callbacks.Callback):
    def __init__(self, model, path):
        self.model = model
        self.path = path
    def on_epoch_end(self, epoch, *args, **kwargs):
        x = self.model.predict(np.random.normal(size=[8,128]))
        x = np.concatenate(list(x*255), axis=1).astype(np.uint8)
        img = PIL.Image.fromarray(x)
        img.save(os.path.join(self.path, f'epoch-{epoch:04d}.png'))




def main(argv):
    assert (
        len(argv) == 1
    ), "Please specify arguments via flags. Use --help for instructions"

    assert not os.path.exists(
        FLAGS.output_dir
    ), "Output directory already exists. Delete manually or specify a new one."
    os.makedirs(FLAGS.output_dir)



    ds = Dataset(FLAGS.dataset)
    loader = elegy.data.DataLoader(ds, batch_size=FLAGS.batch_size, n_workers=os.cpu_count(), worker_type='process')

    wgan = WGAN_GP()
    wgan.states = wgan.init( np.zeros([8,128]) )

    wgan.fit(
        loader,
        epochs=FLAGS.epochs,
        verbose=4,
        callbacks=[SaveImagesCallback(wgan, FLAGS.output_dir)]
    )

if __name__ == "__main__":
    app.run(main)
