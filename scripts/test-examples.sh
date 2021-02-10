set -e

for file in $(ls examples/*.py ) ; do
    cmd="python $file --epochs 2 --steps-per-epoch 1 --batch-size 3"
    echo RUNNING: $cmd
    DISPLAY="" $cmd > /dev/null
done

#WGAN example
tmpdir=`mktemp -d`; rm -r $tmpdir
cmd="python examples/WGAN-GP/main.py --epochs=2 --dataset=examples/WGAN-GP/images/*.png --output_dir=$tmpdir"
echo RUNNING: $cmd
DISPLAY="" $cmd > /dev/null