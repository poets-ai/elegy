set -e

for file in $(find examples -name "*.py" | grep -v utils.py | grep -v imagenet) ; do
    cmd="python $file --epochs 2 --steps-per-epoch 1 --batch-size 3"
    echo RUNNING: $cmd
    DISPLAY="" $cmd > /dev/null
done