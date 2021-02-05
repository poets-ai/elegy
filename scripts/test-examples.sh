set -e

for file in $(find examples -name "*.py" | grep -v utils.py | grep -v imagenet) ; do
    echo RUNNING: ${file}
    DISPLAY="" python $file --epochs 2 --steps-per-epoch 1  > /dev/null
done