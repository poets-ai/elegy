set -e

for file in $(ls examples) ; do
    extension=${file##*.}
    if [ "$extension" == "py" ]; then
        echo RUNNING: examples/$file

        DISPLAY="" python examples/$file --epochs 2 --steps-per-epoch 1 > /dev/null
    fi
done