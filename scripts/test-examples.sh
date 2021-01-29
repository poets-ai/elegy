set -e

for file in $(ls examples) ; do
    extension=${file##*.}
    if [ "$extension" == "py" ]; then
        echo RUNNING: extension/$file

        DISPLAY="" python examples/$file --epochs 1 > /dev/null
    fi
done