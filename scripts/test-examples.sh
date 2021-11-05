
set -e



#----------------------------------------------------------------
# test docs/getting-started
#----------------------------------------------------------------
# create tmp_dir
tmp_dir=$(mktemp -d -t XXXXXXXXXX)

# low-level-api
file="docs/getting-started/low-level-api.ipynb"
echo RUNNING: $file
jupyter nbconvert --log-level "ERROR" --to python --output $tmp_dir/result.py $file > /dev/null
sed -i "s/get_ipython/#get_ipython/" $tmp_dir/result.py
sed -i "s/epochs=100/epochs=2/" $tmp_dir/result.py
sed -i "s/steps_per_epoch=200/steps_per_epoch=2/" $tmp_dir/result.py
sed -i "s/batch_size=64/batch_size=4/" $tmp_dir/result.py
DISPLAY="" python $tmp_dir/result.py > /dev/null

# high-level-api
file="docs/getting-started/high-level-api.ipynb"
echo RUNNING: $file
jupyter nbconvert --log-level "ERROR" --to python --output $tmp_dir/result.py $file > /dev/null
sed -i "s/get_ipython/#get_ipython/" $tmp_dir/result.py
sed -i "s/epochs=100/epochs=2/" $tmp_dir/result.py
sed -i "s/steps_per_epoch=200/steps_per_epoch=2/" $tmp_dir/result.py
sed -i "s/batch_size=64/batch_size=4/" $tmp_dir/result.py
DISPLAY="" python $tmp_dir/result.py > /dev/null

# delete tmp_dir
rm -fr $tmp_dir

#----------------------------------------------------------------
# test examples
#----------------------------------------------------------------
for file in $(find examples -name '*.py' -not -path '*/imagenet/*' -not -path '*/WGAN-GP/*') ; do
    cmd="python $file --epochs 2 --steps-per-epoch 1 --batch-size 3"
    echo RUNNING: $cmd
    DISPLAY="" $cmd > /dev/null
done

#WGAN example
# tmpdir=`mktemp -d`; rm -r $tmpdir
# cmd="python examples/WGAN-GP/main.py --epochs=2 --dataset=examples/WGAN-GP/images/*.png --output_dir=$tmpdir"
# echo RUNNING: $cmd
# DISPLAY="" $cmd > /dev/null