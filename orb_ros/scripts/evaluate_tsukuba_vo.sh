#! /bin/bash

NUM_TEST=5
DATA_PATH=/PATH/TO/TSUKUBA/DATASET
SEQUENCES=('daylight' 'fluorescent' 'lamps' 'flashlight')

trap ctrl_c INT

function ctrl_c() {
        kill %1
        exit
}

function srcc() {
        prefix=''
        if [ "$#" -eq 1 ]; then
                prefix="_$1"
                print $prefix
        fi

        cwd=$(pwd)
        cdir=$(pwd)
        while [ $cdir != $HOME ]; do
                dir_to_check="$cdir/devel$prefix"
                if [ -d $dir_to_check ]; then
                        source "$dir_to_check/setup.bash"
                        echo "source $dir_to_check/setup.bash"
                        break
                fi
                cd ..
                cdir=$(pwd)
        done
        if [ $cdir = $HOME ]; then
                echo "failed to find a catkin... "
        fi

        cd $cwd
}

srcc

# rm -rf $(rospack find orb_ros)/expr/
mkdir -p $(rospack find orb_ros)/expr/

roscore &

sleep 2

rosparam set /slam/dataset tsukuba
rosparam set /slam/model_path $(rospack find orb_ros)/data/models/superpoint.pt
rosparam set /slam/output_path $(rospack find orb_ros)/expr/
rosparam set /slam/visualize False
rosparam set /slam/online True
rosparam set /slam/verbose False

rosparam load $(rospack find orb_ros)/cfg/tsukuba.yaml /slam
expr_path=$(rospack find orb_ros)/expr/
rosparam set /slam/data_path "$DATA_PATH/"

for seq in ${SEQUENCES[*]}; do
        rosparam set /slam/sequence $seq

        mkdir -p $expr_path/$seq

        for ((i = 1; i <= $NUM_TEST; i++)); do
                echo
                echo '#####################################################################'
                echo "$seq, $i/$NUM_TEST"

                rosrun orb_slam2 mono_node __name:=slam -alsologtostderr -colorlogtostderr --minloglevel=2

                mv $expr_path/traj.txt $expr_path/$seq/$i.txt

                sleep 5

        done

done

python3 scripts/evo_tsukuba.py --pkg_path $(rospack find orb_ros)

kill %1
