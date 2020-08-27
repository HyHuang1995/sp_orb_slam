#! /bin/bash

NUM_TEST=5
DATA_PATH=/PATH/TO/EUROC/DATASET
SEQUENCES=('MH_01_easy' 'MH_02_easy' 'MH_03_medium' 'MH_04_difficult' 'MH_05_difficult' 'V1_01_easy' 'V1_02_medium' 'V1_03_difficult' 'V2_01_easy' 'V2_02_medium' 'V2_03_difficult')

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
mkdir -p $(rospack find orb_ros)/expr/kf

roscore &

sleep 2

rosparam set /slam/dataset euroc
rosparam set /slam/output_path $(rospack find orb_ros)/expr/
rosparam set /slam/model_path $(rospack find orb_ros)/data/models/superpoint.pt
rosparam set /slam/visualize False
rosparam set /slam/online True
rosparam set /slam/verbose False

rosparam load $(rospack find orb_ros)/cfg/euroc_mono.yaml /slam
expr_path=$(rospack find orb_ros)/expr/

for seq in ${SEQUENCES[*]}; do
        rosparam set /slam/data_path "$DATA_PATH/$seq/mav0/"

        mkdir -p $expr_path/$seq
        rm -rf $expr_path/kf/$seq
        mkdir -p $expr_path/kf/$seq

        for ((i = 1; i <= $NUM_TEST; i++)); do
                echo
                echo '#####################################################################'
                echo "$seq, $i/$NUM_TEST"

                rosrun orb_slam2 mono_node __name:=slam -alsologtostderr -colorlogtostderr --minloglevel=2

                mv $expr_path/traj.txt $expr_path/$seq/$i.txt
                mv $expr_path/kf.txt $expr_path/kf/$seq/$i.txt

                sleep 10
        done
done

python3 scripts/evo_euroc.py --pkg_path $(rospack find orb_ros)

kill %1
