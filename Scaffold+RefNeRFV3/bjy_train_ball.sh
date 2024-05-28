function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))  
}

port=$(rand 10000 30000)
# data_root="/cpfs01/user/baijiayang/workspace/workspace/data/refnerf/shiny_blender"
data_root="/cpfs01/user/baijiayang/workspace/hdd_workspace/workspace/data/refnerf/shiny_blender"
lod=0
iterations=30_000
begin_normal_iteration=0
warmup="False"
# while [[ "$#" -gt 0 ]]; do
#     case $1 in
#         -l|--logdir) logdir="$2"; shift ;;
#         -d|--data) data="$2"; shift ;;
#         --lod) lod="$2"; shift ;;
#         --gpu) gpu="$2"; shift ;;
#         --warmup) warmup="$2"; shift ;;
#         --voxel_size) vsize="$2"; shift ;;
#         --update_init_factor) update_init_factor="$2"; shift ;;
#         --appearance_dim) appearance_dim="$2"; shift ;;
#         --ratio) ratio="$2"; shift ;;
#         *) echo "Unknown parameter passed: $1"; exit 1 ;;
#     esac
#     shift
# done

gpu=-1
vsize=0.001
update_init_factor=4
appearance_dim=0
ratio=1
gt_normals=0
gt_albedo=0
time=$(date "+%Y-%m-%d_%H:%M:%S")

# load_checkpoint="/cpfs01/user/baijiayang/workspace/CodeVersion2/HpersimCode/SecondVersion/Scaffold_Diffuse_Env/outputs/ball/Scaffold_GS_GTNormal/2024-05-14_14:25:15/point_cloud/iteration_30000"


# 定义一个字符串数组
#  "ball" "car" "helmet" "coffee"
# echo "python train.py --eval  --n_offsets 1 --gt_normals ${gt_normals} --gt_albedo ${gt_albedo} -s ${data_root}/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --warmup --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time"
declare -a arr=("myball" "toaster" "car" "helmet" "coffee")
# 遍历数组
logdir="Scaffold_RegV3_n10_scale1_NormReg_defaultDensity"
for data in "${arr[@]}"
do
    time=$(date "+%Y-%m-%d_%H:%M:%S")
    if [ "$warmup" = "True" ]; then
        python train.py --eval  --n_offsets 10 --gt_normals ${gt_normals} --gt_albedo ${gt_albedo} -s ${data_root}/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --warmup --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time

    else
    # echo "python train.py --eval --n_offsets 1 --gt_normals ${gt_normals} --gt_albedo ${gt_albedo} -s ${data_root}/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time"
    python train.py --eval --n_offsets 10 --gt_normals ${gt_normals} --gt_albedo ${gt_albedo} -s ${data_root}/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time
        
    fi
done

# --eval --n_offsets 1 --gt_normals 0 --gt_albedo 0 -s /cpfs01/user/baijiayang/workspace/hdd_workspace/workspace/data/refnerf/shiny_blender/myball --lod 0 --gpu -1 --voxel_size 0.001 --update_init_factor 4 --appearance_dim 0 --ratio 1 --warmup --iterations 30_000 --port 14407 -m outputs/2024-05-24_21:22:12