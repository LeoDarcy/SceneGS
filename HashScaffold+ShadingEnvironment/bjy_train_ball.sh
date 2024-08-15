function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))  
}

port=$(rand 10000 30000)
# data_root="/cpfs01/user/baijiayang/workspace/workspace/data/refnerf/shiny_blender"
data_root="/cpfs01/user/baijiayang/workspace/Dataset/refnerf/shiny_blender"
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
logdir="Scaffold_GS_HashGridBaseline_GaussianShader"
data="coffee"
gpu=-1
vsize=0.001
update_init_factor=4
appearance_dim=0
ratio=1
gt_normals=0
gt_albedo=0
time=$(date "+%Y-%m-%d_%H:%M:%S")

# if [ "$warmup" = "True" ]; then
#     python train.py --eval --gt_normals ${gt_normals} --gt_albedo ${gt_albedo} -s ${data_root}/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --warmup --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time
# else
#     python train.py --eval --gt_normals ${gt_normals} --gt_albedo ${gt_albedo} -s ${data_root}/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time 
# fi

# sleep 20s
data="ball"
# load_checkpoint="/cpfs01/user/baijiayang/workspace/CodeVersion2/HpersimCode/SecondVersion/Scaffold_Diffuse_Env/outputs/ball/Scaffold_GS_GTNormal/2024-05-14_14:25:15/point_cloud/iteration_30000"

if [ "$warmup" = "True" ]; then
    python train.py --eval   --gt_normals ${gt_normals} --gt_albedo ${gt_albedo} -s ${data_root}/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --warmup --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time
else
    CUDA_LAUNCH_BLOCKING=1 python train.py --eval   --gt_normals ${gt_normals} --gt_albedo ${gt_albedo} -s ${data_root}/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time
fi


# data="helmet"
# sleep 20s
# if [ "$warmup" = "True" ]; then
#     python train.py --eval --gt_normals ${gt_normals} --gt_albedo ${gt_albedo} -s ${data_root}/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --warmup --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time
# else
#     python train.py --eval --gt_normals ${gt_normals} --gt_albedo ${gt_albedo} -s ${data_root}/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time
# fi


# data="teapot"
# sleep 20s
# if [ "$warmup" = "True" ]; then
#     python train.py --eval --gt_normals ${gt_normals} --gt_albedo ${gt_albedo} -s ${data_root}/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --warmup --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time
# else
#     python train.py --eval --gt_normals ${gt_normals} --gt_albedo ${gt_albedo} -s ${data_root}/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time
# fi

data="toaster"
sleep 10s
load_checkpoint="/cpfs01/user/baijiayang/workspace/CodeVersion2/HpersimCode/SecondVersion/Scaffold_Diffuse_Env/outputs/toaster/Scaffold_GS_GTNormal/2024-05-14_14:25:15/point_cloud/iteration_30000"

if [ "$warmup" = "True" ]; then
    python train.py --eval   --gt_normals ${gt_normals} --gt_albedo ${gt_albedo} -s ${data_root}/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --warmup --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time
else
    python train.py --eval   --gt_normals ${gt_normals} --gt_albedo ${gt_albedo} -s ${data_root}/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time
fi

data="car"
sleep 10s
load_checkpoint="/cpfs01/user/baijiayang/workspace/CodeVersion2/HpersimCode/SecondVersion/Scaffold_Diffuse_Env/outputs/car/Scaffold_GS_GTNormal/2024-05-14_14:25:15/point_cloud/iteration_30000"

if [ "$warmup" = "True" ]; then
    python train.py --eval   --gt_normals ${gt_normals} --gt_albedo ${gt_albedo} -s ${data_root}/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --warmup --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time
else
    python train.py --eval   --gt_normals ${gt_normals} --gt_albedo ${gt_albedo} -s ${data_root}/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time
fi