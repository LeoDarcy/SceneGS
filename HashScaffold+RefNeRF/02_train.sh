function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))  
}

port=$(rand 10000 30000)
data_root="/cpfs01/shared/pjlab-lingjun-landmarks/pjlab-lingjun-landmarks_hdd/jianglihan/Hypersim/portable_hard_drive/downloads/"
lod=0
iterations=30_000
begin_normal_iteration = 0
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
logdir="Scaffold_GS_FromScratch"
data="ai_001_002"
gpu=-1
vsize=0.001
update_init_factor=4
appearance_dim=0
ratio=1
load_checkpoint="/cpfs01/user/baijiayang/workspace/workspace/Code/HypersimDataset/Scaffold-GS_EnvTanh/outputs/ai_035_010/Scaffold_GS_Hypersim/2024-04-22_04:13:31/point_cloud/iteration_60000"
time=$(date "+%Y-%m-%d_%H:%M:%S")
echo "python train.py --eval -s ${data_root}/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time"
#--eval -s ./outputs/ai_001_001 --lod 0 --gpu -1 --voxel_size 0.001 --update_init_factor 4 --appearance_dim 0 --ratio 1 --iterations 30_000 --port 15650 -m outputs/ai_001_001/Scaffold_GS_Hypersim/2024-04-28

# --eval -s /cpfs01/shared/pjlab-lingjun-landmarks/pjlab-lingjun-landmarks_hdd/jianglihan/Hypersim/portable_hard_drive/downloads//ai_035_010 --lod 0 --gpu -1 --voxel_size 0.001 --update_init_factor 4 --appearance_dim 0 --ratio 1 --iterations 30_000 --port 15650 -m outputs/ai_035_010/Scaffold_GS_Hypersim/2024-04-19

if [ "$warmup" = "True" ]; then
    python train.py --eval -s ${data_root}/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --warmup --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time
else
    python train.py --eval -s ${data_root}/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time 
fi
# if [ "$warmup" = "True" ]; then
#     python train.py --eval --start_checkpoint ${load_checkpoint}/ -s ${data_root}/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --warmup --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time
# else
#     python train.py --eval --start_checkpoint ${load_checkpoint} -s ${data_root}/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time 
# fi

data="ai_001_003"
if [ "$warmup" = "True" ]; then
    python train.py --eval -s ${data_root}/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --warmup --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time
else
    python train.py --eval -s ${data_root}/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time 
fi

data="ai_001_004"
if [ "$warmup" = "True" ]; then
    python train.py --eval -s ${data_root}/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --warmup --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time
else
    python train.py --eval -s ${data_root}/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time 
fi

data="ai_001_005"
if [ "$warmup" = "True" ]; then
    python train.py --eval -s ${data_root}/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --warmup --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time
else
    python train.py --eval -s ${data_root}/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time 
fi
