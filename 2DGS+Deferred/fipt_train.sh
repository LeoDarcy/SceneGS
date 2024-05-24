# python train.py -s  /cpfs01/user/baijiayang/workspace/workspace/data/hypersim/ai_035_010 --eval -m output/hypersim_ai_035_010 -w --brdf_dim 0 --sh_degree -1 --lambda_predicted_normal 2e-1 --brdf_env 512 ；
# python train.py -s  /nas/shared/pjlab_lingjun_landmarks/renkerui/data/eyefultower/ai_001_001 --eval -m output/hypersim_ai_001_001 -w --brdf_dim 0 --sh_degree -1 --lambda_predicted_normal 2e-1 --brdf_env 512;

# python render.py -m output/hypersim_ai_001_001 --brdf_dim 0 --sh_degree -1 --brdf_mode envmap --brdf_env 512;

# 定义一个字符串数组
declare -a arr=("classroom" "conferenceroom")
datadir="/nas/shared/pjlab_lingjun_landmarks/baijiayang/workspace/data/fipt/real_scene"
# 遍历数组
for i in "${arr[@]}"
do
    # python train.py -s "${datadir}/${i}/" --eval -m "outputs/FIPT/${i}" ;

    python render.py -m "outputs/FIPT/${i}" ;
done