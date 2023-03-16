HOME_DIR="/home/loopai"

ffmpeg \
    -r 30 -i ${HOME_DIR}/Projects/binarization/artifacts/vmaf/original.mp4 \
    -r 30 -i ${HOME_DIR}/Projects/binarization/artifacts/vmaf/generated.mp4 \
    -ss 00:00:00 -to 00:00:03 \
    -lavfi "[0:v]setpts=PTS-STARTPTS[reference]; \
            [1:v]scale=-1:1080:flags=bicubic,setpts=PTS-STARTPTS[distorted]; \
            [distorted][reference]libvmaf=log_fmt=json:log_path=/dev/stdout:model_path=${HOME_DIR}/ffmpeg_sources/vmaf-2.1.1/model/vmaf_v0.6.1.json:n_threads=4" \
    -f null -

