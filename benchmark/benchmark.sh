#! bash
rm -rf build

mkdir build

folder_path="./cuda" # 你的脚本文件夹路径

cd "$folder_path"

# choose correct version of nvcc and CUDA_VISIBLE_DEVICES for you devices!
for file in *; do
    if [ -f "$file" ]; then
        nvcc -arch=sm_80 "$file" -o ../build/"$file".out
    fi
done

cd ../build

for file in *; do
    if [ -f "$file" ]; then
        echo " "
        CUDA_VISIBLE_DEVICES=1 ./"$file"
    fi
done