#!/bin/bash

# run_server.sh

# 设置实验名称（建议包含关键参数，方便区分）
EXP_NAME="fusion_v2_server"

# 确保 logs 目录存在
mkdir -p logs

echo "提交训练任务: $EXP_NAME"

# 使用 nohup 后台运行
# -u: 禁用python输出缓存，确保日志实时写入
# 2>&1: 错误输出合并到标准输出
# --resume \  继续上次
nohup python -u scripts/04_train_server.py \
    --gpu_id 0 \
    --exp_name $EXP_NAME \
    --epochs 2000 \
    --batch_size 64 \
    --log_interval 10 \
    > logs/${EXP_NAME}_nohup.log 2>&1 &

echo "任务已提交！"
echo "查看实时日志: tail -f logs/${EXP_NAME}_nohup.log"