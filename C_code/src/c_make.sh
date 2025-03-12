#!/bin/bash
#SBATCH -J GLASS_setup                        ## 作业名 GLASS_setup
#SBATCH --cpus-per-task=1                     ## 每个任务需要的CPU核心数，默认为1
#SBATCH --ntasks-per-node=1                   ## 每节点任务数为1
#SBATCH -N 1                                  ## 指定节点数为1
#SBATCH -n 1                                  ## 指定总任务数为1
#SBATCH --qos=cpu96                           ## 指定qos为cpu96
#SBATCH -p cpu                                ## 指定队列名cpu

### 程序部分
date

# 激活conda环境
make clean

# 安装所需的Python包
make release

date