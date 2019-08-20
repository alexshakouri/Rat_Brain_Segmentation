#! /bin/bash -l
#SBATCH --partition=qigpu
#SBATCH --nodelist=g5-15
#SBATCH --job-name=Full_Model_test
#SBATCH --array=1
#SBATCH --error=job.%A_%a.err
#SBATCH --output=job.%A_%a.out
#SBATCH --mem=10000
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

module load bio/1.0
source /home/ashakour/MRI_segmentation/keras/bin/activate
module load cudnn/7.0
module load cuda/9.0
module load opencv/3.4.1

srun python /home/ashakour/MRI_segmentation/Rat_Brain_Sementation/train.py
