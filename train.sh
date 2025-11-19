#!/bin/bash
MODEL_NAME=$(python3 -c "import sys; sys.path.append('..'); from config import MODEL_NAME; print(MODEL_NAME)")
name="test_train_datalab"
outdir="outputs"
n_gpu=1

export DATA="/home/peforcioli/assignment2-2025-penebarthanouk/data"

    echo "Launching test for $name "
    
    sbatch <<EOT
#!/bin/bash
#SBATCH -p mesonet 
#SBATCH -N 1
#SBATCH -c 28
#SBATCH --gres=gpu:${n_gpu}
#SBATCH --time=00:20:00
#SBATCH --mem=256G
#SBATCH --account=m25146        # Your project account 
#SBATCH --job-name=gan_train      # Job name
#SBATCH --output=${outdir}/%x_%j.out  # Standard output and error log
#SBATCH --error=${outdir}/%x_%j.err  # Error log
source /home/peforcioli/assignment2-2025-penebarthanouk/venv/bin/activate
# Run your training script with arguments from config.HP_CONFIG
python /home/peforcioli/assignment2-2025-penebarthanouk/train.py \
    --model-name "${MODEL_NAME}" \
    --latent-dim 100 \
    --image-dim 784 \
    --K 15 \
    --sigma 0.2 \
    --c 0.3 \
    --which-loss PR \
    --which-div Chi2 \
    --lambda 0.05 \
    --epochs 200 \
    --lr 0.00001 \
    --batch-size 64 \
    --gpus ${n_gpu} \
    --logs True \
    --num-samples-evaluate 10000 \
    --data "${DATA}"
EOT

