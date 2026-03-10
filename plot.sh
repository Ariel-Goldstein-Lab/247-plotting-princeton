#!/bin/bash
##SBATCH --constraint=gpu80
##SBATCH --nodes=1
##SBATCH --cpus-per-task=4
#SBATCH -o './logs/%A_%x.log'
#SBATCH --error='./logs/%A_%x_error.log'
#SBATCH --mail-user=timna.kleinman+DSBATCH@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL

if [[ "$HOSTNAME" == *"tiger"* ]]
then
    echo "It's tiger"
    module load anaconda
    conda activate 247-main
elif [[ "$HOSTNAME" == *"della"* ]]
then
    echo "It's della-gpu"
    module purge
    module load anaconda3/2021.11
    conda activate jup
else
    module purge
    module load anacondapy
    source activate srm
fi

echo 'Requester:' $USER 'Node:' $HOSTNAME
echo "$@"
echo 'Start time:' `date`
start=$(date +%s)

if [[ -v SLURM_ARRAY_TASK_ID ]]
then
    echo "Running in array mode with task ID: $SLURM_ARRAY_TASK_ID"
    python
else
#    echo "Running in normal mode"
    python -u "$@"
fi

end=$(date +%s)
echo 'End time:' `date`
# Calculate elapsed time in seconds
elapsed_seconds=$(($end-$start))

# Convert to minutes and seconds
minutes=$((elapsed_seconds / 60))
seconds=$((elapsed_seconds % 60))

# Display the result
echo "Elapsed Time: $minutes:$seconds minutes and seconds"
