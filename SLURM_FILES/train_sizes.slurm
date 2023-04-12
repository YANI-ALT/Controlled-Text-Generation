#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Stampede2 SKX nodes
#
#   * Serial Job on SKX Normal Queue *
#
# Notes:
#
#   -- Copy/edit this script as desired.  Launch by executing
#      "sbatch sample.slurm" on a Stampede2 login node.
#
#   -- Serial codes run on a single node (upper case N = 1).
#        A serial code ignores the value of lower case n,
#        but slurm needs a plausible value to schedule the job.
#
#   -- For a good way to run multiple serial executables at the
#        same time, execute "module load launcher" followed
#        by "module help launcher".

#----------------------------------------------------

SBATCH -J myjob                        # Job name
SBATCH -o myjob.o%j                    # Name of stdout output file (%j corresponds to the job id)
SBATCH -e myjob.e%j                    # Name of stderr error file (%j corresponds to the job id)
SBATCH -p gpu-a100                   # Queue (partition) name
SBATCH -N 1                            # Total # of nodes (must be 1 for serial)
SBATCH -n 64				# Number of cores
SBATCH -t 24:00:00                     # Run time (hh:mm:ss)
SBATCH --mail-user=shrutiraghavan@utexas.edu
SBATCH --mail-type=all                 # Send email at begin and end of job (can assign begin or end as well)
SBATCH -A CCR23005         # Allocation name (req'd if you have more than 1)

# Other commands must follow all #SBATCH directives...

module load python3
source $WORK/project3/nlp3_venv/bin/activate
cd $WORK/project3/fp-dataset-artifacts

# Launch serial code...

#python3 run.py --do_train --task nli --dataset snli --output_dir ./trained_model/
python3 run.py --do_train --task nli --dataset dataset_sizes/new_train_10.json --resume_from_checkpoint ./trained_model/checkpoint-68500 --save_total_limit 5 --output_dir ./trained_model_new_10/
python3 run.py --do_train --task nli --dataset dataset_sizes/new_train_20.json --resume_from_checkpoint ./trained_model/checkpoint-68500 --save_total_limit 5 --output_dir ./trained_model_new_20/
python3 run.py --do_train --task nli --dataset dataset_sizes/new_train_30.json --resume_from_checkpoint ./trained_model/checkpoint-68500 --save_total_limit 5 --output_dir ./trained_model_new_30/
python3 run.py --do_train --task nli --dataset dataset_sizes/new_train_60.json --resume_from_checkpoint ./trained_model/checkpoint-68500 --save_total_limit 5 --output_dir ./trained_model_new_60/
python3 run.py --do_train --task nli --dataset dataset_sizes/new_train_90.json --resume_from_checkpoint ./trained_model/checkpoint-68500 --save_total_limit 5 --output_dir ./trained_model_new_90/
python3 run.py --do_train --task nli --dataset dataset_sizes/new_train_120.json --resume_from_checkpoint ./trained_model/checkpoint-68500 --save_total_limit 5 --output_dir ./trained_model_new_120/
python3 run.py --do_train --task nli --dataset dataset_sizes/new_train_150.json --resume_from_checkpoint ./trained_model/checkpoint-68500 --save_total_limit 5 --output_dir ./trained_model_new_150/
python3 run.py --do_train --task nli --dataset dataset_sizes/new_train_180.json --resume_from_checkpoint ./trained_model/checkpoint-68500 --save_total_limit 5 --output_dir ./trained_model_new_180/
python3 run.py --do_train --task nli --dataset dataset_sizes/new_train_210.json --resume_from_checkpoint ./trained_model/checkpoint-68500 --save_total_limit 5 --output_dir ./trained_model_new_210/





# ---------------------------------------------------