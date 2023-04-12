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

# SBATCH -J fudge_classif_2                        # Job name
# SBATCH -o fudge_classif_2.o%j                    # Name of stdout output file (%j corresponds to the job id)
# SBATCH -e fudge_classif_2.e%j                    # Name of stderr error file (%j corresponds to the job id)
# SBATCH -p gpu-a100                   # Queue (partition) name
# SBATCH -N 1                            # Total # of nodes (must be 1 for serial)
# SBATCH -n 32				# Number of cores
# SBATCH -t 24:00:00                     # Run time (hh:mm:ss)
# SBATCH --mail-user=devyani.maladkar@utexas.edu
# SBATCH --mail-type=all                 # Send email at begin and end of job (can assign begin or end as well)
# SBATCH -A CCR23005         # Allocation name (req'd if you have more than 1)

# Other commands must follow all #SBATCH directives...

module load python3
source $WORK/nlp_proj_venv/bin/activate
# conda activate NLP_Proj
cd $WORK/NLP_PROJECT/
# Launch serial code...

#python3 run.py --do_train --task nli --dataset snli --output_dir ./trained_model/
python3 controlled_text_generation.py --input_file dataset/e2e_data/target_attribute.json --gen_model_path trained_models_text_generation/gpt2_e2e_5.pt --per_control 20 --lambda_condition 2 --sample_stratergy sample
python3 controlled_text_generation.py --input_file dataset/e2e_data/target_attribute.json --gen_model_path trained_models_text_generation/gpt2_e2e_5.pt --per_control 20 --lambda_condition 4 --sample_stratergy sample
python3 controlled_text_generation.py --input_file dataset/e2e_data/target_attribute.json --gen_model_path trained_models_text_generation/gpt2_e2e_5.pt --per_control 20 --lambda_condition 6 --sample_stratergy sample
python3 controlled_text_generation.py --input_file dataset/e2e_data/target_attribute.json --gen_model_path trained_models_text_generation/gpt2_e2e_5.pt --per_control 20 --lambda_condition 8 --sample_stratergy sample




# ---------------------------------------------------