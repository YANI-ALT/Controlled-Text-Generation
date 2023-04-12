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

python3 classifier_report.py --classifier_folder classifier_models/models_for_name --class_type name --dataset_path dataset/e2e_data
python3 classifier_report.py --classifier_folder classifier_models/models_for_area --class_type area --dataset_path dataset/e2e_data
python3 classifier_report.py --classifier_folder classifier_models/models_for_Type --class_type Type --dataset_path dataset/e2e_data
python3 classifier_report.py --classifier_folder classifier_models/models_for_price --class_type price --dataset_path dataset/e2e_data
python3 classifier_report.py --classifier_folder classifier_models/models_for_near --class_type near --dataset_path dataset/e2e_data

python3 classifier_report.py --classifier_folder classifier_models/models_for_area --class_type area --dataset_path dataset/e2e_data
python3 classifier_report.py --classifier_folder classifier_models/models_for_food --class_type food --dataset_path dataset/e2e_data
python3 classifier_report.py --classifier_folder classifier_models/models_for_family_friendly --class_type family_friendly --dataset_path dataset/e2e_data



# ---------------------------------------------------