# ==============================================================================
# ZDT1
# ==============================================================================
python run_with_random_and_weighted_sum_init.py --problem_name ZDT1 --dim 2
python run_with_random_and_weighted_sum_init.py --problem_name ZDT1 --dim 4
python run_with_random_and_weighted_sum_init.py --problem_name ZDT1 --dim 8
python run_with_random_and_weighted_sum_init.py --problem_name ZDT1 --dim 16
python run_with_random_and_weighted_sum_init.py --problem_name ZDT1 --dim 30 --save_gif
python run_with_random_and_weighted_sum_init.py --problem_name ZDT1 --dim 32

# ==============================================================================
# ZDT4
# ==============================================================================
python run_with_random_and_weighted_sum_init.py --problem_name ZDT4 --dim 2
python run_with_random_and_weighted_sum_init.py --problem_name ZDT4 --dim 4
python run_with_random_and_weighted_sum_init.py --problem_name ZDT4 --dim 8
python run_with_random_and_weighted_sum_init.py --problem_name ZDT4 --dim 10 --save_gif
python run_with_random_and_weighted_sum_init.py --problem_name ZDT4 --dim 16


# ==============================================================================
# FON
# ==============================================================================
python run_with_random_and_weighted_sum_init.py --problem_name FON --dim 2
python run_with_random_and_weighted_sum_init.py --problem_name FON --dim 3 --save_gif
python run_with_random_and_weighted_sum_init.py --problem_name FON --dim 4
python run_with_random_and_weighted_sum_init.py --problem_name FON --dim 8
python run_with_random_and_weighted_sum_init.py --problem_name FON --dim 16
python run_with_random_and_weighted_sum_init.py --problem_name FON --dim 32
python run_with_random_and_weighted_sum_init.py --problem_name FON --dim 64

# Plot convergence plots
python plot/plot_convergence.py