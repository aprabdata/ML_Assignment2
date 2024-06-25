import mlrose_hiive
import numpy as np
import logging
import networkx as nx
import matplotlib.pyplot as plt
import string
import random
import os


from ast import literal_eval
#import chess

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from mlrose_hiive import QueensGenerator, MaxKColorGenerator
from mlrose_hiive.generators.four_peaks_generator import FourPeaksGenerator
from mlrose_hiive import SARunner, GARunner, RHCRunner

def apply_ga(problem, problem_name):
    fig_feval, ax_feval = plt.subplots(1, constrained_layout=True)
    fig_fitness, ax_fitness = plt.subplots(1, constrained_layout=True)
    fig_time, ax_time = plt.subplots(1, constrained_layout=True)
    fig_fev_time, ax_fev_time = plt.subplots(1, constrained_layout=True)

    fig, ax = plt.subplots(2,2, constrained_layout=True)

    random_seeds = [25, 29, 43, 96, 23, 32, 51, 40, 10, 75, 12, 15, 18, 48, 37, 72, 55, 94, 88, 97]
    fevals = np.zeros(shape=(101, len(random_seeds)))
    fitness = np.zeros(shape=(101, len(random_seeds)))
    times = np.zeros(shape=(101, len(random_seeds)))
    colors = ["b", "g", "r", "c", "m", "y", "k", 'burlywood', "chartreuse"]
    color_i = 0
    for population_size in [10, 50, 100]:
        for mutation_rate in [0.1, 0.2, 0.3]:
            for random_seed, i in zip(random_seeds, range(101)):
                ga = GARunner(problem=problem,
                              experiment_name=problem_name,
                              #output_directory=os.getcwd(),
                              output_directory=None,
                              # note: specify an output directory to have results saved to disk
                              seed=random_seed,
                              iteration_list=10 * np.arange(101),
                              population_sizes=[population_size],
                              mutation_rates=[mutation_rate])

                # the two data frames will contain the results
                df_run_stats, df_run_curves = ga.run()
                #a = df_run_stats['FEvals'].to_numpy()
                fevals[:, i] = df_run_stats['FEvals'].to_numpy()
                fitness[:, i] = df_run_stats['Fitness'].to_numpy()
                times[:, i] = df_run_stats['Time'].to_numpy()

            fevals_mean = fevals.mean(axis=1)
            fevals_std = fevals.std(axis=1)

            fitness_mean = fitness.mean(axis=1)
            fitness_std = fitness.std(axis=1)

            times_mean = times.mean(axis=1)
            times_std = times.std(axis=1)

            max_fitness = fitness_mean.max()
            max_feval = fevals_mean.max()

            print("GA_Pop_" + str(population_size) + "_Mut_" + str(mutation_rate)+ " " + problem_name + " max_fitness: " + str(max_fitness))
            print("GA_Pop_" + str(population_size) + "_Mut_"+ str(mutation_rate) + " " + problem_name + " max_fevals: " + str(max_feval))

            # plt.fill_between(10*np.arange(101), fevals_mean - fevals_std, fevals_mean + fevals_std, alpha=0.1, color=colors[color_i])
            ax_feval.plot(10 * np.arange(101), fevals_mean, label=f"Population: {population_size}, Mutation: {mutation_rate}", color=colors[color_i])
            ax_fitness.plot(10 * np.arange(101), fitness_mean, label=f"Population: {population_size}, Mutation: {mutation_rate}", color=colors[color_i])
            ax_time.plot(10 * np.arange(101), times_mean, label=f"Population: {population_size}, Mutation: {mutation_rate}", color=colors[color_i])
            ax_fev_time.plot(fevals_mean, times_mean, label=f"Population: {population_size}, Mutation: {mutation_rate}", color=colors[color_i])

            ax[0,0].fill_between(10*np.arange(101), fevals_mean - fevals_std, fevals_mean + fevals_std, alpha=0.1, color=colors[color_i])
            ax[0,0].plot(10 * np.arange(101), fevals_mean, label=f"Population: {population_size}, Mutation: {mutation_rate}", color=colors[color_i])

            ax[0,1].fill_between(10*np.arange(101), fitness_mean - fitness_std, fitness_mean + fitness_std, alpha=0.1, color=colors[color_i])
            ax[0,1].plot(10 * np.arange(101), fitness_mean, label=f"Population: {population_size}, Mutation: {mutation_rate}", color=colors[color_i])

            ax[1,0].fill_between(10*np.arange(101), times_mean - times_std, times_mean + times_std, alpha=0.1, color=colors[color_i])
            ax[1,0].plot(10 * np.arange(101), times_mean, label=f"Population: {population_size}, Mutation: {mutation_rate}", color=colors[color_i])

            ax[1,1].plot(fevals_mean, times_mean, label=f"Population: {population_size}, Mutation: {mutation_rate}", color=colors[color_i])

            color_i = color_i + 1
    """
    ax_feval.set_title(f"{problem_name}")
    ax_feval.set_xlabel("Iterations")
    ax_feval.set_ylabel("Fevals")
    ax_feval.grid(visible=True, alpha=0.1)
    ax_feval.legend(frameon=False)
    fig_feval.savefig('GA-Feval '+ problem_name + '.png')
    """

    ax[0,0].set_title("Fevals Vs Iteration")
    ax[0,0].set_xlabel("Iterations")
    ax[0,0].set_ylabel("Fevals")
    ax[0,0].grid(visible=True, alpha=0.1)
    #ax[0,0].legend(frameon=False)

    """
    ax_fitness.set_title(f"{problem_name}")
    ax_fitness.set_xlabel("Iterations")
    ax_fitness.set_ylabel("Fitness")
    ax_fitness.grid(visible=True, alpha=0.1)
    ax_fitness.legend(frameon=False)
    fig_fitness.savefig('GA-Fitness ' + problem_name + '.png')
    """

    ax[0,1].set_title("Fitness Vs Iteration")
    ax[0,1].set_xlabel("Iterations")
    ax[0,1].set_ylabel("Fitness")
    ax[0,1].grid(visible=True, alpha=0.1)
    #ax[0,1].legend(frameon=False)

    """
    ax_time.set_title(f"{problem_name}")
    ax_time.set_xlabel("Iterations")
    ax_time.set_ylabel("Time")
    ax_time.grid(visible=True, alpha=0.1)
    ax_time.legend(frameon=False)
    fig_time.savefig('GA-Time ' + problem_name + '.png')
    """

    ax[1, 0].set_title("Time Vs Iteration")
    ax[1, 0].set_xlabel("Iterations")
    ax[1, 0].set_ylabel("Time")
    ax[1, 0].grid(visible=True, alpha=0.1)
    #ax[1, 0].legend(frameon=False)

    """
    ax_fev_time.set_title(f"{problem_name}")
    ax_fev_time.set_xlabel("Fevals")
    ax_fev_time.set_ylabel("Time")
    ax_fev_time.grid(visible=True, alpha=0.1)
    ax_fev_time.legend(frameon=False)
    fig_fev_time.savefig('GA-Feval-Time ' + problem_name + '.png')
    """

    ax[1,1].set_title("Time Vs Feval")
    ax[1,1].set_xlabel("Fevals")
    ax[1,1].set_ylabel("Time")
    ax[1,1].grid(visible=True, alpha=0.1)
    #ax[1,1].legend(frameon=False)

    # fig.legend(ax.get_lines(), ncol=2, loc="lower center")
    handles, labels = ax[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, -0.07), frameon=False, ncol=3)
    # fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False)
    fig.suptitle(f"GA - {problem_name}")
    # fig.savefig('GA ' + problem_name + '.png')
    fig.savefig('GA ' + problem_name + '.png', bbox_inches='tight')

    #plt.xticks(100 * np.arange(11))
    #plt.grid(visible=True, alpha=0.1)
    #plt.legend(frameon=False)
    #plt.show()
    #plt.savefig('GA '+ problem_name + '.png')
    plt.clf()

def apply_sa(problem, problem_name):
    fig_feval, ax_feval = plt.subplots(1, constrained_layout=True)
    fig_fitness, ax_fitness = plt.subplots(1, constrained_layout=True)
    fig_time, ax_time = plt.subplots(1, constrained_layout=True)
    fig_fev_time, ax_fev_time = plt.subplots(1, constrained_layout=True)

    fig, ax = plt.subplots(2, 2, constrained_layout=True)

    random_seeds = [25, 29, 43, 96, 23, 32, 51, 40, 10, 75, 12, 15, 18, 48, 37, 72, 55, 94, 88, 97]
    fevals = np.zeros(shape=(101, len(random_seeds)))
    fitness = np.zeros(shape=(101, len(random_seeds)))
    times = np.zeros(shape=(101, len(random_seeds)))
    colors = ["b", "g", "r", "c", "m", "y", "k", 'burlywood', "chartreuse"]
    color_i = 0
    for temperature in [0.1, 1.0,  10, 100, 1000, 10000, 100000, 1000000, 10000000]:
        for random_seed, i in zip(random_seeds, range(101)):
            sa = SARunner(problem=problem,
                          experiment_name=problem_name,
                          output_directory=None,  # note: specify an output directory to have results saved to disk
                          seed=random_seed,
                          iteration_list=10 * np.arange(101),
                          max_attempts=500,
                          temperature_list=[temperature],
                          decay_list=[mlrose_hiive.GeomDecay])

            # the two data frames will contain the results
            df_run_stats, df_run_curves = sa.run()
            fevals[:, i] = df_run_stats['FEvals'].to_numpy()
            fitness[:, i] = df_run_stats['Fitness'].to_numpy()
            times[:, i] = df_run_stats['Time'].to_numpy()

        fevals_mean = fevals.mean(axis=1)
        fevals_std = fevals.std(axis=1)

        fitness_mean = fitness.mean(axis=1)
        fitness_std = fitness.std(axis=1)

        times_mean = times.mean(axis=1)
        times_std = times.std(axis=1)

        max_fitness = fitness_mean.max()
        max_feval = fevals_mean.max()

        print("SA_Temp_"+str(temperature)+ " " + problem_name + " max_fitness: " +str(max_fitness))
        print("SA_Temp_" + str(temperature) + " " + problem_name + " max_fevals: " + str(max_feval))


        #plt.fill_between(10*np.arange(101), fevals_mean - fevals_std, fevals_mean + fevals_std, alpha=0.1, color=colors[color_i])
        ax_feval.plot(10 * np.arange(101), fevals_mean, label=f"Temperature: {temperature}", color=colors[color_i])
        ax_fitness.plot(10 * np.arange(101), fitness_mean, label=f"Temperature: {temperature}", color=colors[color_i])
        ax_time.plot(10 * np.arange(101), times_mean, label=f"Temperature: {temperature}", color=colors[color_i])
        ax_fev_time.plot(fevals_mean, times_mean, label=f"Temperature: {temperature}", color=colors[color_i])

        ax[0, 0].fill_between(10*np.arange(101), fevals_mean - fevals_std, fevals_mean + fevals_std, alpha=0.1, color=colors[color_i])
        ax[0, 0].plot(10 * np.arange(101), fevals_mean,label=f"Temperature: {temperature}", color=colors[color_i])

        ax[0, 1].fill_between(10*np.arange(101), fitness_mean - fitness_std, fitness_mean + fitness_std, alpha=0.1, color=colors[color_i])
        ax[0, 1].plot(10 * np.arange(101), fitness_mean,label=f"Temperature: {temperature}", color=colors[color_i])

        ax[1, 0].fill_between(10*np.arange(101), times_mean - times_std, times_mean + times_std, alpha=0.1, color=colors[color_i])
        ax[1, 0].plot(10 * np.arange(101), times_mean,label=f"Temperature: {temperature}", color=colors[color_i])

        ax[1, 1].plot(fevals_mean, times_mean, label=f"Temperature: {temperature}", color=colors[color_i])

        color_i = color_i + 1

    """
    ax_feval.set_title(f"{problem_name}")
    ax_feval.set_xlabel("Iterations")
    ax_feval.set_ylabel("Fevals")
    ax_feval.grid(visible=True, alpha=0.1)
    ax_feval.legend(frameon=False)
    fig_feval.savefig('SA-Feval ' + problem_name + '.png')
    """

    ax[0, 0].set_title("Fevals Vs Iteration")
    ax[0, 0].set_xlabel("Iterations")
    ax[0, 0].set_ylabel("Fevals")
    ax[0, 0].grid(visible=True, alpha=0.1)
    # ax[0,0].legend(frameon=False)

    """
    ax_fitness.set_title(f"{problem_name}")
    ax_fitness.set_xlabel("Iterations")
    ax_fitness.set_ylabel("Fitness")
    ax_fitness.grid(visible=True, alpha=0.1)
    ax_fitness.legend(frameon=False)
    fig_fitness.savefig('SA-Fitness ' + problem_name + '.png')
    """

    ax[0, 1].set_title("Fitness Vs Iteration")
    ax[0, 1].set_xlabel("Iterations")
    ax[0, 1].set_ylabel("Fitness")
    ax[0, 1].grid(visible=True, alpha=0.1)
    # ax[0,1].legend(frameon=False)

    """
    ax_time.set_title(f"{problem_name}")
    ax_time.set_xlabel("Iterations")
    ax_time.set_ylabel("Time")
    ax_time.grid(visible=True, alpha=0.1)
    ax_time.legend(frameon=False)
    fig_time.savefig('SA-Time ' + problem_name + '.png')
    """

    ax[1, 0].set_title("Time Vs Iteration")
    ax[1, 0].set_xlabel("Iterations")
    ax[1, 0].set_ylabel("Time")
    ax[1, 0].grid(visible=True, alpha=0.1)
    # ax[1, 0].legend(frameon=False)

    """
    ax_fev_time.set_title(f"{problem_name}")
    ax_fev_time.set_xlabel("Fevals")
    ax_fev_time.set_ylabel("Time")
    ax_fev_time.grid(visible=True, alpha=0.1)
    ax_fev_time.legend(frameon=False)
    fig_fev_time.savefig('SA-Feval-Time ' + problem_name + '.png')
    """

    ax[1, 1].set_title("Time Vs Feval")
    ax[1, 1].set_xlabel("Fevals")
    ax[1, 1].set_ylabel("Time")
    ax[1, 1].grid(visible=True, alpha=0.1)
    # ax[1,1].legend(frameon=False)

    # fig.legend(ax.get_lines(), ncol=2, loc="lower center")
    handles, labels = ax[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, -0.07), frameon=False, ncol=3)
    #fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False)
    fig.suptitle(f"SA - {problem_name}")
    #fig.savefig('SA ' + problem_name + '.png')
    fig.savefig('SA ' + problem_name + '.png', bbox_inches='tight')

    # plt.xticks(100 * np.arange(11))
    # plt.grid(visible=True, alpha=0.1)
    # plt.legend(frameon=False)
    # plt.show()
    # plt.savefig('GA '+ problem_name + '.png')
    plt.clf()

def apply_rhc(problem, problem_name):
    fig_feval, ax_feval = plt.subplots(1, constrained_layout=True)
    fig_fitness, ax_fitness = plt.subplots(1, constrained_layout=True)
    fig_time, ax_time = plt.subplots(1, constrained_layout=True)
    fig_fev_time, ax_fev_time = plt.subplots(1, constrained_layout=True)

    fig, ax = plt.subplots(2, 2, constrained_layout=True)

    random_seeds = [25, 29, 43, 96, 23, 32, 51, 40, 10, 75, 12, 15, 18, 48, 37, 72, 55, 94, 88, 97]
    fevals = np.zeros(shape=(101, len(random_seeds)))
    fitness = np.zeros(shape=(101, len(random_seeds)))
    times = np.zeros(shape=(101, len(random_seeds)))
    colors = ["b", "g", "r", "c", "m", "y", "k", 'burlywood', "chartreuse"]
    color_i = 0

    for random_seed, i in zip(random_seeds, range(101)):
        rhc = RHCRunner(problem=problem,
                      experiment_name=problem_name,
                      output_directory=None,  # note: specify an output directory to have results saved to disk
                      seed=random_seed,
                      iteration_list=10 * np.arange(101),
                      max_attempts=500,
                      restart_list = [0])

        # the two data frames will contain the results
        df_run_stats, df_run_curves = rhc.run()
        fevals[:, i] = df_run_stats['FEvals'].to_numpy()
        fitness[:, i] = df_run_stats['Fitness'].to_numpy()
        times[:, i] = df_run_stats['Time'].to_numpy()

    fevals_mean = fevals.mean(axis=1)
    fevals_std = fevals.std(axis=1)

    fitness_mean = fitness.mean(axis=1)
    fitness_std = fitness.std(axis=1)

    times_mean = times.mean(axis=1)
    times_std = times.std(axis=1)

    max_fitness = fitness_mean.max()
    max_feval = fevals_mean.max()

    print("RHC " + problem_name + " max_fitness: " + str(max_fitness))
    print("RHC " + problem_name + " max_fevals: " + str(max_feval))

    ax[0, 0].fill_between(10 * np.arange(101), fevals_mean - fevals_std, fevals_mean + fevals_std, alpha=0.1, color=colors[color_i])
    ax[0, 0].plot(10 * np.arange(101), fevals_mean, color=colors[color_i])

    ax[0, 1].fill_between(10 * np.arange(101), fitness_mean - fitness_std, fitness_mean + fitness_std, alpha=0.1,color=colors[color_i])
    ax[0, 1].plot(10 * np.arange(101), fitness_mean,  color=colors[color_i])

    ax[1, 0].fill_between(10 * np.arange(101), times_mean - times_std, times_mean + times_std, alpha=0.1, color=colors[color_i])
    ax[1, 0].plot(10 * np.arange(101), times_mean,  color=colors[color_i])

    #ax[1, 1].fill_between(10 * np.arange(101), fevals_mean - fevals_std, fevals_mean + fevals_std, alpha=0.1,color=colors[color_i])
    ax[1, 1].plot(fevals_mean, times_mean,  color=colors[color_i])

    """"
    # plt.fill_between(10*np.arange(101), fevals_mean - fevals_std, fevals_mean + fevals_std, alpha=0.1, color=colors[color_i])
    ax_feval.plot(10 * np.arange(101), fevals_mean, label=f"Number of Random Restarts: {restart}", color=colors[color_i])
    ax_fitness.plot(10 * np.arange(101), fitness_mean, label=f"Number of Random Restarts: {restart}", color=colors[color_i])
    ax_time.plot(10 * np.arange(101), times_mean, label=f"Number of Random Restarts: {restart}", color=colors[color_i])
    ax_fev_time.plot(fevals_mean, times_mean, label=f"Number of Random Restarts: {restart}", color=colors[color_i])

    ax[0, 0].plot(10 * np.arange(101), fevals_mean,label=f"Number of Random Restarts: {restart}", color=colors[color_i])
    ax[0, 1].plot(10 * np.arange(101), fitness_mean,label=f"Number of Random Restarts: {restart}", color=colors[color_i])
    ax[1, 0].plot(10 * np.arange(101), times_mean,label=f"Number of Random Restarts: {restart}", color=colors[color_i])
    ax[1, 1].plot(fevals_mean, times_mean, label=f"Number of Random Restarts: {restart}", color=colors[color_i])
    color_i = color_i + 1
    """
    """
    ax_feval.set_title(f"{problem_name}")
    ax_feval.set_xlabel("Iterations")
    ax_feval.set_ylabel("Fevals")
    ax_feval.grid(visible=True, alpha=0.1)
    ax_feval.legend(frameon=False)
    fig_feval.savefig('RHC-Feval ' + problem_name + '.png')
    """
    ax[0, 0].set_title("Fevals Vs Iteration")
    ax[0, 0].set_xlabel("Iterations")
    ax[0, 0].set_ylabel("Fevals")
    ax[0, 0].grid(visible=True, alpha=0.1)
    # ax[0,0].legend(frameon=False)
    """
    ax_fitness.set_title(f"{problem_name}")
    ax_fitness.set_xlabel("Iterations")
    ax_fitness.set_ylabel("Fitness")
    ax_fitness.grid(visible=True, alpha=0.1)
    ax_fitness.legend(frameon=False)
    fig_fitness.savefig('RHC-Fitness ' + problem_name + '.png')
    """
    ax[0, 1].set_title("Fitness Vs Iteration")
    ax[0, 1].set_xlabel("Iterations")
    ax[0, 1].set_ylabel("Fitness")
    ax[0, 1].grid(visible=True, alpha=0.1)
    # ax[0,1].legend(frameon=False)
    """
    ax_time.set_title(f"{problem_name}")
    ax_time.set_xlabel("Iterations")
    ax_time.set_ylabel("Time")
    ax_time.grid(visible=True, alpha=0.1)
    ax_time.legend(frameon=False)
    fig_time.savefig('RHC-Time ' + problem_name + '.png')
    """
    ax[1, 0].set_title("Time Vs Iteration")
    ax[1, 0].set_xlabel("Iterations")
    ax[1, 0].set_ylabel("Time")
    ax[1, 0].grid(visible=True, alpha=0.1)
    # ax[1, 0].legend(frameon=False)
    """
    ax_fev_time.set_title(f"{problem_name}")
    ax_fev_time.set_xlabel("Fevals")
    ax_fev_time.set_ylabel("Time")
    ax_fev_time.grid(visible=True, alpha=0.1)
    ax_fev_time.legend(frameon=False)
    fig_fev_time.savefig('RHC-Feval-Time ' + problem_name + '.png')
    """
    ax[1, 1].set_title("Time Vs Feval")
    ax[1, 1].set_xlabel("Fevals")
    ax[1, 1].set_ylabel("Time")
    ax[1, 1].grid(visible=True, alpha=0.1)
    # ax[1,1].legend(frameon=False)

    # fig.legend(ax.get_lines(), ncol=2, loc="lower center")
    #handles, labels = ax[1, 1].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, -0.07), frameon=False, ncol=3)
    #fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False)
    fig.suptitle(f"RHC - {problem_name}")
    fig.savefig('RHC ' + problem_name + '.png')

    # plt.xticks(100 * np.arange(11))
    # plt.grid(visible=True, alpha=0.1)
    # plt.legend(frameon=False)
    # plt.show()
    # plt.savefig('GA '+ problem_name + '.png')
    plt.clf()


if __name__ == "__main__":
    # Generate a new Max K problem using a fixed seed.
    maxK_problem_small = MaxKColorGenerator().generate(seed=123456, number_of_nodes=10, max_connections_per_node=3, max_colors=3, maximize=True)
    maxK_problem_large = MaxKColorGenerator().generate(seed=123456, number_of_nodes=15, max_connections_per_node=4, max_colors=3, maximize=True)

    #apply_ga(maxK_problem_small, "KColor - 10 Nodes 3 Max Connection 3 Colors")
    #apply_ga(maxK_problem_large, "KColor - 15 Nodes 4 Max Connection 3 Colors")

    apply_sa(maxK_problem_small, "KColor - 10 Nodes 3 Max Connection 3 Colors")
    apply_sa(maxK_problem_large, "KColor - 15 Nodes 4 Max Connection 3 Colors")

    #apply_rhc(maxK_problem_small, "KColor - 10 Nodes 3 Max Connection 3 Colors")
    #apply_rhc(maxK_problem_large, "KColor - 15 Nodes 4 Max Connection 3 Colors")

    # Generate a new 4 peaks problem using a fixed seed.
    fourPeak_problem_small = FourPeaksGenerator.generate(seed=123456, size=20, t_pct=0.1)
    fourPeak_problem_large = FourPeaksGenerator.generate(seed=123456, size=40, t_pct=0.1)

    #apply_ga(fourPeak_problem_small, "Four Peaks - Size 20")
    #apply_ga(fourPeak_problem_large, "Four Peaks - Size 40")

    apply_sa(fourPeak_problem_small, "Four Peaks - Size 20")
    apply_sa(fourPeak_problem_large, "Four Peaks - Size 40")

    #apply_rhc(fourPeak_problem_small, "Four Peaks - Size 20")
    #apply_rhc(fourPeak_problem_large, "Four Peaks - Size 40")


    #print("hello world")

"""
Reference:
mlrose__hiive: https://github.com/hiive/mlrose/

pyperch: https://github.com/jlm429/pyperch/
"""
