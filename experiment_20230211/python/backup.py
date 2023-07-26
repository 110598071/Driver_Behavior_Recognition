import Image_dataset
import pytorch_model
import torchsummary
import torch
import matplotlib.pyplot as plt
from PIL import Image

FC_NUM_EPOCHS = 10
FC_LEARNING_RATE = 0.0001

WHOLE_NUM_EPOCHS = 20
WHOLE_LEARNING_RATE = 0.000004

if __name__ == '__main__':
    # Hyperperameter
    # lstm_hidden_dim, lstm_layers, epochs, batch_size, lr_rate, optimizer_no

    # PSO init parameter
    # pop_size, dimension, upper_bounds, lower_bounds, model

    pop_size = 20
    upper = [256, 20, 400, 128, 100, 100]
    lower = [1, 1, 1, 2, 1, 1]
    solver = PSOSolver(pop_size, 6, upper, lower, pso_model)
    solver.initialize()

    for iteration in range(10):
        print(f"========iteration {iteration+1}========")
        solver.move_to_new_positions()
        solver.update_best_solution()

        for i,solution in enumerate(solver.solutions):
            
            print(f"solution {i+1}:")
            print(f"{solution} RMSE:{solver.current_solutions_value[i]}")
            print()
            
        print("global best solution:")
        print(f"{solver.global_best_solution}:{solver.global_best_objective_value}\n")