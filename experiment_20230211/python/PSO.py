import random

class PSOSolver():
    def __init__(self, pop_size, dimension, upper_bounds, lower_bounds, model, fitness_threshold, cognition_factor = 0.5, social_factor = 0.5):
        
        self.pop_size = pop_size
        self.dimension = dimension
        self.upper_bounds = upper_bounds
        self.lower_bounds = lower_bounds
        
        self.solutions = [] # current solution
        self.current_solutions_value = []
        self.individual_best_solution = [] # individual best solution
        self.individual_best_objective_value = [] # individual best val
        
        self.global_best_solution = [] # global best solution
        self.global_best_objective_value = 0
        self.cognition_factor = cognition_factor # particle movement follows its own search experience
        self.social_factor = social_factor  # particle movement follows the swarm search experience
        self.model = model

        self.fitness_threshold = fitness_threshold
        
    def initialize(self):
        min_index = 0
        min_val = 10000000.0
        
        for i in range(self.pop_size):
            print('init', i)
            solution = None
            objective = 100.0

            while(objective > self.fitness_threshold):
                # 初始化隨機解
                solution = []
                for d in range(self.dimension):
                    rand_pos = int(self.lower_bounds[d]+random.random()*(self.upper_bounds[d]-self.lower_bounds[d]))
                    solution.append(rand_pos)

                objective = self.model(solution)

            # update invidual best solution
            self.individual_best_solution.append(solution)
            self.individual_best_objective_value.append(objective)
            self.solutions.append(solution)
            
            # record the smallest objective val
            if objective < min_val:
                min_index = i
                min_val = objective
            
        # udpate so far the best solution
        self.global_best_solution = self.solutions[min_index].copy()
        self.global_best_objective_value = min_val
        
    def move_to_new_positions(self):
        for i,solution in enumerate(self.solutions):
            alpha = self.cognition_factor * random.random()
            beta = self.social_factor * random.random()

            for d in range(self.dimension):
                # 計算加速度
                v = alpha*(self.individual_best_solution[i][d]-self.solutions[i][d])+beta*(self.global_best_solution[d]-self.solutions[i][d])
                
                # 加上加速度到新位置
                self.solutions[i][d] += int(v)
                
                # 判斷新位置有沒有超過upper/lower bounds
                self.solutions[i][d] = int(min(self.solutions[i][d], self.upper_bounds[d]))
                self.solutions[i][d] = int(max(self.solutions[i][d], self.lower_bounds[d]))
    
    def update_best_solution(self):
        self.current_solutions_value = []
        for i,solution in enumerate(self.solutions):
            obj_val = self.model(solution)
            self.current_solutions_value.append(obj_val)
            #udpate indivisual solution
            if obj_val < self.individual_best_objective_value[i]:
                self.individual_best_solution[i] = solution
                self.individual_best_objective_value[i] = obj_val
                
                if obj_val < self.global_best_objective_value:
                    self.global_best_solution = solution
                    self.global_best_objective_value = obj_val