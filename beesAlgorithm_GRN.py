import numpy as np
from resources import bool_to_int, LSearch, FitnessF
import random
from nAttractors import NAttractors
from reducedMAB import ReducedMAB
from multiprocessing import Pool
import time
from bees_algorithm import BeesAlgorithm
from bee import Bee

class BeesAlgorithm_GRN:
    def __init__(self, file_name):
        self.T = 0
        self.ns = 0
        self.nb = 0
        self.ne = 0
        self.nrb = 0
        self.nre = 0
        self.ngh_shr = False
        self.init_ngh = 0
        self.min_ngh = 0
        self.stlim = 0
        self.sites = 0
        self.colony = None
        self.fitness_mab = None
        self.fitness_fixed_points = None
        self.best_fitness = 0
        self.solut_counter = 0
        self.synch_mode = True
        self.save_files = file_name
        self.search_threshold = 0
        self.save_threshold = 0
        self.search_type = LSearch.REINIT
        self.attr_type = FitnessF.TARGET
        self.number_of_attractors = 0

    def bees_alg_initialisation(self, file_BA_pars, file_GRN_pars):
        print("Initialising Bees Algorithm")
        try:
            with open(file_BA_pars, "r") as infile_ba:
                self.T = int(infile_ba.readline().strip().split()[0])
                self.ns = int(infile_ba.readline().strip().split()[0])
                self.nb = int(infile_ba.readline().strip().split()[0])
                self.ne = int(infile_ba.readline().strip().split()[0])
                self.nrb = int(infile_ba.readline().strip().split()[0])
                self.nre = int(infile_ba.readline().strip().split()[0])
                self.ngh_shr = infile_ba.readline().strip().split()[0].lower() == "true"
                self.init_ngh = int(infile_ba.readline().strip().split()[0])
                self.min_ngh = int(infile_ba.readline().strip().split()[0])
                self.stlim = int(infile_ba.readline().strip().split()[0])
                self.synch_mode = infile_ba.readline().strip().split()[0].lower() == "true"
                self.search_threshold = float(infile_ba.readline().strip().split()[0])
                self.save_threshold = float(infile_ba.readline().strip().split()[0])
                search_type = infile_ba.readline().strip().split()[0]
                if search_type.lower() == "delta":
                    self.search_type = LSearch.DELTA
                elif search_type.lower() == "reinit":
                    self.search_type = LSearch.REINIT
                else:
                    raise ValueError("Invalid SearchType parameter in BA_parameters")
                attr_type = infile_ba.readline().strip().split()[0]
                if attr_type.lower() == "matrix":
                    self.attr_type = FitnessF.ATTRACT_MATR
                elif attr_type.lower() == "optimal":
                    self.attr_type = FitnessF.ATTRACT_OPT
                else:
                    raise ValueError("Invalid AttrType parameter in BA_parameters")
        except FileNotFoundError:
            raise FileNotFoundError("Can't open BA parameter file")

        print("Read BA parameters:")
        print(f"T = {self.T}")
        print(f"ns = {self.ns}")
        print(f"nb = {self.nb}")
        print(f"ne = {self.ne}")
        print(f"nrb = {self.nrb}")
        print(f"nre = {self.nre}")
        print(f"nghShr = {self.ngh_shr}")
        if self.ngh_shr:
            print(f"initNgh = {self.init_ngh}")
            print(f"minNgh = {self.min_ngh}")
        else:
            print(f"ngh = {self.init_ngh}")
        print(f"stlim = {self.stlim}")
        print(f"synchMode = {self.synch_mode}")
        print(f"saveThreshold = {self.save_threshold}")
        print(f"local search = {self.search_type}\n")

        try:
            with open(file_GRN_pars, 'r') as infile_grn:
                self.number_of_nodes = int(infile_grn.readline().strip().split()[0])
                self.lower_extreme = int(infile_grn.readline().strip().split()[0])
                self.upper_extreme = int(infile_grn.readline().strip().split()[0])
                self.upper_bound = [self.upper_extreme] * self.number_of_nodes
                self.lower_bound = [self.lower_extreme] * self.number_of_nodes
        except FileNotFoundError:
            raise FileNotFoundError("Can't open GRN parameters file")
        
        print("Read GRN parameters:")
        print(f"nodes = {self.number_of_nodes}")
        print(f"lowerExtreme = {self.lower_extreme}")
        print(f"upperExtreme = {self.upper_extreme}")
        print(f"update mode = {'synchronous' if self.synch_mode else 'asynchronous'}\n")

    def read_attractors(self, file_attractors):
        print("Read Attractors")
        nodes = self.number_of_nodes
        print(nodes)
        temp_point = [False] * nodes

        try:
            with open(file_attractors, "r") as infile:
                self.number_of_attractors = int(infile.readline().strip())
                nodes = self.number_of_nodes
                fixed_points = []

                for _ in range(self.number_of_attractors):
                    temp_point = list(map(int, infile.readline().split()))
                    fixed_points.append(bool_to_int(nodes, temp_point))

            return fixed_points
        except FileNotFoundError:
            raise FileNotFoundError("Can't open fixed points file")
        except ValueError:
            raise ValueError(f"Error: Invalid data format in '{file_attractors}'.")
        
    def set_obj_functions(self, target_file, attr_file):
        print("Set Obj Function")
        type_of_function = self.attr_type
        fixed_points = None

        if type_of_function == FitnessF.ATTRACT_MATR:
            fixed_points = self.read_attractors(attr_file)
            self.fitness_fixed_points = NAttractors(
                self.number_of_attractors,
                self.number_of_nodes,
                fixed_points,
                type_of_function
                )
        elif type_of_function == FitnessF.ATTRACT_OPT:
            fixed_points = self.read_attractors(attr_file)
            self.fitness_fixed_points = NAttractors(
                self.number_of_attractors,
                self.number_of_nodes,
                fixed_points,
                type_of_function
                )
        else:
            raise ValueError("Call to non existent fitness function type in set_obj_function()")
        print(self.fitness_fixed_points)
    
    def bees_algorithm(self):
        print("Running Bees Algorithm Optimisation...")

        optimiser = BeesAlgorithm(
            self.fitness_fixed_points.calculate_fitness,
            self.lower_bound,
            self.upper_bound,
            ns=self.ns,
            nb=self.nb,
            ne=self.ne,
            nrb=self.nrb,
            nre=self.nre,
            stlim=self.stlim
        )

        optimiser.performFullOptimisation(max_iteration=100)
        best_solution = optimiser.best_solution
        best_fitness = best_solution.score
        best_coords = best_solution.values
        curr_sites = optimiser.current_sites

        print(best_fitness)
        print(best_coords)
        print(curr_sites)
            
    def display_results(self):
        saved = self.get_solut_counter()
        o_funct = self.get_attract_function()

        print("Best solutions found: \n")
        for i in range(saved):
            file_name = f"{self.get_save_files()}{i}.txt"
            member = self.get_colony_member(0)
            solution = Bee(
                member.get_number_of_nodes(),
                member.get_lower_extreme(),
                member.get_upper_extreme(),
                file_name,
                self.init_ngh
                )
            print(f"Solution {i}\n")
            o_funct.evaluate_display(solution, self.get_synch_mode())

    def get_save_files(self):
        return self.save_files