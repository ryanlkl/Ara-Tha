from resources import bool_to_int

class FitnessFunction:
    def __init__(self, number_of_nodes, types):
        self.funct_type = types
        self.total_configs = 1 << number_of_nodes
        self.configurations = self._generate_configurations(number_of_nodes)
        self.states_list = [-1] * self.total_configs
        
    def _generate_configurations(self, number_of_nodes):
        return [[(i >> j) & 1 == 1 for j in range(number_of_nodes  -1, -1, -1)] for i in range(self.total_configs)]
    
    def get_total_configs(self):
        return self.total_configs
    
    def get_node_state(self, GRN_configuration, node):
        if GRN_configuration >= self.get_total_configs():
            raise ValueError("Number of possible configurations exceeded\n")
        else:
            return self.configurations[GRN_configuration][node]
        
    def get_state(self, GRN_configuration):
        if GRN_configuration >= self.get_total_configs():
            raise ValueError("Number of possible configurations exceeded\n")
        else:
            return self.configurations[GRN_configuration]
    
    def find_attractors(self, solution, synch_update):
        print("Find Attractors")
        nds = solution.get_number_of_nodes()
        stts = self.get_total_configs()
        current_config = [False] * nds
        temp_sequence = [-1] * stts
        next_config = [False] * nds

        cursor_states = 0
        cursor_temp = 0
        while cursor_states < stts:
            cursor_temp = 0
            while self.states_list[cursor_states] != -1 and cursor_states < stts - 1:
                cursor_states += 1
            
            stop, fixed = False, False

            for i in range(nds):
                current_config[i] = self.get_node_state(cursor_states, i)

            temp_sequence[cursor_temp] = bool_to_int(nds, current_config)

            while not stop:
                if synch_update:
                    solution.synch_next_state(current_config, next_config)
                else:
                    solution.asynch_next_state(current_config, next_config)
                
                next_state = bool_to_int(nds, next_config)

                if self.states_list[next_state] >= 0:  # check if old fixed point
                    stop = True
                    fixed = True
                    next_state = self.states_list[next_state]
                elif next_state == temp_sequence[cursor_temp]:  # check if new fixed point
                    stop = True
                    fixed = True
                elif self.states_list[next_state] == -10:  # check if old cycle
                    stop = True
                else:
                    # check if new cycle
                    i = 0
                    while stop != True and i < cursor_temp:
                        if temp_sequence[i] == next_state:
                            stop = True
                        i += 1

                if stop and fixed:  # if fixed point
                    for i in range(cursor_temp + 1):
                        if i < stts:
                            self.states_list[temp_sequence[i]] = next_state
                elif stop:  # if cycle
                    for i in range(cursor_temp + 1):
                        if i < stts:
                            self.states_list[temp_sequence[i]] = -10
                else:  # continue
                    for i in range(nds):
                        current_config[i] = next_config[i]
                    cursor_temp += 1
                    if cursor_temp == stts:  # check no overrun
                        stop = True
                    else:
                        temp_sequence[cursor_temp] = next_state

            # Reset temp sequence for next iteration
            for i in range(cursor_temp + 1):
                if i < stts:
                    temp_sequence[i] = -1
            cursor_states += 1

    def get_funct_type(self):
        return self.funct_type
