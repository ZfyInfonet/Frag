import networkx as nx
import random
import util
import copy
import config


class GeneticAlgorithm:
    def __init__(self, sfc_list, g_now: nx.Graph, population_size):
        self.population_size = population_size
        self.node_num = g_now.number_of_nodes()
        self.sfc_list = sfc_list
        self.g_now = g_now
        g_init = copy.deepcopy(g_now)
        self.g_init = util.clearGraph(g_init)
        self.total_vnf_list = []
        init_dna = []
        for sfc in self.sfc_list:
            if sfc.is_mapped:
                for vnf in sfc.vnf_list:
                    self.total_vnf_list.append(vnf)
                    init_dna.append(vnf.node_id)
        self.dna_length = len(self.total_vnf_list)
        self.init_dna = init_dna
        self.group = self.init_group()

    def checkDnaConstraint(self, DNA):
        new_sfc_list = copy.deepcopy(self.sfc_list)
        new_total_vnf_list = copy.deepcopy(self.total_vnf_list)
        new_graph = copy.deepcopy(self.g_init)
        for i in range(self.dna_length):
            vnf = new_total_vnf_list[i]
            place_result = util.addVnfToNode(vnf, DNA[i], new_graph)
            if not place_result:
                return False, None
        link_flag = True        # assume all the vnf links can be mapped to graph.
        for sfc in new_sfc_list:
            if sfc.is_mapped is True:
                for link in sfc.vnf_link_list:
                    flag = False        # link has not been mapped yet.
                    src_node, dst_node = link.vnfs[0].node_id, link.vnfs[1].node_id
                    paths = new_graph.graph['paths'][src_node - 1][dst_node - 1]
                    for path in paths:
                        if util.checkPathResource(link, path, new_graph):
                            util.addVnfLinkToPath(link, path, new_graph)
                            flag = True     # if an available path exists
                            break
                    if not flag:            # if any link cannot be mapped to graph, link flag turns to False
                        link_flag = False
                        break
            else:
                break
        if not link_flag:
            return False, None
        return True, new_graph

    def calFitness(self, new_dna, parameter_list):

        is_valid, new_graph = self.checkDnaConstraint(new_dna)
        if is_valid:
            # check graph resource utilization
            var_cpu, var_mem, var_bw = util.calResourceUsageFromGraph(new_graph)
            VAR = parameter_list[0] * var_cpu + parameter_list[1] * var_mem + parameter_list[2] * var_bw
            # check migration cost
            mig_list = []
            for i in range(self.dna_length):
                if self.init_dna[i] != new_dna[i]:
                    migration = {'vnf': self.total_vnf_list[i], 'src_node': self.init_dna[i], 'dst_node': new_dna[i]}
                    mig_list.append(migration)

            COST = 0
            for mig in mig_list:
                COST += util.calMigrationCost(mig['vnf'], mig['src_node'], mig['dst_node'], self.g_init)

            fitness = -(parameter_list[3] * VAR + parameter_list[4] * COST)
            return True, fitness
        else:
            return False, float('inf')

    def init_group(self):
        # step 1

        group = [self.init_dna]
        # step 2
        load_balance_dna = []
        g_new = copy.deepcopy(self.g_init)
        tot_vnf_list = copy.deepcopy(self.total_vnf_list)
        valid_flag = True
        for vnf in tot_vnf_list:
            max_total_residual_node = None
            max_total_residual_ratio = 0
            for node in g_new.nodes:
                cpu_res, mem_res = g_new.nodes[node]['cpu_residual'], g_new.nodes[node]['mem_residual']
                cpu_cap, mem_cap = g_new.nodes[node]['cpu_capacity'], g_new.nodes[node]['mem_capacity']
                if cpu_res >= vnf.req_cpu and mem_res >= vnf.req_mem:
                    total_residual_ratio = cpu_res / cpu_cap + mem_res / mem_cap
                    if total_residual_ratio > max_total_residual_ratio:
                        max_total_residual_ratio = total_residual_ratio
                        max_total_residual_node = node
            if not util.addVnfToNode(vnf, max_total_residual_node, g_new):
                valid_flag = False
        if valid_flag:
            for vnf in tot_vnf_list:
                load_balance_dna.append(vnf.node_id)
            group.append(load_balance_dna)
        while 0 < len(group) < config.population_size:
            # step 3
            start = 0
            end = 1
            step = 1 / config.tau_1
            seq = [start + step * i for i in range(int((end - start) / step) + 1)]

            for i in seq:
                parameters = [config.a1, config.a2, config.a3, i, 1 - i]
                group = self.evolution(group, config.evolution, parameters)
            # step 4
            parameters = [config.a1, config.a2, config.a3, 0, 1]
            for i in range(config.tau_2):
                group = self.evolution(group, config.evolution, parameters)
            parameters = [config.a1, config.a2, config.a3, 1, 0]
            for i in range(config.tau_2):
                group = self.evolution(group, config.evolution, parameters)

            # step 5
            parameters = [config.a1, config.a2, config.a3, config.a1, config.a2]
            for i in range(config.tau_3):
                group = self.evolution(group, config.evolution, parameters)
        return group

    def die(self, group, parameters):
        fitness_list = []
        valid_list = []
        for dna in group:
            is_valid, fitness = self.calFitness(dna, parameters)
            fitness_list.append(fitness)
            valid_list.append(is_valid)
        new_group = []
        new_fitness = []
        for i in range(len(group)):
            if valid_list[i]:
                new_group.append(group[i])
                new_fitness.append(fitness_list[i])
        while len(new_group) > self.population_size:
            min_fit_id = new_fitness.index(min(new_fitness))
            new_group.remove(new_group[min_fit_id])
            new_fitness.remove(min(new_fitness))
        return new_group

    def evolution(self, group, iterate_num, parameters):
        if len(group) == 0:
            return group
        print(group)
        for i in range(iterate_num):
            fitness_list = []
            for dna in group:
                is_valid, fitness = self.calFitness(dna, parameters)
                fitness_list.append(fitness)
            max_fit_dna = group[fitness_list.index(max(fitness_list))]
            group.append(max_fit_dna)
            if len(group) > self.population_size:
                group = self.die(group, parameters)
                if len(group) == self.population_size:
                    return group
            # crossover
            group = self.cross(group)
            if len(group) > self.population_size:
                group = self.die(group, parameters)
                if len(group) == self.population_size:
                    return group
            # mutation
            group = self.mutation(group)
            if len(group) > self.population_size:
                group = self.die(group, parameters)
                if len(group) == self.population_size:
                    return group
        return group

    def cross(self, group, cross_probability=config.crossover_probability,
              swap_probability=config.crossover_swap_probability):
        new_group = []
        for father in group:
            new_group.append(father)
            child = copy.deepcopy(father)
            if random.random() < cross_probability:
                mother = random.choice(group)
                for pos in range(self.dna_length):
                    if random.random() < swap_probability:
                        child[pos] = mother[pos]
            new_group.append(child)
        return new_group

    def mutation(self, group, mutation_probability=config.mutation_probability):
        new_group = []
        for man in group:
            new_man = copy.deepcopy(man)
            new_group.append(new_man)
            if random.random() < mutation_probability:
                mutation_pos = random.randint(0, self.dna_length - 1)
                mutation_value = random.randint(1, self.node_num)
                new_man[mutation_pos] = mutation_value
                new_group.append(new_man)
        return new_group


def run(g_cp: nx.Graph, sfc_list, t, overload_nodes, overload_edges, GAT_model = None):
    migration_count = 0
    migration_total_cost = 0
    GA = GeneticAlgorithm(sfc_list, g_cp, config.population_size)
    parameters = [config.a1, config.a2, config.a3, config.a1, config.a2]
    group = GA.evolution(GA.group, config.tot_evolution, parameters)
    max_dna = None
    max_fitness = -float('inf')
    for dna in group:
        is_valid, fitness = GA.calFitness(dna, parameters)
        if is_valid and fitness > max_fitness:
            max_dna = dna
    mig_list = []
    if max_dna is None:
        return 0, 0
    for i in range(GA.dna_length):
        if GA.init_dna[i] != max_dna[i]:
            migration = {'vnf': GA.total_vnf_list[i], 'src_node': GA.init_dna[i], 'dst_node': max_dna[i]}
            mig_list.append(migration)
    for mig in mig_list:
        vnf = mig['vnf']

        dst_node = mig['dst_node']
        result, cost = util.migVnfToNode(vnf, dst_node, g_cp)
        if result:
            migration_count += 1
            migration_total_cost += cost
            sfc = sfc_list[vnf.sfc_id]
            for link in sfc.vnf_link_list:
                if vnf in link.vnfs:
                    another_end_index = 1 - link.vnfs.index(vnf)
                    another_vnf = link.vnfs[another_end_index]
                    another_node = another_vnf.node_id
                    path_set = g_cp.graph['paths'][dst_node - 1, another_node - 1]
                    for path in path_set:
                        if util.checkPathResource(link, path, g_cp):
                            util.delVnfLinkMapping(link, g_cp)
                            util.addVnfLinkToPath(link, path, g_cp)
                            break
    return migration_count, migration_total_cost

