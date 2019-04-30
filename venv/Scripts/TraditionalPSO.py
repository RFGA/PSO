import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random


class Particle(object):
    # pbest is the fitness of particle

    def __init__(self, qtd_dimension, c1, c2, func_name='sphere'):
        _, self.boot_interval, self.fit_func = FitnessFunctions.get_func_details(func_name)
        # self.id = str(uuid.uuid4())
        self.id = 0

        self.c1 = c1
        self.c2 = c2
        self.velocity = np.zeros(qtd_dimension, int)

        self.position = []
        for d in range(qtd_dimension):
            self.position = np.insert(self.position, d, random.uniform(self.boot_interval[0], self.boot_interval[1]))

        self.p_best_position = self.position
        # The value in the current position of p_best

        self.p_best_value = self.fit_func(self.position)

        # self.p_best_value = float('inf')
        self.p_best_historic = []

    def move(self):
        self.position = self.position + self.velocity


class FitnessFunctions:
    sphere_limits = (-100, 100)
    sphere_initialization = (50, 100)

    rastrigin_limits = (-5.12, 5.12)
    rastrigin_initialization = (2.56, 5.12)

    rosenbrock_limits = (-30, 30)
    rosenbrock_initialization = (15, 30)

    @staticmethod
    def get_func_details(func_name):
        if func_name == 'sphere':
            return FitnessFunctions.sphere_limits, FitnessFunctions.sphere_initialization, \
                   FitnessFunctions.sphere_function
        elif func_name == 'rosenbrock':
            return FitnessFunctions.rosenbrock_limits, FitnessFunctions.rosenbrock_initialization, \
                   FitnessFunctions.rosenbrock_function
        elif func_name == 'rastrigin':
            return FitnessFunctions.rastrigin_limits, FitnessFunctions.rastrigin_initialization, \
                   FitnessFunctions.rastrigin_function

    @staticmethod
    def sphere_function(particle_position) -> float:
        fitness = 0
        # return (particle_position ** 2).sum() #if numpy.array
        for x in particle_position:
            fitness += (x ** 2)
        return fitness

    @staticmethod
    def rastrigin_function(particle_position):
        y = (particle_position ** 2) - 10 * np.cos(2.0 * np.pi * particle_position) + 10
        return y.sum()

    @staticmethod
    def rosenbrock_function(x):
        a = x[1:] - (x[:-1] ** 2)  # [:-1]->todos exceto os 2 últimos
        b = x[:1] - 1  # [:1]->do começo ao fim-1
        y = 100 * (a ** 2) + (b ** 2)
        return y.sum()


class Space:

    def __init__(self, qtd_dimension, particles, func_name: str, topology='global'):
        self.func_name = func_name
        self.space_limit, self.boot_interval, self.fit_func = FitnessFunctions.get_func_details(func_name)

        self.qtd_dimension = qtd_dimension
        self.g_best_value = float('inf')
        self.particles = particles
        self.g_best_position = []
        for d in range(qtd_dimension):
            # self.g_best_position.append(np.array(random.uniform(-100, 100)))
            self.g_best_position = np.insert(self.g_best_position, d, random.uniform(self.boot_interval[0],
                                                                                     self.boot_interval[1]))
        # For the local topology
        self.groups = self.init_local_communication() if topology == 'local' else []
        self.n_grp = len(self.groups)
        self.l_best_values = np.array([float('inf')]*self.n_grp)
        self.l_best_positions = np.array([[float(0)]*qtd_dimension]*self.n_grp)

        # Only the position on the particles's array (Focal topology)
        self.p_focal_pos = self.init_focal_communication() if topology == 'focal' else 0
        self.f_best_value = float('inf')
        self.f_best_position = self.particles[self.p_focal_pos].position

        self.all_p_bests = np.array([0.0]*len(self.particles))

        # define_communication_topology
    def init_local_communication(self):
        n = len(self.particles)
        groups = []
        for i in range(n):
            groups.insert(i, [i, (i - 1) % n, (i + 1) % n])  # Create the array with the ring formation
        return groups

    def init_focal_communication(self):
        return int(np.random.uniform(0, len(self.particles)))

    def set_f_best(self, by_focal: bool):
        if by_focal:
            self.f_best_value = self.particles[self.p_focal_pos].p_best_value
            self.f_best_position = self.particles[self.p_focal_pos].p_best_position
        else:
            self.f_best_value = min(self.all_p_bests)
            position = int(np.where(self.all_p_bests == self.f_best_value)[0])

            bests3 = sorted(self.all_p_bests)
            mean_bests3 = (bests3[0] + bests3[1] + bests3[2]) / 3
            self.f_best_value = mean_bests3

            self.f_best_position = self.particles[position].p_best_position

            # Other option: take the mean of the 3 fist values and define as self.f_best_value

            # bs = ['inf', 'inf', 'inf']
            # b = self.all_p_bests[0]
            # i = 0
            # less = -1
            # while max(bs) == 'inf':
            #     for self.x in range(1, len(self.all_p_bests)-1):
            #         if b > self.all_p_bests[x] and x != less:
            #             b = self.all_p_bests[x]
            #     less = self.x
            #     bs[i] = b
            #     i += 1

    def move_particle_f(self, coef_val, clerc: bool, vmax, move_focal: bool):
        r1 = np.random.rand(self.qtd_dimension)
        r2 = np.random.rand(self.qtd_dimension)

        if move_focal:
            particle = self.particles[self.p_focal_pos]
            v0 = particle.velocity
            if clerc:
                f_velocity = coef_val * (v0 + particle.c1 * r1 * (particle.p_best_position - particle.position) +
                                         particle.c2 * r2 * (self.g_best_position - particle.position))
            else:
                f_velocity = coef_val * v0 + particle.c1 * r1 * (particle.p_best_position - particle.position) + \
                             particle.c2 * r2 * (self.g_best_position - particle.position)

            np.place(f_velocity, f_velocity > vmax, [vmax])
            particle.velocity = f_velocity
            particle.move()
        else:
            for particle in self.particles:
                v0 = particle.velocity

                if clerc:
                    f_velocity = coef_val * (v0 + particle.c1 * r1 * (particle.p_best_position - particle.position) +
                                             particle.c2 * r2 * (self.g_best_position - particle.position))
                else:
                    f_velocity = coef_val * v0 + particle.c1 * r1 * (particle.p_best_position - particle.position) + \
                                 particle.c2 * r2 * (self.g_best_position - particle.position)

                np.place(f_velocity, f_velocity > vmax, [vmax])
                particle.velocity = f_velocity
                particle.move()

    @staticmethod
    def inside_the_limit(particle: Particle, limit_interval: tuple) -> bool:
        if any(i < limit_interval[0] or i > limit_interval[1] for i in particle.position):
            return False
        else:
            return True

    # For the local topology #Could be used set_p_best()...
    def set_p_best_l(self):
        for i in range(len(self.groups)):
            group = self.groups[i]
            n = len(group)
            for j in range(n):
                p_of_group = self.particles[group[j]]
                possible_p_best = self.fit_func(p_of_group.position)
                if self.inside_the_limit(p_of_group, self.space_limit) and p_of_group.p_best_value > possible_p_best:
                    p_of_group.p_best_value = possible_p_best
                    p_of_group.p_best_position = p_of_group.position

    def set_l_best(self):
        for i in range(len(self.groups)):
            group = self.groups[i]
            n = len(group)
            for j in range(n):
                p_group = self.particles[group[j]]
                if self.inside_the_limit(p_group, self.space_limit) \
                        and self.l_best_values[i] > p_group.p_best_value:  # 'i' is the group position with 3 particles

                    self.l_best_values[i] = p_group.p_best_value
                    self.l_best_positions[i] = p_group.p_best_position
                    # self.l_best_positions[i] = np.array(p_group.p_best_position)

    # For the local topology
    def move_particle_l(self, coef_val, clerc: bool, vmax):
        for i in range(len(self.groups)):
            group = self.groups[i]
            n = len(group)
            for j in range(n):
                p_group = self.particles[group[j]]
                # for dimension in range(self.qtd_dimension):
                # For operations with vector, isn't necessary to use repetition structure.
                # global c1, c2

                r1 = np.random.rand(self.qtd_dimension)
                r2 = np.random.rand(self.qtd_dimension)
                v0 = p_group.velocity
                if clerc:
                    f_velocity = coef_val * (v0 + p_group.c1 * r1 * (p_group.p_best_position - p_group.position) +
                                             p_group.c2 * r2 * (self.l_best_positions[i] - p_group.position))
                else:
                    f_velocity = coef_val * v0 + p_group.c1 * r1 * (p_group.p_best_position - p_group.position) + \
                                 p_group.c2 * r2 * (self.l_best_positions[i] - p_group.position)

                # lista = [[5 if i > 5 else i for i in lista]
                # verify if in f_velocity there is vel > vmax and limit with vmax
                np.place(f_velocity, f_velocity > vmax, [vmax])
                p_group.velocity = f_velocity
                p_group.move()

    # Se a partícula estiver fora dos limites, deixar que ela volte sozinha, mantendo a coerência!!!!!!!!!
    # Pág.8: condições de contorno:uma particula não é avaliada (seu pbest) quando ela sai do espaço de pesquisa viável.
    # Atualizar apenas o gbest na velocidade e nova posição da particula para que ela tenda a voltar.
    def set_p_best(self):
        for particle in self.particles:

            # possible_p_best = FitnessFunctions.rosenbrock_function(particle.position)
            possible_p_best = self.fit_func(particle.position)

            if self.inside_the_limit(particle, self.space_limit) and particle.p_best_value > possible_p_best:
                particle.p_best_value = possible_p_best
                particle.p_best_position = particle.position
                self.all_p_bests[particle.id] = particle.p_best_value

    def set_g_best(self):
        for particle in self.particles:
            if self.g_best_value > particle.p_best_value:
                self.g_best_value = particle.p_best_value
                self.g_best_position = particle.p_best_position

    def move_particle(self, coef_val, clerc: bool, vmax):

        for particle in self.particles:
            # for dimension in range(self.qtd_dimension):
            # For operations with vector, isn't necessary to use repetition structure.
            #global c1, c2

            r1 = np.random.rand(self.qtd_dimension)
            r2 = np.random.rand(self.qtd_dimension)
            v0 = particle.velocity

            if clerc:
                f_velocity = coef_val * (v0 + particle.c1 * r1 * (particle.p_best_position - particle.position) +
                                         particle.c2 * r2 * (self.g_best_position - particle.position))
            else:
                f_velocity = coef_val * v0 + particle.c1 * r1 * (particle.p_best_position - particle.position) + \
                             particle.c2 * r2 * (self.g_best_position - particle.position)


#            print(f_velocity)
#           lista = [[5 if i > 5 else i for i in lista]
            np.place(f_velocity, f_velocity > vmax, [vmax])
            particle.velocity = f_velocity
            particle.move()


class PSO:

    def __init__(self, n_particles=30, n_dimensions=30, n_iterations=10000, c1=2.05, c2=2.05, inertia=(1, (0.8, 0.8)),
                 fitness_func_name='sphere', topology='global', vmax=50):
        self.func_name = fitness_func_name
        self.topology = topology
        _, self.boot_interval, self.f_function = FitnessFunctions.get_func_details(self.func_name)

        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.n_iterations = n_iterations
        self.c1 = c1
        self.c2 = c2
        self.inertia = inertia
        self.vmax = vmax

        self.particles = self.set_particles()

        # Setting the space of problem
        self.space = Space(self.n_dimensions, self.particles, self.func_name, topology=self.topology)

    # Setting the new particles and putting ID
    def set_particles(self):
        particles = [Particle(self.n_dimensions, self.c1, self.c2, self.func_name) for _ in range(self.n_particles)]
        for id in range(self.n_particles):
            particles[id].id = id
        return particles

    def inertia_control_type(self, type_i: int, w_interval: tuple, n_iteration=1, current_iter=1):
        if type_i == 1:
            return w_interval[0]
        elif type_i == 2:
            return w_interval[0] - ((w_interval[0] - w_interval[1]) * (current_iter / n_iteration))
        elif type_i == 3:
            phi = self.c1 + self.c2
            return 2 / np.abs(2 - phi - np.sqrt(phi**2 - 4*phi))

    def print_details(self, data_name: str):
        with open('data/' + data_name + '.txt', 'a') as file:
            file.write(str(self.func_name)+"Inertia: "+str(self.inertia[0]) + "\n")

    def print_data(self, iteration, data_name: str, content):
        data_name = data_name.replace(" ", "")

        # Other option: -> file = open('/data/' + data_name + '.txt', 'a')...file.write()...file.close()
        with open('data/'+data_name+'.txt', 'a') as file:
            # type() is enough, but there is isinstance()
            if type(content) == list or type(content) == (np.ndarray):
                file.write(str(iteration) + ", " + str(min(content)) + "\n")
                # for x in range(len(content)):
                #     file.write(str(iteration)+","+str(x)+", "+str(content[x])+"\n")
            else:
                file.write(str(iteration)+"*"+str(content) + "\n")

    def start(self):

        iteration = 0
        clerc = True if (self.inertia[0] == 3) else False
        if self.topology == 'global':
            self.print_details("g_best_value")
            while iteration < self.n_iterations:
                print(self.space.g_best_value)
                self.print_data(iteration, "g_best_value", self.space.g_best_value)

                w = self.inertia_control_type(self.inertia[0], self.inertia[1], self.n_iterations, iteration)
                # print(w)
                self.space.move_particle(w, clerc, self.vmax)
                self.space.set_p_best()
                self.space.set_g_best()

                # space.print_particles()
                iteration += 1
            print("The best solution is: ", self.space.g_best_value, " in ", self.space.g_best_position,
                  " in n_iterations: ", iteration)
            # print('Min. lbest: ', min(self.space.l_best_values))
            print(self.boot_interval)

        elif self.topology == 'local':
            self.print_details("l_best_values")
            while iteration < self.n_iterations:
                print(min(self.space.l_best_values))
                self.print_data(iteration, "l_best_values", self.space.l_best_values)

                w = self.inertia_control_type(self.inertia[0], self.inertia[1], self.n_iterations, iteration)
                self.space.move_particle_l(w, clerc, self.vmax)
                self.space.set_p_best_l()
                self.space.set_l_best()
                iteration += 1

            print("The best solution is: ", self.space.l_best_values, " in ", self.space.l_best_positions,
                  " in n_iterations: ", iteration)
            # print('Min. lbest: ', min(self.space.l_best_values))
            print(self.boot_interval)
        elif self.topology == 'focal':
            self.print_details("f_best_value")
            while iteration < self.n_iterations:

                self.space.set_p_best()
                self.space.set_f_best(by_focal=True)
                w = self.inertia_control_type(self.inertia[0], self.inertia[1], self.n_iterations, iteration)
                self.space.move_particle_f(w, clerc, self.vmax, move_focal=False)
                self.space.set_p_best()
                self.space.set_f_best(by_focal=False)
                self.space.move_particle_f(w, clerc, self.vmax, move_focal=True)
                print(self.space.f_best_value)
                self.print_data(iteration, "f_best_value", self.space.f_best_value)
                iteration += 1

            print("The best solution is: ", self.space.f_best_value, " in ", self.space.f_best_position, " in n_iterations: ", iteration)
            # print('Min. lbest: ', min(self.space.l_best_values))
            print("Function: ", self.func_name, "Init: ", self.boot_interval)


# W = eval(input("Valor para coeficiente de inércia W: "))  # eval() execute the expression of input
# (self, n_particles=30, n_dimensions=30, n_iterations=10000, c1=2.05, c2=2.05, inertia=(1, (0.8, 0.8)),
# fitness_func_name='sphere', topology='global', vmax=50):

# rosenbrock, rastrigin, sphere
pso = PSO(30, 30, 10000, inertia=(3, (0.8, 0.8)), fitness_func_name='sphere', topology='local', vmax=1)
pso.start()
