import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
import uuid


class Particle(object):
    # pbest is the fitness of particle

    def __init__(self, qtd_dimension, c1, c2):

        # self.id = str(uuid.uuid4())
        self.id = 0

        self.c1 = c1
        self.c2 = c2

        # self.velocity = np.array([0] * qtd_dimension)
        self.velocity = np.zeros(qtd_dimension, int)

        self.position = []
        for d in range(qtd_dimension):
            # self.position.append(random.uniform(-100, 100))
            self.position = np.insert(self.position, d, random.uniform(-100, 100))

        self.p_best_position = self.position
        # The value in the current position of p_best
        self.p_best_value = Space.fitness_function_sphere(self.position)
        # self.p_best_value = float('inf')
        self.p_best_historic = []

    def __str__(self):
        print("Estou em:  ", self.position, " e pbest está: ", self.p_best_position)

    def move(self):
        self.position = self.position + self.velocity


class FitnessFunctions:

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

    def __init__(self, qtd_dimension, particles):

        self.qtd_dimension = qtd_dimension
        self.g_best_value = float('inf')
        # self.particles = []
        self.particles = particles
        self.g_best_position = []
        for d in range(qtd_dimension):
            # self.g_best_position.append(np.array(random.uniform(-100, 100)))
            self.g_best_position = np.insert(self.g_best_position, d, random.uniform(-100, 100))



    def print_particles(self):
        for particle in self.particles:
            particle.__str__()
            # print("\n")

    @staticmethod
    def fitness_function_sphere(particle_position: [float]) -> float:
        fitness = 0
        # return (particle_position ** 2).sum() #if numpy.array
        for x in particle_position:
            fitness += (x ** 2)
        return fitness


# Se a partícula estiver fora dos limites, deixar que ela volte sozinha, mantendo a coerência!!!!!!!!!

    def set_p_best(self):
        for particle in self.particles:
            # print(particle.p_best_value)
            possible_p_best = FitnessFunctions.sphere_function(particle.position)
            # possible_p_best = self.fitness_function_sphere(particle.position)

            # if particle.p_best_position in range(-100, 100):

            if particle.p_best_value > possible_p_best:
                particle.p_best_value = possible_p_best
                particle.p_best_position = particle.position
        # print("\n")

    def set_g_best(self):
        for particle in self.particles:
            print(self.g_best_value)
            if self.g_best_value > particle.p_best_value:
                self.g_best_value = particle.p_best_value
                self.g_best_position = particle.p_best_position
        print("\n")

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
                 vmax=50):
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.n_iterations = n_iterations
        self.c1 = c1
        self.c2 = c2
        self.inertia = inertia
        self.vmax = vmax

        #self.n_particles = int(input("Qtd. partículas: "))
        #self.n_dimensions = int(input("Qtd. dimensões: "))
        #self.n_iterations = int(input("Qtd. iterações: "))

        #self.c1 = self.c2 = float(input("Valor para as constantes c1 e c2: "))

        #W = eval(input("Valor para coeficiente de inércia W: "))  # eval() execute the expression of input

        # Setting the new particles and putting ID
        self.particles = [Particle(self.n_dimensions, self.c1, self.c2) for _ in range(self.n_particles)]
        for id in range(1, self.n_particles+1):
            print(id-1)
            self.particles[id-1].id = id

        # Setting the space of problem
        self.space = Space(self.n_dimensions, self.particles)

    # with open('g_best_results.txt', 'a') as file:
    #    result = file.write('g_best   iteration_n'+"\n")

    def inertia_control_type(self, type_i: int, w_interval: tuple, n_iteration=1, current_iter=1):
        if type_i == 1:
            return w_interval[0]
        elif type_i == 2:
            return w_interval[0] - ((w_interval[0] - w_interval[1]) * (current_iter / n_iteration))
        elif type_i == 3:
            phi = self.c1 + self.c2
            return 2 / np.abs(2 - phi - np.sqrt(phi**2 - 4*phi))

    # def returns_variable_inertia():
    #    global W, iterations
    #    return W[iterations]

    # if type(W) is tuple:
    #    W = np.linspace(W[0], W[1], n_iterations)
    #    w = [returns_variable_inertia()]

    def start(self):

        iterations = 0
        w = 0

        while iterations < self.n_iterations:

            # with open('g_best_results.txt', 'a') as file:
            #   result = file.write(str(space.g_best_value) + " " + str(iterations)+"\n")30
            w = self.inertia_control_type(self.inertia[0],self.inertia[1], self.n_iterations, iterations)
            print(w)
            clerc = True if (self.inertia[0] == 3) else False
            self.space.move_particle(w, clerc, self.vmax)
            self.space.set_p_best()
            self.space.set_g_best()

            # space.print_particles()
            iterations += 1

        print("The best solution is: ", self.space.g_best_value, " in ", self.space.g_best_position, " in n_iterations: ", iterations)


# (self, n_particles=30, n_dimensions=30, n_iterations=10000, c1=2.05, c2=2.05, inertia=(1, (0.8, 0.8)), vmax=50):
pso = PSO(30, 30, 10000, vmax=100)
# VELOCIDADE BAIXA TENDE A FAZER DEMORAR A CONVERGÊNCIA
pso.start()


# P = Particle(30)
# print(P.position)
# print(P.velocity)
# P.move()
# print(P.position)