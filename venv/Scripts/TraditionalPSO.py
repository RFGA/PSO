import numpy as np
import random
import uuid


class Particle(object):
    # pbest is the fitness of particle

    def __init__(self, qtd_dimension):

        self.id = str(uuid.uuid4())

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
            print(particle.p_best_value)
            possible_p_best = self.fitness_function_sphere(particle.position)

            # if particle.p_best_position in range(-100, 100):

            if particle.p_best_value > possible_p_best:
                particle.p_best_value = possible_p_best
                particle.p_best_position = particle.position
        print("\n")

    def set_g_best(self):
        for particle in self.particles:
            if self.g_best_value > particle.p_best_value:
                self.g_best_value = particle.p_best_value
                self.g_best_position = particle.p_best_position

    def move_particle(self):

        for particle in self.particles:
            # for dimension in range(self.qtd_dimension):
            # For operations with vector, isn't necessary to use repetition structure.
            global W, c1, c2

            r1 = np.random.rand(self.qtd_dimension)
            r2 = np.random.rand(self.qtd_dimension)
            v0 = particle.velocity

            f_velocity = W * v0 + c1 * r1 * (particle.p_best_position - particle.position) + c2 * r2 * \
                (self.g_best_position - particle.position)

            if f_velocity.any() <= 1.0:
                particle.velocity = f_velocity
                particle.move()



n_particles = int(input("Qtd. partículas: "))
n_dimensions = int(input("Qtd. dimensões: "))
n_iterations = int(input("Qtd. iterações: "))

c1 = c2 = float(input("Valor para as constantes c1 e c2: "))
W = float(input("Valor para coeficiente de inércia W: "))

particles = [Particle(n_dimensions) for _ in range(n_particles)]
space = Space(n_dimensions, particles)

# with open('g_best_results.txt', 'a') as file:
#    result = file.write('g_best   iteration_n'+"\n")

iterations = 0
while iterations < n_iterations:
    space.move_particle()
    space.set_p_best()
    space.set_g_best()

   # with open('g_best_results.txt', 'a') as file:
    #    result = file.write(str(space.g_best_value) + " " + str(iterations)+"\n")

    # space.print_particles()
    iterations += 1

print("The best solution is: ", space.g_best_value, " in ", space.g_best_position, " in n_iterations: ", iterations)

# P = Particle(30)
# print(P.position)
# print(P.velocity)
# P.move()
# print(P.position)