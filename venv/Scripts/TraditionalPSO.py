import numpy as np
import random

# inertia
W = None
c1 = 0.8
c2 = 0.9


class Particle(object):
    # pbest is the fitness of particle

    def __init__(self, qtd_dimension):

        # o valor poderia ser o retorno da função fitness para a posição inicial gerada
        # self.p_best_value = Space.fitness_function_sphere(self) (falta testar...)
        self.p_best_value = float('inf')
        self.velocity = np.array([0] * qtd_dimension)

        self.position = []
        for d in range(qtd_dimension):
            # self.position.append(random.uniform(-100, 100))
            self.position = np.insert(self.position, d, random.uniform(-100, 100))
        # self.position = np.array(self.position)
        self.p_best_position = self.position

    def __str__(self):
        print("Estou em:  ", self.position, " e meu pbest é: ", self.p_best_position)

    def move(self):
        self.position = self.position + self.velocity


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
        print("\n")

    @staticmethod
    def fitness_function_sphere(particle: Particle) -> float:
        fitness = 0
        for x in particle.position:
            # fitness += (x ** 2)
            fitness += (x ** 2) +
        #return fitness
        return particle.position ** 2 + particle.position ** 2 + 1

    def set_p_best(self):
        for particle in self.particles:
            possible_p_best = self.fitness_function_sphere(particle)
            if particle.p_best_value > possible_p_best:
                particle.p_best_value = possible_p_best
                particle.p_best_position = particle.position

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

            e1 = random.random()
            e2 = random.random()
            v0 = particle.velocity

            f_velocity = v0 + c1 * e1 * (particle.p_best_position - particle.position) + c2 * e2 * \
                (self.g_best_position - particle.position)
            particle.velocity = f_velocity
            particle.move()


n_particles = int(input("Qtd. partículas: "))
n_dimensions = int(input("Qtd. dimensões: "))
n_iterations = int(input("Qtd. iterações: "))
c = float(input("Valor para as constantes c1 e c2: "))

particles = [Particle(n_dimensions) for _ in range(n_particles)]
space = Space(n_dimensions, particles)

iterations = 0
while iterations < n_iterations:
    space.move_particle()
    space.set_p_best()
    space.set_g_best()

    space.print_particles()
    iterations += 1

print("The best solution is: ", space.g_best_value, " in ", space.g_best_position, " in n_iterations: ", iterations)

# P = Particle(30)
# print(P.position)
# print(P.velocity)
# P.move()
# print(P.position)
