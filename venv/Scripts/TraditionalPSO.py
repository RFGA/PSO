import numpy as np
import random


class Particle:
    # pbest is the fitness of particle

    def __init__(self, qtd_dimension):

        # o valor poderia ser o retorno da função fitness para a posição inicial gerada
        # self.p_best_value = Space.fitness_function_sphere(self) (falta testar...)
        self.p_best_value = float('inf')
        self.velocity = np.array([1] * qtd_dimension)

        self.position = []
        for i in range(qtd_dimension):
            self.position.append(random.uniform(-100, 100))

        self.position = np.array(self.position)
        self.p_best_position = self.position

    def move(self):
        self.position = self.position + self.velocity


class Space:
    def __init__(self, qtd_dimension, n_particles):
        self.g_best_value = float['inf']
        self.g_best_position = []
        self.n_particles = n_particles
        self.particles = []
        for i in range(qtd_dimension):
            self.g_best_position.append(random.uniform(-100, 100))

    @staticmethod
    def fitness_function_sphere(particle):
        value = 0
        for x in particle.position:
            value = (x ** 2) + value
        return value

    def set_p_best(self):
        pass

    def set_g_best(self):
        pass

    def move_particle(self):
        pass


P = Particle(30)
print(P.position)
print(P.velocity)
P.move()
print(P.position)
