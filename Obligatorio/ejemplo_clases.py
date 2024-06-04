from abc import ABC, abstractmethod


def suma(a,b):
    return a+b


class Agent(ABC):
    def __init__(self, nacim, altura):
        self.nacim = nacim
        self.altura = altura
        self.lentes = True

    def rango_edad(self):
        if self.nacim > 1990:
            self.edad = "Joven"
        else:
            self.edad = "Viejo"
        return self.edad
    
    @abstractmethod
    def nacim_altura(): # no hace nada, lo define en los demas agentes que lo instancien
        pass

    @abstractmethod
    def altura_nacim_peso(self): # no hace nada, lo define en los demas agentes
        pass


class DQN_Agent(Agent):
    def __init__(self, nacim, altura, func):
        super().__init__(nacim, altura)
        self.func = func
    
    def estatura(self):
        if self.altura > 170:
            self.alto = "Altisimo"
        else:
            self.alto = "Bajisimo"
        return self.alto
    
    def nacim_altura(self):
        return print(f'Nacimiento: {self.nacim}, Altura: {self.altura}')
    
    def altura_nacim_peso(self):
        pass # no hace nada porque no tiene 'peso' como parametro

    
class DoubleDQN_Agent(DQN_Agent):
    def __init__(self, nacim, altura, peso):
        super().__init__(nacim, altura)
        self.peso = peso
    
    def kgs(self):
        if self.peso > 90:
            self.forma = "Gordo"
        else:
            self.forma = "Flaco"
        return self.forma
    
    def altura_nacim_peso(self):
        return print(f'Altura: {self.altura}, Nacimiento: {self.nacim}, Forma: {self.peso}')


persona1 = Agent(1993, 160) # ahora la clase es abstracta, por lo que no la puedo instanciar directamente

persona2 = DQN_Agent(1987, 180, suma)
persona2.func(20, 33)
persona2.lentes
persona2.rango_edad()
persona2.estatura()
persona2.nacim_altura()

persona3 = DoubleDQN_Agent(1991, 174, 74)
persona3.lentes
persona3.rango_edad()
persona3.estatura()
persona3.kgs()
persona3.nacim_altura()
persona3.altura_nacim_peso()

