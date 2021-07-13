#Deap libraries
from deap import base
from deap import creator
from deap import tools
import elitism

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

raw = pd.read_csv('http://hilpisch.com/pyalgo_eikon_eod_data.csv',index_col=0, parse_dates=True).dropna() 
data = pd.DataFrame(raw["GLD"])
w = data["GLD"].to_numpy()



def rolling(N, Array):
    """
    Basic rolling function similiar to pandas.DataFrame.rolling
    N: Size of the moving window if a float is provided it would be rounded
    Array: Numpy array or pandas Series 
    """
    if isinstance(Array, np.ndarray) or isinstance(Array, pd.Series):
        lista=[0]
        N = round(N)
        #List with Nans 
        moving_aves =  [np.nan]*(N-1)
    
        for i, x in enumerate(Array, 1):
            lista.append(lista[i-1] + x)
            if i>=N:
                try:
                    moving_ave = (lista[i] - lista[i-N])/N
                    moving_aves.append(moving_ave)
                except:
                    moving_aves.append(np.nan)
        return moving_aves
    
    else:
        raise TypeError("Expected Array to be a numpy narray or panadas Series")

def _MM(individual):

    """
    Provides Moving Average calculation for Numpy array or Pandas Series 
    between to time windows
    """
    #Individuals corresponding to varriables in MM
    short = individual[0]
    long = individual[1]
    
    #Rolling for short and long signal
    a = rolling(short, w)  
    b = rolling(long, w)  
            
    s = []
    e = []
            
    for z in range(len(w)):
        if a[z] > b[z]:
            s.append(+1)
        else:
            s.append(-1)
      
    s = np.asarray(s)
            
    n = np.log(w/np.roll(w,1))   
            
    for z in range(len(w)):
        e.append(np.roll(s[z],1)*n[z])
    e = np.asarray(e)
            
    rendimiento = np.sum(e)
            
    riesgo = np.std(e)
    fitness = rendimiento - riesgo
            
            
    return rendimiento,


def feasible(individual):
    """Feasibility function for the individual. Returns True if feasible False
    otherwise"""
    if 0 < individual[0] < 100 and 100< individual[1] <250 :
        return True
    
    return False


#Constants (Change this values)
Dimensions = 2
Bound_low, Bound_up = 0,250

Population_size = 50
P_crossover = 0.9
P_mutation = 0.5
Max_generations = 10
Hall_of_fame_size = 30
Crowding_factor = 20.0


#Deap

toolbox = base.Toolbox()

#Objective -> Maximize

creator.create("Fitness_Max", base.Fitness, weights=(1.0,))

#Population

creator.create("Individual", list, fitness=creator.Fitness_Max)

def randomFloat(low, up):
    return [random.uniform(l, u) for l, u in zip([low] * Dimensions, [up] * Dimensions)]

toolbox.register("attrFloat", randomFloat, Bound_low, Bound_up)

#Create individual

toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.attrFloat)

#Create Generation
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

#Objective Function and Penalizing function

toolbox.register("evaluate", _MM)
toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 0))

#Values for operators
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=Bound_low, up=Bound_up, eta=Crowding_factor)
toolbox.register("mutate", tools.mutPolynomialBounded, low=Bound_low, up=Bound_up, eta=Crowding_factor, indpb=1.0/Dimensions)

#Genetic algorithm main flow 
def main():

    #First Generation
    population = toolbox.populationCreator(n=Population_size)

    #Prepare information
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    #Hall of fame
    hof = tools.HallOfFame(Hall_of_fame_size)

    # perform the Genetic Algorithm flow with elitism:
    population, logbook = elitism.eaSimpleWithElitism(population, toolbox, cxpb=P_crossover, mutpb=P_mutation,
                                              ngen=Max_generations, stats=stats, halloffame=hof, verbose=True)

    # print info for best solution found:
    best = hof.items[0]
    print("-- Best Individuals = ", np.round(best))
    print("-- Best Fitness = ", np.round(best.fitness.values[0],2))

    # extract statistics:
    minFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    # plot statistics:
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average fitness over Generations')

    plt.show()

if __name__ == "__main__":
    main()