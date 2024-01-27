"""
    Simulates the spread of a virus within a population.
    Name: Covid-19 Simulation
    Abdullah Saad
    asaad02@uoguelph.ca
"""
" import the modules"
import sys
import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

###############################
#####     Functions       #####
###############################

###################################################
## initialized the population for the simulation ##
###################################################

def population(population_size, mean_age, max_age,x_bounds=[], y_bounds=[]):
    """
    Initializes the population for the simulation.

    Args:
        population_size (int): Number of individuals in the population.
        mean_age (int): Mean age of the population.
        max_age (int): Maximum age of individuals.
        x_bounds (list): Bounds for the x-coordinate (optional).
        y_bounds (list): Bounds for the y-coordinate (optional).

    Returns:
        numpy.ndarray: A matrix representing the initialized population.
    """

    #This matrix is a 14 columns * population size Using numpy package 

    # initialize population matrix 15 * the population size 
    #numpy. zeros() function returns a new array element's value as 0.
    population = np.zeros((population_size, 15))
    # initialize unique IDs for each population
    population[:, 0] = [i for i in range(population_size)]

    # initialize random coordinates 
    #############################################################
    #####     Draw samples from a uniform distribution      #####
    #############################################################

    #numpy.random. uniform (low=0.0, high=1.0, size=None)
    # x coodinate and y coodinate
    population[:, 1] = np.random.uniform(low=x_bounds[0] + 0.01, high=x_bounds[1] - 0.01,size=(population_size))
    population[:, 2] = np.random.uniform(low=y_bounds[0] + 0.01, high=y_bounds[1] - 0.01,size=(population_size))

    ##############################################################################
    #####     Draw random samples from a normal (Gaussian) distribution     ######
    ##############################################################################

    # initialize random headings in x direction and y direction 
    #loc is (mean),scale is (standard deviation)
    population[:, 3] = np.random.normal(loc=0, scale=1 / 3,size=(population_size))
    population[:, 4] = np.random.normal(loc=0, scale=1 / 3,size=(population_size))
    # initialize random speeds
    population[:, 5] = np.random.normal(loc=0.01,scale=0.01 / 3)
    # initalize ages of each population 
    # Integer (-2147483648 to 2147483647)
    population[:, 7] = np.int32(np.random.normal(loc=mean_age,scale=(max_age - mean_age) / 3,size=(population_size)))
    # clip Elements less then 0 age 
    population[:, 7] = np.clip(population[:, 7], a_min=0,a_max=max_age) 
    # recovery_vector
    population[:, 9] = np.random.normal(loc=0.5, scale=0.5 / 3, size=(population_size,))

    # return the array matrix 
    return population


###################################################
## initialized the bounds for the simulation ##
###################################################
def bounds(population, xbounds, ybounds):

    """
    Updates the headings of populations when they move out of bounds.

    Args:
        population (numpy.ndarray): Matrix representing the population.
        x_bounds (list): Bounds for the x-coordinate.
        y_bounds (list): Bounds for the y-coordinate.

    Returns:
        numpy.ndarray: Updated population matrix.
    """

    #########################################################
    #####     Normal (Gaussian) Distribution       ##########
    #########################################################

    #update the heading of the out of the bounds 
    new = population[:,3][(population[:,1] <= xbounds[:,0]) &(population[:,3] < 0)].shape
    population[:,3][(population[:,1] <= xbounds[:,0]) &(population[:,3] < 0)] = np.clip(np.random.normal(loc = 0.5, scale = 0.5/3,size = new),a_min = 0.05, a_max = 1)
    
    new = population[:,3][(population[:,1] >= xbounds[:,1]) & (population[:,3] > 0)].shape
    population[:,3][(population[:,1] >= xbounds[:,1]) &(population[:,3] > 0)] = np.clip(-np.random.normal(loc = 0.5, scale = 0.5/3,size = new),a_min = -1, a_max = -0.05)
    
    
    ##############################################################################
    #update y heading
    new = population[:,4][(population[:,2] <= ybounds[:,0]) &(population[:,4] < 0)].shape
    population[:,4][(population[:,2] <= ybounds[:,0]) &(population[:,4] < 0)] = np.clip(np.random.normal(loc = 0.5, scale = 0.5/3,size = new),a_min = 0.05, a_max = 1)

    new = population[:,4][(population[:,2] >= ybounds[:,1]) &(population[:,4] > 0)].shape
    population[:,4][(population[:,2] >= ybounds[:,1]) &(population[:,4] > 0)] = np.clip(-np.random.normal(loc = 0.5, scale = 0.5/3,size = new),a_min = -1, a_max = -0.05)

    return population
###################################################
## updates heading and speed##
###################################################
def randoms(population,population_size, heading=0.02):

    """
    Updates the headings and speeds of populations randomly.

    Args:
        population (numpy.ndarray): Matrix representing the population.
        heading (float): Maximum heading change for update.

    Returns:
        numpy.ndarray: Updated population matrix.
    """
    random = np.random.random(size=(population_size,))
    new = random[random <= heading].shape
    population[:, 3][random <= heading] = np.random.normal(loc=0,scale=1 / 3,size=new)
    # y
    random = np.random.random(size=(population_size,))
    new = random[random <= heading].shape
    population[:, 4][random <= heading] = np.random.normal(loc=0,scale=1 / 3,size=new)
    # randomize speeds
    random = np.random.random(size=(population_size,))
    new = random[random <= heading].shape
    population[:, 5][random <= heading] = np.random.normal(loc=0.01,scale=0.01 / 3,size=new)
    return population
###################################################
## Finding a new infection##
###################################################
"""
    Simulates the spread of infection within the population.

    Args:
        population (numpy.ndarray): Matrix representing the population.
        population_size (int): Number of individuals in the population.
        infection_range (float): Range within which infection can spread.
        infection_chance (float): Probability of infection upon contact.
        frame (int): Current frame or time step in the simulation.

    Returns:
        numpy.ndarray: Updated population matrix with new infections marked.
    """

def infection(population,population_size, infection_range, infection_chance, frame):
    
    #find new infections
    infected= population[population[:,6] == 1]

    new_infection = []

    #speed up the disease when less half infected before the lockdown 
    if len(infected) < (population_size // 2):
        for patient in infected:
            #define infection zone for patient
            disease_zone = [patient[1] - infection_range, patient[2] - infection_range,patient[1] + infection_range, patient[2] + infection_range]

            #find healthy people surrounding infected patient
            person = np.int32(population[:,0][(disease_zone[0] < population[:,1]) & (population[:,1] < disease_zone[2]) &(disease_zone[1] < population [:,2]) & (population[:,2] < disease_zone[3]) &(population[:,6] == 0)])
            for i in person:
                #roll die to see if healthy person will be infected
                #a = infection_chance
                b = np.random.random()
                if b <= infection_chance:
                    population[i][6] = 1
                    population[i][8] = frame
                    new_infection.append(i)
    else:
        #more then half are infected 
        healthy1 = population[population[:,6] == 0]
        sick = population[population[:,6] == 1]
        for person in healthy1:
            #define infecftions range around healthy person
            infection_zone = [person[1] - infection_range, person[2] - infection_range,person[1] + infection_range, person[2] + infection_range]
            if person[6] == 0: #if person is not already infected, find if infected are nearby
                #find infected nearby healthy person
                people = len(sick[:,6][(infection_zone[0] < sick[:,1]) & (sick[:,1] < infection_zone[2]) &(infection_zone[1] < sick [:,2]) & (sick[:,2] < infection_zone[3]) &(sick[:,6] == 1)])
                if people > 0:
                    infection_chance = 0.01
                    a = infection_chance
                    b = a * people
                    c = np.random.random() 
                    if c <= b:
                        #roll die to see if healthy person will be infected
                        population[np.int32(person[0])][6] = 1
                        population[np.int32(person[0])][8] = frame
                        new_infection.append(np.int32(person[0]))

    return population
###################################################
## Finding for recovering and daying ##
###################################################

def recovering_or_dying(population, frame, recovery_duration, mortality_chance):

    """
    Simulates recovery and mortality among the sick population.

    Args:
        population (numpy.ndarray): Matrix representing the population.
        frame (int): Current frame or time step in the simulation.
        recovery_duration (tuple): Tuple of recovery duration range (min, max).
        mortality_chance (float): Probability of mortality for sick individuals.

    Returns:
        numpy.ndarray: Updated population matrix with recovered or deceased individuals.
    """
    #find sick people
    sick = population[population[:,6] == 1]

    #define vector of how long everyone has been sick
    illness = frame - sick[:,8]
    recovery_duration = (50, 300) 
    recovery = (illness - recovery_duration[0]) / np.ptp(recovery_duration)
    recovery = np.clip(recovery, a_min = 0, a_max = None)

    #update states of sick people 
    person = sick[:,0][recovery >= sick[:,9]]

    cured = []
    died = []
    
    # decide whether to die or recover
    for i in person:
        if np.random.random() <= mortality_chance:
            # die
            sick[:, 6][sick[:, 0] == i] = 3
            died.append(np.int32(sick[sick[:, 0] == i][:, 0][0]))
        else:
            # recover 
            sick[:, 6][sick[:, 0] == i] = 2
            cured.append(np.int32(sick[sick[:, 0] == i][:, 0][0]))

    if len(died) > 0:
        sys.stdout.write('\n')
        sys.stdout.write('Days [%i] These people died by Covid-19: [%s]' % (frame, len(died)))
    if len(cured) > 0:
        sys.stdout.write('\n')
        sys.stdout.write('Days [%i] These people got recovered: [%s]' % (frame, len(cured)))

    # put array back into population
    population[population[:, 6] == 1] = sick

    return population
###################################################
##             corona     Simulation            ##
##################################################
def Corona_simulation(frame, population, destinations, population_size,wander_factor,speed,infectious,first_day_of_infection,lockdown ,lockdown_percentage,complying,infection_chance =0.4, recovery_duration=(50, 200), mortality_chance=0.08,infection_range=0.01,infected_plot=[]):
    
    """
    Simulates the spread of a virus within a population.

    Args:
        frame (int): Current frame or time step in the simulation.
        population (numpy.ndarray): Matrix representing the population.
        destinations (list): List of destination coordinates for individuals.
        population_size (int): Number of individuals in the population.
        wander_factor (float): Factor controlling wander behavior.
        speed (float): Speed of individuals.
        infectious (list): List of infectious individuals.
        first_day_of_infection (int): Day when the first infection occurs.
        lockdown (bool): Indicates if a lockdown is in effect.
        lockdown_percentage (float): Percentage of population affected by lockdown.
        complying (float): Percentage of the population complying with lockdown.
        infection_chance (float): Probability of infection upon contact (default 0.4).
        recovery_duration (tuple): Tuple of recovery duration range (min, max) (default (50, 200)).
        mortality_chance (float): Probability of mortality for sick individuals (default 0.08).
        infection_range (float): Range within which infection can spread (default 0.01).
        infected_plot (list): List to store the number of infected individuals for each frame.

    Returns:
        None
    """
    # which day the first infection 
    if frame == first_day_of_infection:
        population[0][6] = 1
        population[0][8] = 75
    # check for interacting 
    interacting =len(population[population[:,13] != 0])
    if interacting > 0 and len(population[population[:,14] == 0]) > 0:
        population = motion(population, destinations)
        population = checking(population, destinations,wander_factor,speed)
    # if infected stay at the location 
    if interacting > 0 and len(population[population[:,14] == 1]) > 0:
        population = staying(population,destinations,wander_factor)
    if len(population[:,13] == 0) > 0:
        #x_bounds = np.array([xbounds[0.03], xbounds[1.08 ]] * len(population[population[:,13] == 0]))
        #y_bounds = np.array([ybounds[0.00], ybounds[0.96]] * len(population[population[:,13] == 0]))
        xbounds=[0.1, 1.1]
        ybounds=[0.02, 0.98]
        x_bounds = np.array([[xbounds[0] + 0.02, xbounds[1] - 0.02]] * len(population))
        y_bounds = np.array([[ybounds[0] + 0.02, ybounds[1] - 0.02]] * len(population))
        population[population[:,13] == 0] = bounds(population[population[:,13] == 0], x_bounds, y_bounds)
        
    # if lockdown
    if lockdown:
        if len(infectious) == 0:
            k = 0
        else:
            #The max() function returns the item with the highest value
            k = np.max(infectious)
        lockdown_on = (population_size * lockdown_percentage)
        a = int(len(population))
        if len(population[population[:,6] == 1]) > len(population) - population_size  or k >= (len(population) - population_size):
            #reduceing  speed of all members of society
            population[:,5] = np.clip(population[:,5], a_min = None, a_max = 0.001)
            #lockdown speed is 1 for not complying
            lockdown_speed = np.zeros((population_size))
            lockdown_speed[np.random.uniform(population_size) >= ( 1- complying )] = 1
            #set speeds of complying people to 0
            population[:,5][lockdown_speed == 0] = 0
        else:
            population = randoms(population, population_size)
    else:
        population = randoms(population, population_size)

    #0 speed for the dead one 
    population[:,3:5][population[:,6] == 3] = 0

    # update positions
    population[:,1] = population[:,1] + (population[:,3] * population[:,5])
    population[:,2] = population[:,2] + (population [:,4] * population[:,5])
    
    # find new infections
    population = infection(population,population_size, infection_range, infection_chance, frame)
    # recover and die
    population = recovering_or_dying(population, frame, recovery_duration, mortality_chance)
    # find new infections
    infected_plot.append(len(population[population[:, 6] == 1]))


###################################################
##             figure      build             ##
##################################################
def figure_build(xbounds, ybounds,simulation_steps,population_size):

    """
    Builds the figure for visualization.

    Returns:
        matplotlib.figure.Figure: The created figure.
        matplotlib.gridspec.GridSpec: The grid specification for subplots.
        matplotlib.axes._subplots.AxesSubplot: The main simulation subplot.
        matplotlib.axes._subplots.AxesSubplot: The subplot for the number of infected individuals.
        matplotlib.axes._subplots.AxesSubplot: The subplot for the state of infected individuals.
    """
    
    fig = plt.figure(figsize=(8, 12))
    spec = fig.add_gridspec(ncols=4, nrows=4)

    ax1 = fig.add_subplot(spec[:-1, 1:])
    plt.title('Infection Simulation')
    plt.xlim(xbounds[0], xbounds[1])
    plt.ylim(ybounds[0], ybounds[1])

    ax2 = fig.add_subplot(spec[-1, 1:])
    ax2.set_title('Number of Infected')
    ax2.set_xlim(0, simulation_steps)
    ax2.set_ylim(0, population_size + 100)

    ax3 = fig.add_subplot(spec[:-1, 0])
    ax3.set_title('Infected')
    ax3.set_xlim([0, simulation_steps])
    ax3.set_ylim([0, 100])

    return fig, spec, ax1, ax2, ax3


###################################################
##             visualisation of simulation       ##
##################################################
def visualisation( frame,population, susceptible,infectious,recovered,fatalities1,fig, spec, ax1, ax2,ax3,inf_sum,hea_sum,rec_sum,dead_sum, simulation_steps,population_size):
    

    """
    Visualizes the simulation at each frame.

    Args:
        frame (int): Current frame or time step in the simulation.
        population (numpy.ndarray): Matrix representing the population.
        susceptible (list): List tracking the number of susceptible individuals over time.
        infectious (list): List tracking the number of infectious individuals over time.
        recovered (list): List tracking the number of recovered individuals over time.
        fatalities (list): List tracking the number of fatalities over time.
        fig (matplotlib.figure.Figure): The figure for visualization.
        spec (matplotlib.gridspec.GridSpec): The grid specification for subplots.
        ax1 (matplotlib.axes._subplots.AxesSubplot): The main simulation subplot.
        ax2 (matplotlib.axes._subplots.AxesSubplot): The subplot for the number of infected individuals.
        ax3 (matplotlib.axes._subplots.AxesSubplot): The subplot for the state of infected individuals.
        inf_sum (list): List tracking the cumulative number of infected individuals.
        hea_sum (list): List tracking the cumulative number of healthy individuals.
        rec_sum (list): List tracking the cumulative number of recovered individuals.
        dead_sum (list): List tracking the cumulative number of fatalities.

    Returns:
        None
    """

    x_plot = [0, 1.1]
    y_plot = [0, 1]
    world_size = [2, 2] #x and y sizes of the world
    ax1.clear()
    ax2.clear()
    #ax3.clear()
    ax1.set_xlim(x_plot[0], x_plot[1])
    ax1.set_ylim(y_plot[0], y_plot[1])

    # plot population segments
    healthy = population[population[:,6] == 0][:,1:3]
    ax1.scatter(healthy[:, 0], healthy[:, 1], color='Green', s=2, label='healthy')

    infected = population[population[:, 6] == 1][:, 1:3]
    ax1.scatter(infected[:, 0], infected[:, 1], color='red', s=2, label='infected')

    immune = population[population[:, 6] == 2][:, 1:3]
    ax1.scatter(immune[:, 0], immune[:, 1], color='Blue', s=2, label='immune')

    fatalities = population[population[:, 6] == 3][:, 1:3]
    ax1.scatter(fatalities[:, 0], fatalities[:, 1], color='black', s=2, label='dead')

    # add text descriptors
    ax1.text(x_plot[0],y_plot[1] + ((y_plot[1] - y_plot[0]) / 100) ,'Day: [%i], Population: [%i], Healthy: [%i] infected: [%i] Immune: [%i] Fatalities: [%i]' % (frame,len(population),len(healthy),len(infected),len(immune),len(fatalities)),fontsize=6)
    ax1.legend(loc='best', fontsize=10)
    
    ax2.set_title('susceptible/infectious/recovered/fatalities')
    #ax2.text(0, population_size* 0.05,fontsize=6, alpha=0.5)
    #ax2.set_xlim(0, simulation_steps)
    ax2.set_ylim(0, population_size + 200)
    ax2.plot(susceptible, color='Green', label='susceptible')
    ax2.plot(infectious, color='Red', label='infectious')
    ax2.plot(recovered, color='Blue', label='recovered')
    ax2.plot(fatalities1, color='black', label='fatalities')
    #ax2.legend(loc='best', fontsize=6)
    


    ax2.set_ylim(0, population_size + 200)

    
    ax3.fill_between(frame,hea_sum, color='Green',alpha=0.5)
    ax3.fill_between(frame,inf_sum,color='red')
    ax3.fill_between(frame,rec_sum,color='blue')
    ax3.fill_between(frame,dead_sum,color='black')

    
    #beautify
    plt.tight_layout()

    
    #initialise
    #plt.show()
    plt.draw()
    plt.pause(0.0001)

###################################################
##             counting function                   ##
##################################################
# count susceptible/infectious/recovered/fatalities1 and store them in array for plotter
def counting(population,population_size,susceptible,infectious,recovered,fatalities1,inf_sum,hea_sum,rec_sum,dead_sum,simulation_steps,c,frame):
    #current state 0=healthy// 1=sick// 2=immune//3=dead//4=immune but infectious
    """
    Counts the number of individuals in different states and stores them in arrays for plotting.

    Args:
        population (numpy.ndarray): Matrix representing the population.
        population_size (int): Total population size.
        susceptible (list): List tracking the number of susceptible individuals over time.
        infectious (list): List tracking the number of infectious individuals over time.
        recovered (list): List tracking the number of recovered individuals over time.
        fatalities1 (list): List tracking the number of fatalities over time.
        inf_sum (list): List tracking the cumulative number of infectious individuals.
        hea_sum (list): List tracking the cumulative number of healthy individuals.
        rec_sum (list): List tracking the cumulative number of recovered individuals.
        dead_sum (list): List tracking the cumulative number of fatalities.
        simulation_steps (int): Total number of simulation steps.
        c (int): Counter variable for simulation steps.
        frame (int): Current frame or time step in the simulation.

    Returns:
        None
    """
    population_size = population.shape[0]
    susceptible.append(len(population[population[:,6] == 0]))
    infectious.append(len(population[population[:,6] == 1]))
    recovered.append(len(population[population[:,6] == 2]))
    fatalities1.append(len(population[population[:,6] == 3]))
    ###########

    hea_sum.append(len(population[population[:,6] == 0]))

    inf_sum.append((len(population[population[:,6] == 1])/ 100) * 4)

    rec_sum.append((len(population[population[:,6] == 2])/ 100) * 4)

    dead_sum.append((len(population[population[:,6] == 3])/ 100) * 4)
    
    reinfect = False 

    ##########################
    # if reinfect 
    if reinfect:
        # decrease the susceptible 
        susceptible.append(population_size - (infectious[-1] +fatalities1[-1]))
    else:
        # decrease susceptible 
        susceptible.append(population_size - (infectious[-1] +recovered[-1] +fatalities1[-1]))


###################################################
##             moting function                   ##
##################################################
def motion(population, destinations):
    """
    Move individuals in the population towards their respective destinations.

    Args:
        population (numpy.ndarray): Matrix representing the population.
        destinations (numpy.ndarray): Matrix representing the destinations for each individual.

    Returns:
        numpy.ndarray: Updated population matrix with new positions and headings.
    """
    #unique() function is used to find the unique elements of an array
    motion = np.unique(population[:,13][population[:,13] != 0])
    for i in motion :
        motion_x = destinations[:,int((i - 1) * 2)]
        motion_y = destinations[:,int(((i - 1) * 2) + 2)]
        #1 : current x coordinate
        #2 : current y coordinate
        x_coordinate = population[:,1]
        y_coordinate = population[:,1]
        #new move
        x_heading=  motion_x - x_coordinate
        y_heading = motion_y - y_coordinate
        #3 : current heading in x direction
        #4 : current heading in y direction
        population[:,3][(population[:,13] == i) & (population[:,14] == 0)] = x_heading[(population[:,13] == i) & (population[:,14] == 0)]
        population[:,4][(population[:,13] == i) &(population[:,14] == 0)] = y_heading[(population[:,13] == i) &(population[:,14] == 0)]
        #change speed to 0.02
        population[:,5][(population[:,13] == i) &(population[:,14] == 0)] = 0.02
    return population
###################################################
##             checking function                   ##
##################################################
def checking (population, destinations,wander_factor,speed):
    """
    Check for individuals who have arrived at their destinations and update their properties.

    Args:
        population (numpy.ndarray): Matrix representing the population.
        destinations (numpy.ndarray): Matrix representing the destinations for each individual.
        wander_factor (float): Factor controlling wander range.
        speed (float): Current speed of individuals.

    Returns:
        numpy.ndarray: Updated population matrix.
    """
    # unique() function is used to find the unique elements of an array
    motion = np.unique(population[:,13][(population[:,13] != 0)])
    for i in motion:
        motion_x = destinations[:,int((i - 1) * 2)]
        motion_y = destinations[:,int(((i - 1) * 2) + 1)]
        #see who arrived at destination and filter out who already was there
        #The abs() function returns the absolute value of the specified number.
        arrived = population[(np.abs(population[:,1] - motion_x) < (population[:,11] * wander_factor)) & (np.abs(population[:,2] - motion_y) < (population[:,12] * wander_factor)) &(population[:,14] == 0)]
        if len(arrived) > 0:
            #mark those as arrived
            arrived[:,14] = 1
            #insert random headings and speeds for those at destination
            # how many population arrives
            arrives =len(arrived)
            update_heading  = 1
            update_speed = 1
            arrived = randoms(arrived, arrives)
            #insert the update  population
            population[(np.abs(population[:,1] - motion_x) < (population[:,11] * wander_factor)) & (np.abs(population[:,2] - motion_y) < (population[:,12] * wander_factor)) &(population[:,12] == 0)] = arrived
    return population
###################################################
##             staying destination function                   ##
##################################################
def staying(population,destinations,wander_factor):
    """
    Keep individuals who have arrived at their destinations within their wander ranges.

    Args:
        population (numpy.ndarray): Matrix representing the population.
        destinations (numpy.ndarray): Matrix representing the destinations for each individual.
        wander_factor (float): Factor controlling wander range.

    Returns:
        numpy.ndarray: Updated population matrix.
    """
    #Function that keeps those who have been marked as arrived at their
    #destination within their respective wander ranges
    #unique() function is used to find the unique elements of an array
    motion = np.unique(population[:,13][(population[:,13] != 0) &(population[:,14] == 1)])
    for i in motion:
        motion_x = destinations[:,int((i - 1) * 2)][(population[:,14] == 1) &(population[:,13] == i)]
        motion_y = destinations[:,int(((i - 1) * 2) + 1)][(population[:,14] == 1) &(population[:,13] == i)]
        arrives = population[(population[:,14] == 1) &(population[:,13] == i)]

        new = arrives[:,3][arrives[:,1] > (motion_x + (arrives[:,11] * wander_factor))].shape
        arrives[:,3][arrives[:,1] > (motion_x + (arrives[:,11] * wander_factor))] = -np.random.normal(0.5,0.5 / 3,new)

        #x < (motion - wander), set heading (+)
        new = arrives[:,3][arrives[:,1] < (motion_x - (arrives[:,11] * wander_factor))].shape
        arrives[:,3][arrives[:,1] < (motion_x - (arrives[:,11] * wander_factor))] = np.random.normal(0.5,0.5 / 3,new)
        #y > motion + wander, set heading (-)
        new = arrives[:,4][arrives[:,2] > (motion_y + (arrives[:,12] * wander_factor))].shape
        arrives[:,4][arrives[:,2] > (motion_y + (arrives[:,12] * wander_factor))] = -np.random.normal(0.5,0.5 / 3,new)
        #y <  motion - wander, heading (+)
        new = arrives[:,4][arrives[:,2] < (motion_y - (arrives[:,12] * wander_factor))].shape
        arrives[:,4][arrives[:,2] < (motion_y - (arrived[:,12] * wander_factor))] = np.random.normal(0.5, 0.5 / 3,new)
        #slowing speed
        arrives[:,5] = np.random.normal(0.005,0.005 / 3,arrives[:,5].shape)
        # install it into population 
        #reinsert into population
        population[(population[:,14] == 1) &(population[:,13] == i)] = arrives
    return population
###################################################
##             reset function                   ##
##################################################
def reset(i,population,destinations,population_size,mean_age,max_age,xbounds,ybounds):

    """
    Reinitialize the simulation by resetting the population and destinations.

    Args:
        i (int): Current frame or day.
        population (numpy.ndarray): Matrix representing the population.
        destinations (numpy.ndarray): Matrix representing the destinations for each individual.
        population_size (int): Total population size.
        mean_age (int): Mean age of the population.
        max_age (int): Maximum age of the population.
        xbounds (list): X-coordinate bounds of the world.
        ybounds (list): Y-coordinate bounds of the world.

    Returns:
        int: Updated frame or day (set to 0).
        numpy.ndarray: Reinitialized population matrix.
        numpy.ndarray: Initialized destination matrix.
    """

    #reinitialize the population 
    population = population(population_size, mean_age, max_age, xbounds, ybounds)
    # reset the frame
    i = 0 
    # initialize destination matrix
    destinations = np.zeros((population_size, 1 * 2))
    



def main():
    wander_factor = 1        # area around the boundary
    speed = 0.05             # speed of population
    world_size = [4, 4]      # x and y sizes of the world
    x_plot = [0, world_size[0]]  # size of the simulated world in coordinates
    y_plot = [0, world_size[1]]



    xbounds = [x_plot[0] + 0.02, x_plot[1] - 0.02]
    ybounds = [x_plot[0] + 0.02, x_plot[1] - 0.02]
    infection_range = 0.25

    susceptible = []
    infectious = []
    recovered = []
    fatalities1 = []

    inf_sum = []
    hea_sum = []
    rec_sum = []
    dead_sum = []
    cumulative_sum = []
    c = []

    

    simulation_steps = 2000  # Days
    xbounds = [0, 1]
    ybounds = [0, 1]
    reinfect = False

    population_size = 2000
    mean_age = 45
    max_age = 105
    mean_speed = 0.01
    std_speed = 0.01 / 3
    disease_range = 1
    infection_chance = 0.6
    recovery_duration = (50, 100)
    mortality_chance = 0.08
    lockdown_percentage = 0
    complying = 0

    #fig, spec, ax1, ax2, ax3 = figure_build(xbounds, ybounds,simulation_steps,population_size)
    infected_plot = []
    fatalities_plot = []

    print('Welcome To Covid-19 Simulation')
    first_day_of_infection = int(input("Please enter the first day of infection (between day 1 - 500): "))
    print('Covid-19 Scenario')
    lockdown = input("Lockdown scenario True/False: ").lower()
    
    if lockdown == "true":
        lockdown = True
    elif lockdown == "false":
        lockdown = False
        lockdown_percentage = 0
        complying = 0
    else:
        print("Error: Answer must be True or False")
        lockdown = input("Lockdown scenario True/False: ").lower()
        lockdown_percentage = 0
        complying = 0

    if lockdown:
        lockdown_percentage = float(input("Lockdown scenario after what percentage of the population gets infected (0-100): (Please enter without %): ")) / 100
        complying = float(input("Percentage of people complying with the lockdown (0-100): (Please enter without %): ")) / 100

    self_isolation = input("Self-isolation scenario True/False: ").lower()
    
    if self_isolation == "true":
        speed = 0.01
    elif self_isolation == "false":
        speed = 1

    Covid_population = population(population_size, mean_age, max_age, xbounds, ybounds)
    destinations = np.zeros((population_size, 1 * 2))
    fig, spec, ax1, ax2, ax3 = figure_build(xbounds, ybounds,simulation_steps,population_size)

    fargs = (Covid_population, destinations, population_size, disease_range, infection_chance, recovery_duration, mortality_chance, xbounds, ybounds, complying)
    animation = FuncAnimation(fig, Corona_simulation, fargs=fargs, frames=simulation_steps, interval=33)

    for i in range(simulation_steps):
        Corona_simulation(i, Covid_population, destinations, population_size, wander_factor, speed, infectious, first_day_of_infection, lockdown, lockdown_percentage, complying)
        visualisation(i, Covid_population, susceptible, infectious, recovered, fatalities1, fig, spec, ax1, ax2, ax3, inf_sum, hea_sum, rec_sum, dead_sum, simulation_steps, population_size)
        counting(Covid_population, population_size, susceptible, infectious, recovered, fatalities1, inf_sum, hea_sum, rec_sum, dead_sum, simulation_steps, c, i)
        healthy = Covid_population[Covid_population[:, 6] == 0][:, 1:3]
        infected = Covid_population[Covid_population[:, 6] == 1][:, 1:3]
        immune = Covid_population[Covid_population[:, 6] == 2][:, 1:3]
        fatalities = Covid_population[Covid_population[:, 6] == 3][:, 1:3]
        
        if len(immune) + len(fatalities) == population_size:
            reset(i, Covid_population, destinations, population_size, mean_age, max_age, xbounds, ybounds)
        
        sys.stdout.write('\r')
        sys.stdout.write('Days: %i , total: [%i], healthy: [%i] infected: [%i] immune: [%i] fatalities: [%i]' % (i, len(Covid_population), len(healthy), len(infected), len(immune), len(fatalities)))
        sys.stdout.write('\r')

if __name__ == '__main__':
    main()