'''
Name : Abdullah Saad
Email : asaad02@uoguelph.ca
Covid-19 Simulation
install :  pip3 install matplotlib , pip3 install numpy
Run : python3 old.py
'''

# Import necessary libraries
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Function to initialize the population for the simulation
def initialize_population(population_size, mean_age, max_age, x_bounds, y_bounds):
    """
    Initializes a population for the simulation with random attributes. Each individual in the population
    is assigned unique properties such as location, heading direction, speed, and age. 
    Args:
        population_size (int): The total number of individuals in the simulation.
        mean_age (float): The average age of the population.
        max_age (int): The maximum age for individuals in the population.
        x_bounds (list): The minimum and maximum x-coordinates defining the spatial bounds of the population.
        y_bounds (list): The minimum and maximum y-coordinates defining the spatial bounds of the population.
    Returns:
        numpy.ndarray: A 2D array representing the initialized population, with each row representing an individual.
    """
    # Creating an empty array to store population data. Each individual is represented by a row.
    population = np.zeros((population_size, 15))

    # Assigning unique IDs to each individual (from 0 to population_size-1)
    population[:, 0] = np.arange(population_size)
    
    # Initializing random x and y coordinates within the specified bounds
    population[:, 1] = np.random.uniform(low=x_bounds[0] + 0.01, high=x_bounds[1] - 0.01, size=population_size)
    population[:, 2] = np.random.uniform(low=y_bounds[0] + 0.01, high=y_bounds[1] - 0.01, size=population_size)

    # Initializing random headings (directions of movement) using a normal distribution
    population[:, 3:5] = np.random.normal(loc=0, scale=1 / 3, size=(population_size, 2))

    # Initializing random speeds using a normal distribution
    population[:, 5] = np.random.normal(loc=0.01, scale=0.01 / 3, size=population_size)

    # Initializing ages based on a normal distribution centered around the mean_age
    population[:, 7] = np.clip(np.random.normal(loc=mean_age, scale=(max_age - mean_age) / 3, size=population_size), a_min=0, a_max=max_age).astype(np.int32)

    # Initializing a recovery vector using a normal distribution
    population[:, 9] = np.random.normal(loc=0.5, scale=0.5 / 3, size=population_size)

    return population

def update_population_bounds(population, x_bounds, y_bounds):
    """
    Updates the headings of individuals in the population when they reach the defined boundaries.

    Args:
    population (numpy.ndarray): The population matrix.
    x_bounds (numpy.ndarray): The x-coordinate boundaries.
    y_bounds (numpy.ndarray): The y-coordinate boundaries.

    Returns:
    numpy.ndarray: The updated population matrix.
    """
    # Update x-direction heading for individuals at the left boundary
    out_of_bounds_left = (population[:, 1] <= x_bounds[:, 0]) & (population[:, 3] < 0)
    count_left = population[:, 3][out_of_bounds_left].shape[0]
    population[:, 3][out_of_bounds_left] = np.clip(np.random.normal(loc=0.5, scale=0.5 / 3, size=count_left),a_min=0.05, a_max=1)

    # Update x-direction heading for individuals at the right boundary
    out_of_bounds_right = (population[:, 1] >= x_bounds[:, 1]) & (population[:, 3] > 0)
    count_right = population[:, 3][out_of_bounds_right].shape[0]
    population[:, 3][out_of_bounds_right] = np.clip(-np.random.normal(loc=0.5, scale=0.5 / 3, size=count_right),a_min=-1, a_max=-0.05)

    # Update y-direction heading for individuals at the bottom boundary
    out_of_bounds_bottom = (population[:, 2] <= y_bounds[:, 0]) & (population[:, 4] < 0)
    count_bottom = population[:, 4][out_of_bounds_bottom].shape[0]
    population[:, 4][out_of_bounds_bottom] = np.clip(np.random.normal(loc=0.5, scale=0.5 / 3, size=count_bottom),a_min=0.05, a_max=1)

    # Update y-direction heading for individuals at the top boundary
    out_of_bounds_top = (population[:, 2] >= y_bounds[:, 1]) & (population[:, 4] > 0)
    count_top = population[:, 4][out_of_bounds_top].shape[0]
    population[:, 4][out_of_bounds_top] = np.clip(-np.random.normal(loc=0.5, scale=0.5 / 3, size=count_top),a_min=-1, a_max=-0.05)

    return population

###################################################
## updates heading and speed##
###################################################
def update_heading_and_speed(population,population_size, heading=0.02):

    # randomly update heading
    # x
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
def infection(population, population_size, infection_range, infection_chance, frame):
    """
    Identifies and processes new infections within the population.

    Args:
        population (numpy.ndarray): The population matrix.
        population_size (int): Total number of individuals in the population.
        infection_range (float): Range around infected individuals where infection can spread.
        infection_chance (float): Probability of an individual getting infected.
        frame (int): Current frame or day number of the simulation.

    Returns:
        numpy.ndarray: The updated population matrix with new infections processed.
    """
    # Identifying currently infected individuals
    infected = population[population[:, 6] == 1]
    new_infections = []

    # Accelerating disease spread before reaching half the population size
    if len(infected) < (population_size // 2):
        for patient in infected:
            # Defining the infection zone for each infected individual
            x_min, y_min, x_max, y_max = patient[1] - infection_range, patient[2] - infection_range, patient[1] + infection_range, patient[2] + infection_range

            # Identifying healthy individuals within the infection zone
            susceptible_individuals = np.int32(population[:, 0][(x_min < population[:, 1]) & (population[:, 1] < x_max) & (y_min < population[:, 2]) & (population[:, 2] < y_max) & (population[:, 6] == 0)])
            for individual in susceptible_individuals:
                # Determine if a healthy individual gets infected based on probability
                if np.random.random() <= infection_chance:
                    population[individual][6] = 1  # Mark as infected
                    population[individual][8] = frame  # Record the frame of infection
                    new_infections.append(individual)
    else:
        # When more than half the population is infected
        healthy = population[population[:, 6] == 0]
        for person in healthy:
            # Defining the infection zone around each healthy individual
            x_min, y_min, x_max, y_max = person[1] - infection_range, person[2] - infection_range, person[1] + infection_range, person[2] + infection_range

            # Counting the number of infected individuals nearby
            infected_nearby = len(infected[(x_min < infected[:, 1]) & (infected[:, 1] < x_max) & (y_min < infected[:, 2]) & (infected[:, 2] < y_max)])
            if infected_nearby > 0:
                adjusted_infection_chance = infection_chance * infected_nearby
                # Determine if a healthy individual gets infected based on adjusted probability
                if np.random.random() <= adjusted_infection_chance:
                    population[np.int32(person[0])][6] = 1  # Mark as infected
                    population[np.int32(person[0])][8] = frame  # Record the frame of infection
                    new_infections.append(np.int32(person[0]))

    return population

def update_health_status(population, frame, recovery_duration, mortality_chance):
    """
    Processes recovery or death of sick individuals in the population.

    Args:
        population (numpy.ndarray): The population matrix.
        frame (int): The current time frame of the simulation.
        recovery_duration (tuple): A tuple representing the range of recovery duration.
        mortality_chance (float): The probability of death for sick individuals.

    Returns:
        numpy.ndarray: The updated population matrix with recovery and death processed.
    """
    # Identifying individuals who are sick
    sick = population[population[:, 6] == 1]

    # Calculating the duration of illness for each sick individual
    duration_sick = frame - sick[:, 8]
    normalized_recovery_time = (duration_sick - recovery_duration[0]) / np.ptp(recovery_duration)
    normalized_recovery_time = np.clip(normalized_recovery_time, a_min=0, a_max=None)

    # Lists to track recovered and deceased individuals
    recovered_individuals = []
    deceased_individuals = []

    # Assessing each sick individual for recovery or death
    for index, individual in enumerate(sick):
        if normalized_recovery_time[index] >= individual[9]:
            if np.random.random() <= mortality_chance:
                # Individual dies
                sick[index, 6] = 3
                deceased_individuals.append(sick[index, 0])
            else:
                # Individual recovers
                sick[index, 6] = 2
                recovered_individuals.append(sick[index, 0])

    # Updating the population matrix with the new states
    population[population[:, 6] == 1] = sick

    return population

###################################################
##             corona     Simulation            ##
##################################################
def Corona_simulation(frame, population, destinations, population_size,wander_factor,speed,infectious,first_day_of_infection,lockdown ,lockdown_percentage,complying,infection_chance =0.4, recovery_duration=(50, 200), mortality_chance=0.08,infection_range=0.01,infected_plot=[]):
    

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

        xbounds=[0.1, 1.1]
        ybounds=[0.02, 0.98]
        x_bounds = np.array([[xbounds[0] + 0.02, xbounds[1] - 0.02]] * len(population))
        y_bounds = np.array([[ybounds[0] + 0.02, ybounds[1] - 0.02]] * len(population))
        population[population[:,13] == 0] = update_population_bounds(population[population[:,13] == 0], x_bounds, y_bounds)
        
    if lockdown:
        if len(infectious) == 0:
            k = 0
        else:
            #The max() function returns the item with the highest value
            k = np.max(infectious)
        if len(population[population[:,6] == 1]) > len(population) - population_size  or k >= (len(population) - population_size):
            #reduceing  speed of all members of society
            population[:,5] = np.clip(population[:,5], a_min = None, a_max = 0.001)
            #lockdown speed is 1 for not complying
            lockdown_speed = np.zeros((population_size))
            lockdown_speed[np.random.uniform(population_size) >= ( 1- complying )] = 1
            #set speeds of complying people to 0
            population[:,5][lockdown_speed == 0] = 0
        else:
            population = update_heading_and_speed(population, population_size)
    else:
        population = update_heading_and_speed(population, population_size)

    #0 speed for the dead one 
    population[:,3:5][population[:,6] == 3] = 0

    # update positions
    population[:,1] = population[:,1] + (population[:,3] * population[:,5])
    population[:,2] = population[:,2] + (population [:,4] * population[:,5])
    
    # find new infections
    population = infection(population,population_size, infection_range, infection_chance, frame)
    # recover and die
    population = update_health_status(population, frame, recovery_duration, mortality_chance)
    # find new infections
    infected_plot.append(len(population[population[:, 6] == 1]))

###################################################
##             counting function                   ##
##################################################
# count susceptible/infectious/recovered/fatalities1 and store them in array for plotter
def counting(population, population_size, susceptible, infectious, recovered, fatalities1, inf_sum, hea_sum, rec_sum, dead_sum, simulation_steps, frame):
    """
    Counts and updates the status of the population for each simulation frame.

    Args:
        population (numpy.ndarray): The population matrix.
        population_size (int): Total number of individuals in the population.
        susceptible, infectious, recovered, fatalities1 (list): Lists to track the number of individuals in each health status.
        inf_sum, hea_sum, rec_sum, dead_sum (list): Lists to track the cumulative summary of health statuses.
        simulation_steps (int): Total number of steps in the simulation.
        c: Unused parameter (consider removing if not needed).
        frame (int): The current frame or step in the simulation.
    """

    # Current state description: 0=healthy, 1=sick, 2=immune, 3=dead, 4=immune but infectious
    # Updating the counts for each health status
    susceptible.append(len(population[population[:, 6] == 0]))
    infectious.append(len(population[population[:, 6] == 1]))
    recovered.append(len(population[population[:, 6] == 2]))
    fatalities1.append(len(population[population[:, 6] == 3]))

    # Updating the cumulative summaries
    hea_sum.append(len(population[population[:, 6] == 0]))
    inf_sum.append((len(population[population[:, 6] == 1]) / population_size) * 4)
    rec_sum.append((len(population[population[:, 6] == 2]) / population_size) * 4)
    dead_sum.append((len(population[population[:, 6] == 3]) / population_size) * 4)

    
###################################################
##             moting function                   ##
##################################################
def motion(population, destinations):
    """
    Updates the movement of the population towards their respective destinations.

    Args:
        population (numpy.ndarray): The population matrix.
        destinations (numpy.ndarray): The destinations matrix.

    Returns:
        numpy.ndarray: Updated population matrix after movement.
    """
    # Identify unique motions in the population
    unique_motions = np.unique(population[:, 13][population[:, 13] != 0])
    for i in unique_motions:
        # Destination coordinates for the current motion
        motion_x = destinations[:, int((i - 1) * 2)]
        motion_y = destinations[:, int(((i - 1) * 2) + 1)]

        # Current x and y coordinates of the population
        x_coordinate = population[:, 1]
        y_coordinate = population[:, 2]

        # Calculate new headings towards the destination
        x_heading = motion_x - x_coordinate
        y_heading = motion_y - y_coordinate

        # Update population heading in x and y direction
        population[:, 3][(population[:, 13] == i) & (population[:, 14] == 0)] = x_heading[(population[:, 13] == i) & (population[:, 14] == 0)]
        population[:, 4][(population[:, 13] == i) & (population[:, 14] == 0)] = y_heading[(population[:, 13] == i) & (population[:, 14] == 0)]

        # Change speed to a fixed value (e.g., 0.02)
        population[:, 5][(population[:, 13] == i) & (population[:, 14] == 0)] = 0.02

    return population

###################################################
##             checking function                   ##
##################################################
def checking(population, destinations, wander_factor, speed):
    """
    Checks if individuals have arrived at their destinations and updates their status accordingly.

    Args:
        population (numpy.ndarray): The population matrix.
        destinations (numpy.ndarray): The destinations matrix.
        wander_factor (float): Factor affecting the wandering behavior around the destination.
        speed (float): Movement speed of individuals.

    Returns:
        numpy.ndarray: Updated population matrix after checking arrival status.
    """
    # Identify unique motions in the population
    unique_motions = np.unique(population[:, 13][population[:, 13] != 0])
    for i in unique_motions:
        # Destination coordinates for the current motion
        motion_x = destinations[:, int((i - 1) * 2)]
        motion_y = destinations[:, int(((i - 1) * 2) + 1)]

        # Identify individuals who have arrived at the destination
        arrived = population[(np.abs(population[:, 1] - motion_x) < (population[:, 11] * wander_factor)) & (np.abs(population[:, 2] - motion_y) < (population[:, 12] * wander_factor)) & (population[:, 14] == 0)]

        if len(arrived) > 0:
            # Mark individuals as having arrived
            arrived[:, 14] = 1
            # Randomize headings and speeds for those at the destination
            arrived = update_heading_and_speed(arrived, len(arrived))
            # Update the population with the new status of arrived individuals
            population[(np.abs(population[:, 1] - motion_x) < (population[:, 11] * wander_factor)) & (np.abs(population[:, 2] - motion_y) < (population[:, 12] * wander_factor)) & (population[:, 14] == 0)] = arrived

    return population

def staying(population, destinations, wander_factor):
    """
    Keeps individuals who have arrived at their destination within their wander range.

    Args:
        population (numpy.ndarray): The population matrix.
        destinations (numpy.ndarray): The destinations matrix.
        wander_factor (float): Factor determining how far individuals can wander from the destination.

    Returns:
        numpy.ndarray: Updated population matrix with individuals kept within wander range.
    """
    # Identify unique groups that have arrived at their destination
    arrived_groups = np.unique(population[:, 13][(population[:, 13] != 0) & (population[:, 14] == 1)])
    
    for i in arrived_groups:
        # Destination coordinates for the arrived group
        motion_x = destinations[:, int((i - 1) * 2)][(population[:, 14] == 1) & (population[:, 13] == i)]
        motion_y = destinations[:, int(((i - 1) * 2) + 1)][(population[:, 14] == 1) & (population[:, 13] == i)]
        
        # Individuals in the arrived group
        arrived_individuals = population[(population[:, 14] == 1) & (population[:, 13] == i)]

        # Adjusting heading based on wander factor
        # X-coordinate adjustments
        new = arrived_individuals[:, 3][arrived_individuals[:, 1] > (motion_x + (arrived_individuals[:, 11] * wander_factor))].shape
        arrived_individuals[:, 3][arrived_individuals[:, 1] > (motion_x + (arrived_individuals[:, 11] * wander_factor))] = -np.random.normal(0.5, 0.5 / 3, new)

        new = arrived_individuals[:, 3][arrived_individuals[:, 1] < (motion_x - (arrived_individuals[:, 11] * wander_factor))].shape
        arrived_individuals[:, 3][arrived_individuals[:, 1] < (motion_x - (arrived_individuals[:, 11] * wander_factor))] = np.random.normal(0.5, 0.5 / 3, new)

        # Y-coordinate adjustments
        new = arrived_individuals[:, 4][arrived_individuals[:, 2] > (motion_y + (arrived_individuals[:, 12] * wander_factor))].shape
        arrived_individuals[:, 4][arrived_individuals[:, 2] > (motion_y + (arrived_individuals[:, 12] * wander_factor))] = -np.random.normal(0.5, 0.5 / 3, new)

        new = arrived_individuals[:, 4][arrived_individuals[:, 2] < (motion_y - (arrived_individuals[:, 12] * wander_factor))].shape
        arrived_individuals[:, 4][arrived_individuals[:, 2] < (motion_y - (arrived_individuals[:, 12] * wander_factor))] = np.random.normal(0.5, 0.5 / 3, new)

        # Slowing down the speed of arrived individuals
        arrived_individuals[:, 5] = np.random.normal(0.005, 0.005 / 3, arrived_individuals[:, 5].shape)

        # Updating the population with adjusted individuals
        population[(population[:, 14] == 1) & (population[:, 13] == i)] = arrived_individuals

    return population

###################################################
##             figure      build             ##
##################################################
def figure_build():
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    spec = fig.add_gridspec(ncols=3, nrows=2)

    # Billiard Ball Animation
    ax1 = fig.add_subplot(spec[0, :2])
    ax1.set_title('Dynamic Population Movement', fontsize=14, fontweight='bold')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.axis('off')


    # Time-series Plot
    ax2 = fig.add_subplot(spec[1, 0])
    ax2.set_title('Population Health Status Over Time', fontsize=12)
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Number of Individuals')

    # Cumulative Distribution
    ax3 = fig.add_subplot(spec[1, 1:3])
    ax3.set_title('Cumulative Health Status Distribution', fontsize=12)
    ax3.set_xlabel('Days')
    ax3.set_ylabel('Percentage (%)')

    ax4 = fig.add_subplot(spec[0, 2])
    ax4.set_title('Population Density Heatmap', fontsize=14)
    ax4.set_xlabel('X Coordinate')
    ax4.set_ylabel('Y Coordinate')

    return fig, ax1, ax2, ax3, ax4

def visualisation(frame, population, susceptible, infectious, recovered, fatalities1, ax1, ax2, ax3, ax4, inf_sum, hea_sum, rec_sum, dead_sum, population_size):
    color_map = {'healthy': 'lime', 'infected': 'red', 'immune': 'deepskyblue', 'dead': 'slategrey'}
    status = {'healthy': 0, 'infected': 1, 'immune': 2, 'dead': 3}
    # Billiard Ball Animation
    ax1.clear()
    for label, value in status.items():
        segment = population[population[:, 6] == value][:, 1:3]
        ax1.scatter(segment[:, 0], segment[:, 1], color=color_map[label], s=20, label=label.capitalize())
    
    # Time-series plot
    ax2.clear()
    ax2.plot(susceptible, color=color_map['healthy'], label='Healthy')
    ax2.plot(infectious, color=color_map['infected'], label='Infected')
    ax2.plot(recovered, color=color_map['immune'], label='Recovered')
    ax2.plot(fatalities1, color=color_map['dead'], label='Fatalities')
    ax2.legend(loc='upper right', fontsize=10)
    # Cumulative distribution
    ax3.clear()
    x_range = range(len(hea_sum))
    ax3.fill_between(x_range, hea_sum, color=color_map['healthy'], alpha=0.5)
    ax3.fill_between(x_range, inf_sum, color=color_map['infected'], alpha=0.5)
    ax3.fill_between(x_range, rec_sum, color=color_map['immune'], alpha=0.5)
    ax3.fill_between(x_range, dead_sum, color=color_map['dead'], alpha=0.5)
    # Custom Color Map for Heatmap
    colors = [(0, 1, 0, 0), (1, 0, 0, 1)]  # Green to Red
    n_bins = [100]  # Discretizes the interpolation into bins
    cmap = mcolors.LinearSegmentedColormap.from_list(name='custom', colors=colors, N=n_bins[0])
    # Population Density Heatmap
    ax4.clear()
    ax4.hist2d(population[:, 1], population[:, 2], bins=100, cmap=cmap)
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])
    # Update layout
    plt.tight_layout()
    plt.draw()
    plt.pause(0.0001)

def get_user_input(prompt, validation_func, default_value):
    while True:
        user_input = input(f"{prompt} (Default: {default_value}) or 'd' for default: ")
        if user_input.lower() == 'd':
            return default_value
        if validation_func(user_input):
            return user_input
        else:
            print("Invalid input. Please try again.")

def is_digit_and_in_range(val, min_val, max_val):
    return val.isdigit() and min_val <= int(val) <= max_val

def is_percentage(val):
    try:
        val = float(val)
        return 0 <= val <= 100
    except ValueError:
        return False

def main():
    # Default values
    default_population_size = 2000
    default_mean_age = 45
    default_max_age = 105
    default_x_bounds = [0.02, 1.08]
    default_y_bounds = [0.02, 0.98]
    default_simulation_steps = 1000
    default_first_day_of_infection = 1
    default_lockdown = False
    default_lockdown_percentage = 0
    default_complying = 0
    default_self_isolation = False

    print('Welcome To Covid-19 Simulation')
    print('Please enter the following parameters to start the simulation:')

    # User inputs with defaults
    population_size = int(get_user_input("Please enter the population size (100-10000): ", lambda x: is_digit_and_in_range(x, 100, 10000), default_population_size))
    mean_age = int(get_user_input("Please enter the mean age of the population (0-100): ", lambda x: is_digit_and_in_range(x, 0, 100), default_mean_age))
    max_age = int(get_user_input("Please enter the maximum age of the population (0-100): ", lambda x: is_digit_and_in_range(x, 0, 100), default_max_age))
    x_bounds = [float(get_user_input("Please enter the minimum x-coordinate (0-1): ", lambda x: is_percentage(x), default_x_bounds[0])), float(get_user_input("Please enter the maximum x-coordinate (0-1): ", lambda x: is_percentage(x), default_x_bounds[1]))]
    y_bounds = [float(get_user_input("Please enter the minimum y-coordinate (0-1): ", lambda x: is_percentage(x), default_y_bounds[0])), float(get_user_input("Please enter the maximum y-coordinate (0-1): ", lambda x: is_percentage(x), default_y_bounds[1]))]
    simulation_steps = int(get_user_input("Please enter the number of simulation steps (100-10000): ", lambda x: is_digit_and_in_range(x, 100, 10000), default_simulation_steps))
    first_day_of_infection = int(get_user_input("Please enter the first day of infection (1-500): ", lambda x: is_digit_and_in_range(x, 1, 500), default_first_day_of_infection))

    lockdown = get_user_input("Lockdown scenario (True/False): ", lambda val: val.lower() in ['true', 'false', 'd'], default_lockdown) == 'true'
    if lockdown:
        lockdown_percentage = float(get_user_input("Enter lockdown percentage (0-100): ", is_percentage, default_lockdown_percentage))
        complying = float(get_user_input("Enter percentage of people complying with lockdown (0-100): ", is_percentage, default_complying))
    else:
        lockdown_percentage = 0
        complying = 0
    
    print("#" * 50)
    print('Starting simulation...')
    print("#" * 50)
    
    # Initialize population and destinations
    population = initialize_population(population_size, mean_age, max_age, x_bounds, y_bounds)
    destinations = np.zeros((population_size, 2))

    # Initialize the figure
    fig, ax1, ax2, ax3, ax4 = figure_build()
    wander_factor = 0.5
    speed = 0.01
    infection_range = 0.01
    infection_chance = 0.4
    recovery_duration = (50, 200)
    mortality_chance = 0.08
    infected_plot = []
    # Lists to track the number of individuals in each health status
    susceptible = []
    infectious = []
    recovered = []
    fatalities1 = []
    inf_sum = []
    hea_sum = []
    rec_sum = []
    dead_sum = []
    # Lists to track the cumulative summary of health statuses
    # Current state description: 0=healthy, 1=sick, 2=immune, 3=dead, 4=immune but infectious

    # Main simulation loop
    for frame in range(simulation_steps):
        Corona_simulation(frame, population, destinations, population_size,wander_factor,speed,infectious,first_day_of_infection,lockdown ,lockdown_percentage,complying)
        visualisation(frame, population, susceptible, infectious, recovered, fatalities1, ax1, ax2, ax3, ax4, inf_sum, hea_sum, rec_sum, dead_sum, population_size)
        counting(population, population_size, susceptible, infectious, recovered, fatalities1, inf_sum, hea_sum, rec_sum, dead_sum, simulation_steps,frame)
        update_status_message(frame, population)

def update_status_message(frame, population):
    healthy = population[population[:, 6] == 0][:, 1:3]
    infected = population[population[:, 6] == 1][:, 1:3]
    immune = population[population[:, 6] == 2][:, 1:3]
    fatalities = population[population[:, 6] == 3][:, 1:3]
    sys.stdout.write('\r')
    sys.stdout.write('Days: %i, total: [%i], healthy: [%i] infected: [%i] immune: [%i] fatalities: [%i]' % (frame,len(population),len(healthy),len(infected), len(immune),len(fatalities)))


if __name__ == '__main__':
    main()
