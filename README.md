# COVID-19 Simulation

## Introduction
This Python code, developed by Abdullah Saad, simulates the spread of COVID-19 within a population. It models the movement and interactions of individuals, tracking their health status over time. The simulation takes into account various parameters, such as population size, age distribution, spatial bounds, and infection-related factors.

## Installation
Before running the simulation, make sure you have the required libraries installed. You can install them using pip3:


pip3 install matplotlib
pip3 install numpy
python3 covid19_simulation.py

## Code Overview

The code consists of several functions and parameters to control the simulation:

1. `initialize_population`: Initializes the population with random attributes such as location, heading direction, speed, and age.

2. `update_population_bounds`: Updates the headings of individuals when they reach the defined boundaries.

3. `update_heading_and_speed`: Updates the heading and speed of individuals based on random factors.

4. `infection`: Identifies and processes new infections within the population.

5. `update_health_status`: Processes recovery or death of sick individuals in the population.

6. `Corona_simulation`: The main simulation function that orchestrates the entire simulation process.

7. `counting`: Counts and updates the status of the population for each simulation frame.

8. `motion`: Updates the movement of the population towards their respective destinations.

9. `checking`: Checks if individuals have arrived at their destinations and updates their status accordingly.

10. `staying`: Keeps individuals who have arrived at their destination within their wander range.

11. `figure_build`: Builds the visualization figure for the simulation.

12. `visualisation`: Updates the visual representation of the simulation, including animations and plots.

13. `get_user_input`: Helper function to get user input with validation.

14. `is_digit_and_in_range`: Helper function to validate if input is a digit within a specified range.

15. `is_percentage`: Helper function to validate if input is a valid percentage.

16. `update_status_message`: Updates the status message with the current state of the population.


## Running the Simulation

Upon running the simulation, you will be prompted to input various parameters such as population size, age distribution, spatial bounds, simulation steps, and more. The simulation will then execute and provide visualizations of the population's health status over time.

## Note

This simulation provides insights into how infectious diseases can spread within a population. It is a simplified model and should not be used for making real-world predictions.

# Simulation-based Analysis of COVID-19 Transmission and Containment Strategies

## Abstract

This white paper presents a comprehensive analysis of COVID-19 transmission and containment strategies using mathematical modeling and simulation techniques. The objective is to simulate various infection scenarios, evaluate the effectiveness of non-pharmaceutical interventions (NPIs), and provide insights for policy decisions related to quarantine measures, school restrictions, and economic activities. The simulation employs Python programming with NumPy and matplotlib libraries to visualize the spread of the virus within a population. By emphasizing the importance of physical distancing and lockdown procedures, the simulation highlights the potential impact of adhering to restriction rules in reducing the risk of infection.

## Introduction

The emergence of the novel coronavirus, SARS-CoV-2, and its associated disease, COVID-19, has led to a global pandemic. Drawing upon historical experiences with past epidemics and pandemics, this paper emphasizes the need for simulations to assess infection-related scenarios and develop strategies for effective response. With pharmaceutical interventions still in development, non-pharmaceutical interventions (NPIs) such as physical distancing and lockdown measures have played a crucial role in curbing the virus's spread.

## Simulation Framework

The simulation model is designed to replicate real-world dynamics by incorporating equations that forecast the spread of the virus over time. It accurately represents the degree of contact between individuals and implements various containment strategies to limit the number of contacts, thereby influencing the number of infected cases. By generating statistical data, the simulation demonstrates the significance of adhering to restriction rules and provides insights into the potential outcomes of following such measures.

## Simulation Implementation

Utilizing the "billiard ball model," each individual within the population is represented as a 2D cycle, categorized as healthy, infected, immune, or deceased. The simulation program, implemented in Python, employs randomness and probability to determine infection transmission, recovery, and mortality rates. Key parameters such as population size, infection chances, and recovery rates are considered in the simulation's calculations.

## SIR/SIDARTHE Model

To capture the dynamics of the disease, the SIR/SIDARTHE model is utilized, dividing the population into compartments such as susceptible, infected, recovered, and deceased individuals. This model allows for a more precise analysis of the epidemic's progression, considering factors such as immunity, recovery, and contagiousness. The system of differential equations governing the model is employed to simulate the spread of the virus and estimate the impact of different containment strategies.

The SIR/SIDARTHE model consists of the following equations:

Susceptible individuals: `dS/dt = -βSI - αS`

Infected individuals: `dI/dt = βSI - (γ + δ)I`

Recovered individuals: `dR/dt = γI + ρT`

Deceased individuals: `dD/dt = δI`

Asymptomatic individuals: `dA/dt = αS - θA - ρA`

Diagnosed individuals: `dT/dt = θA - γT - ρT`

Hospitalized individuals: `dH/dt = γT - λH - κH`

Extubated individuals: `dE/dt = λH - ξE`

Transferred individuals: `dT/dt = κH - μT`

where:

- S: Susceptible individuals
- I: Infected individuals
- R: Recovered individuals
- D: Deceased individuals
- A: Asymptomatic individuals
- T: Diagnosed individuals
- H: Hospitalized individuals
- E: Extubated individuals
- θ: Rate of asymptomatic diagnosis
- α: Rate of becoming asymptomatic
- β: Infection rate
- γ: Recovery rate
- δ: Mortality rate
- ρ: Rate of transitioning from diagnosed to recovered
- λ: Rate of transitioning from diagnosed to hospitalized
- κ: Rate of transitioning from hospitalized to transferred
- μ: Rate of transitioning from transferred to diagnosed
- ξ: Rate of transitioning from extubated to recovered

## Simulation Scenarios

The simulation explores various scenarios, including the effects of lockdown measures and self-isolation. By introducing lockdown at a specified percentage of infected individuals, the simulation assesses the effectiveness of this intervention in controlling the spread of the virus. Additionally, the impact of self-isolation is evaluated by reducing the speed and infection range of individuals within the simulation.

## Results and Analysis

The simulation generates visual representations of infection patterns, including histograms and line charts, under different scenarios. These visualizations illustrate the dynamics of susceptible, infected, recovered, and deceased individuals within the population. By comparing the outcomes of normal infection simulation, lockdown scenarios, and self-isolation measures, policymakers can gain valuable insights into the potential efficacy of different containment strategies.

## Conclusion

Mathematical modeling and simulation play a crucial role in understanding the dynamics of COVID-19 transmission and assessing the effectiveness of various containment strategies. This white paper has presented a comprehensive analysis of the simulation-based approach, highlighting the importance of adherence to restriction rules and providing valuable insights for policymakers in their decision-making process. By leveraging the power of simulations, we can proactively plan and execute strategies to mitigate the spread of COVID-19 and minimize its impact on society.
