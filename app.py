import streamlit as st
import pandas as pd
import random
import csv

# Function to read the CSV file and convert it to the desired format
def read_csv_to_dict(file_path):
    program_ratings = {}

    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        # Skip the header
        header = next(reader)

        for row in reader:
            program = row[0]
            ratings = [float(x) for x in row[1:]]  # Convert the ratings to floats
            program_ratings[program] = ratings

    return program_ratings

##################################### PARAMETERS AND DATASET ################################################################
# Sample rating programs dataset for each time slot.
file_path = "Modified_TV_Scheduling.csv"

program_ratings_dict = read_csv_to_dict(file_path)
ratings = program_ratings_dict

all_programs = list(ratings.keys())  # all programs
all_time_slots = list(range(6, 24))  # time slots

# Defining fitness function
def fitness_function(schedule):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]
    return total_rating

# Initializing the population
def initialize_pop(programs, time_slots):
    if not programs:
        return [[]]

    all_schedules = []
    for i in range(len(programs)):
        for schedule in initialize_pop(programs[:i] + programs[i + 1:], time_slots):
            all_schedules.append([programs[i]] + schedule)

    return all_schedules

# Selection
def finding_best_schedule(all_schedules):
    best_schedule = []
    max_ratings = 0

    for schedule in all_schedules:
        total_ratings = fitness_function(schedule)
        if total_ratings > max_ratings:
            max_ratings = total_ratings
            best_schedule = schedule

    return best_schedule

# Genetic Algorithm Functions
# Crossover
def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

# Mutation
def mutate(schedule):
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutation_point] = new_program
    return schedule

# Genetic Algorithm
def genetic_algorithm(initial_schedule, generations, population_size, crossover_rate, mutation_rate, elitism_size):
    population = [initial_schedule]

    for _ in range(population_size - 1):
        random_schedule = initial_schedule.copy()
        random.shuffle(random_schedule)
        population.append(random_schedule)

    for generation in range(generations):
        new_population = []

        # Elitism
        population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
        new_population.extend(population[:elitism_size])

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population

    return population[0]

############################################ STREAMLIT APP ########################################################################
def main():
    st.title("Genetic Algorithm TV Scheduling App")

    st.header("Step 1: Input Genetic Algorithm Parameters")
    # Input sliders for parameters
    CO_R = st.slider(
        label="Crossover Rate (CO_R)",
        min_value=0.0,
        max_value=0.95,
        value=0.9,
        step=0.01
    )

    MUT_R = st.slider(
        label="Mutation Rate (MUT_R)",
        min_value=0.01,
        max_value=0.05,
        value=0.01,
        step=0.001
    )

    st.write(f"**Crossover Rate (CO_R):** {CO_R}")
    st.write(f"**Mutation Rate (MUT_R):** {MUT_R}")

    st.header("Step 2: Generate and Display Schedule")

    # Initial schedule
    initial_best_schedule = finding_best_schedule(initialize_pop(all_programs, all_time_slots))

    # Button to run the genetic algorithm
    if st.button("Run Genetic Algorithm"):
        rem_t_slots = len(all_time_slots) - len(initial_best_schedule)
        genetic_schedule = genetic_algorithm(
            initial_best_schedule,
            generations=100,
            population_size=50,
            crossover_rate=CO_R,
            mutation_rate=MUT_R,
            elitism_size=2
        )

        final_schedule = initial_best_schedule + genetic_schedule[:rem_t_slots]

        # Display final schedule in table format
        st.subheader("Generated Schedule")
        schedule_df = pd.DataFrame({
            "Time Slot": [f"{time_slot:02d}:00" for time_slot in all_time_slots],
            "Program": final_schedule
        })
        st.table(schedule_df)

if __name__ == "__main__":
    main()
