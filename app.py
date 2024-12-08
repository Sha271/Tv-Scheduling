import streamlit as st
import pandas as pd
import csv

# Function to read the CSV file and convert it to a pandas DataFrame
def read_csv_to_dataframe(file_path):
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)
        data = [row for row in reader]
        
    return pd.DataFrame(data, columns=header)

# Main Streamlit App
def main():
    st.title("Genetic Algorithm TV Scheduling App")

    # Section 1: Input Genetic Algorithm Parameters
    st.header("Input your Genetic Algorithm Parameters")

    # Input parameters with default values and ranges
    CO_R = st.slider(
        label="Crossover Rate (CO_R)",
        min_value=0.0,
        max_value=0.95,
        value=0.8,
        step=0.01
    )

    MUT_R = st.slider(
        label="Mutation Rate (MUT_R)",
        min_value=0.01,
        max_value=0.05,
        value=0.02,
        step=0.001
    )

    st.write("### Selected Parameters")
    st.write(f"- **Crossover Rate (CO_R):** {CO_R}")
    st.write(f"- **Mutation Rate (MUT_R):** {MUT_R}")

    # Placeholder for genetic algorithm logic
    st.write("""
    Once the parameters are set, the genetic algorithm will run to generate a schedule.
    
    (This is a placeholder for the algorithm integration.)
    """)

    # Section 2: Display Generated Schedule
    st.header("View Generated Schedule")

    # Read the CSV file (Replace with your uploaded file path)
    file_path = "/mnt/data/Modified_TV_Scheduling.csv"
    schedule_df = read_csv_to_dataframe(file_path)

    st.write("### Generated TV Schedule")
    st.dataframe(schedule_df)

    st.write("""
    This table displays the scheduled programs with respective ratings for each hour.
    You can adjust the genetic algorithm parameters to generate a new schedule.
    """)

if __name__ == "__main__":
    main()
