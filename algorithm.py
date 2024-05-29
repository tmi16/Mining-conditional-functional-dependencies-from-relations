import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, scrolledtext
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.feature_selection import VarianceThreshold

def load_dataset(file_path):
    """
    Load the dataset from a CSV file.
    
    Parameters:
    - file_path: str, the path to the CSV file.
    
    Returns:
    - DataFrame containing the dataset.
    """
    try:
        data = pd.read_csv(file_path)
        print("Dataset loaded successfully. Here are the first few rows:")
        print(data.head())
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")


def preprocess_data(data):
    """
    Preprocess the dataset for CFD discovery.
    
    Parameters:
    - data: DataFrame, the dataset to preprocess.
    
    Returns:
    - DataFrame, the preprocessed dataset.
    """
    # String data normalization: convert to lowercase and remove Spaces at both ends
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].str.lower().str.strip()
    
    # Delete rows that contain missing values
    data = data.dropna()

    # Identify and remove outliers
    for col in data.select_dtypes(include=['int', 'float']):
        lower_bound, upper_bound = data[col].quantile(0.01), data[col].quantile(0.99)
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

    # Discretization of numerical data: discretization using adaptive methods
    for col in data.select_dtypes(include=['int', 'float']):
        data[col] = pd.qcut(data[col], q=4, duplicates='drop', labels=False)


    return data


def generate_optimized_candidate_cfds(data, support_threshold=0.01, sample_size=0.1, num_samples=5):
    """
    Generate and optimize candidate CFDs from a dataset using a series of steps including initial generation,
    pruning based on support, and dynamic evaluation using multiple sample-based quick validations.

    Parameters:
    - data: DataFrame, the dataset from which to generate CFDs.
    - support_threshold: float, minimum proportion of dataset that must support the candidate (default 0.01).
    - sample_size: float or int, the size of the sample used for dynamic pruning (default 0.1).
    - num_samples: int, the number of times to validate each candidate with different samples (default 5).
    """
    # Step 1: Generate Initial Candidates
    candidates = []
    unique_counts = data.nunique()
    potential_condition_columns = unique_counts[unique_counts < 20].index.tolist()
    potential_dependent_columns = unique_counts[unique_counts >= 20].index.tolist()

    for condition_col in potential_condition_columns:
        for dependent_col in potential_dependent_columns:
            if condition_col != dependent_col:
                condition_values = data[condition_col].unique()
                dependent_values = data[dependent_col].unique()

                for cond_val in condition_values:
                    for dep_val in dependent_values:
                        candidates.append(((condition_col, cond_val), (dependent_col, dep_val)))

    # Step 2: Prune Candidates Based on Support
    pruned_candidates = []
    for candidate in candidates:
        condition_col, condition_val = candidate[0]
        dependent_col, dependent_val = candidate[1]

        condition_met_rows = data[data[condition_col] == condition_val]
        both_met_rows = condition_met_rows[condition_met_rows[dependent_col] == dependent_val]

        support = len(both_met_rows) / len(data)
        if support >= support_threshold:
            pruned_candidates.append(candidate)

    # Step 3: Dynamic Pruning Based on Multiple Sample-Based Quick Validations
    dynamically_pruned_candidates = []
    for candidate in pruned_candidates:
        valid_count = 0
        for _ in range(num_samples):
            sample_data = data.sample(frac=sample_size) if isinstance(sample_size, float) else data.sample(n=sample_size)
            condition_col, condition_val = candidate[0]
            dependent_col, dependent_val = candidate[1]

            condition_met_rows = sample_data[sample_data[condition_col] == condition_val]
            both_met_rows = condition_met_rows[condition_met_rows[dependent_col] == dependent_val]

            dynamic_support = len(both_met_rows) / len(condition_met_rows) if len(condition_met_rows) > 0 else 0
            if dynamic_support >= support_threshold:
                valid_count += 1

        # Consider candidate valid if it passes the support threshold in at least half of the samples
        if valid_count >= num_samples / 2:
            dynamically_pruned_candidates.append(candidate)

    print(f"Generated {len(candidates)} initial candidates.")
    print(f"Pruned to {len(pruned_candidates)} candidates after support check.")
    print(f"Dynamically pruned to {len(dynamically_pruned_candidates)} candidates after multiple samples validation.")
    return dynamically_pruned_candidates



def validate_single_candidate(data, candidate, tolerance=0.3):
    """
    Validate a single candidate CFD, allowing for a small tolerance of mismatches.
    
    Parameters:
    - data: DataFrame, the dataset to validate CFDs against.
    - candidate: tuple, specifying the condition and dependent columns and their required values.
    - tolerance: float, the proportion of mismatches allowed (default is 5%).
    
    Returns:
    - bool: whether the candidate is valid within the specified tolerance.
    - tuple: the candidate CFD being validated.
    """
    condition_col, condition_val = candidate[0]
    dependent_col, dependent_val = candidate[1]

    try:
        if pd.api.types.is_string_dtype(data[condition_col]):
            condition_val = str(condition_val) if condition_val is not None else None
        else:
            condition_val = pd.to_numeric(condition_val, errors='coerce')

        if pd.api.types.is_string_dtype(data[dependent_col]):
            dependent_val = str(dependent_val) if dependent_val is not None else None
        else:
            dependent_val = pd.to_numeric(dependent_val, errors='coerce')

    except Exception as e:
        print(f"Error converting data types: {e}")
        return False, candidate

    mask = data[condition_col].eq(condition_val)
    if not mask.any():
        print(f"No match for condition: {condition_col} == {condition_val}")
        return False, candidate

    valid_mask = data.loc[mask, dependent_col].eq(dependent_val)
    valid_rate = valid_mask.sum() / mask.sum()

    if valid_rate < (1 - tolerance):
        print(f"Dependency failed: {dependent_col} == {dependent_val} where {condition_col} == {condition_val}, valid rate: {valid_rate:.2f}")
        return False, candidate

    return True, candidate



def validate_candidate_cfds_parallel(data, candidate_cfds, max_workers=10):
    """
    Validate candidate CFDs in parallel using ThreadPoolExecutor.
    """
    validated_cfds = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(validate_single_candidate, data, candidate): candidate for candidate in candidate_cfds}

        for future in as_completed(futures):
            try:
                is_valid, candidate = future.result()
                if is_valid:
                    validated_cfds.append(candidate)
            except Exception as e:
                print(f"An error occurred during validation of candidate {candidate}: {e}")

    print(f"Validated {len(validated_cfds)} CFDs in parallel.")
    return validated_cfds


def run_cfd_discovery(text_widget):
    filepath = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")])
    if not filepath:
        return
    data = load_dataset(filepath)
    if data is not None:
        tolerance = simpledialog.askfloat("Enter Tolerance", "Please enter the tolerance rate (e.g., 0.1 for 10%):", minvalue=0.0, maxvalue=1.0)
        if tolerance is None:
            return
        data = preprocess_data(data)
        candidates = generate_optimized_candidate_cfds(data)
        validated_cfds = validate_candidate_cfds_parallel(data, candidates, tolerance)
        
        # Display CFDs in text widget
        text_widget.config(state=tk.NORMAL)  # Enable text widget editing
        text_widget.delete('1.0', tk.END)  # Clear existing text
        for cfd in validated_cfds:
            text_widget.insert(tk.END, f"Condition: {cfd[0]}, Dependency: {cfd[1]}\n")
        text_widget.config(state=tk.DISABLED)  # Disable text widget editing
        messagebox.showinfo("Results", f"Number of validated CFDs: {len(validated_cfds)}")

def main():
    root = tk.Tk()
    root.title("CFD Discovery Tool")
    
    frame = tk.Frame(root)
    frame.pack(pady=20)
    
    tk.Label(frame, text="Welcome to the CFD Discovery Tool", font=("Arial", 16)).pack(pady=10)
    tk.Button(frame, text="Load Data and Run CFD Discovery", command=lambda: run_cfd_discovery(text_widget), font=("Arial", 14)).pack(pady=10)
    
    text_widget = scrolledtext.ScrolledText(frame, width=60, height=10, font=("Arial", 12), state=tk.DISABLED)
    text_widget.pack(pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    main()

