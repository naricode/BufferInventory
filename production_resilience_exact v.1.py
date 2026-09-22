# Import the CSV module so every numerical result can be saved in a transparent text format.
import csv
# Import the math module for logarithms and exponentials used in the closed-form stochastic model.
import math
# Import Path so output paths work consistently on Windows, macOS, and Linux.
from pathlib import Path
# Import NumPy for reproducible Monte Carlo validation of the analytical solution.
import numpy as np
# Import Matplotlib to create the revised sensitivity figures used in the manuscript.
import matplotlib.pyplot as plt

# Define the folder where all reproducible outputs will be written.
#OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
# option 2 for saving files
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

# Hide the main Tkinter window
root = tk.Tk()
root.withdraw()

# Open Windows folder selection dialog
selected_folder = filedialog.askdirectory(
    title="Select the output folder"
)

# Check whether a folder was selected
if selected_folder:
    OUTPUT_DIR = Path(selected_folder)
    print(f"Output folder selected: {OUTPUT_DIR}")
else:
    print("No folder selected.")
    raise SystemExit("Program cancelled.")









# Create the output folder if it does not already exist.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Fix the random seed used only for validation so the analytical results can be independently reproduced.
RANDOM_SEED = 20260906

# Define the exact stochastic optimization model for one set of parameter values.
def solve_exact_model(params):
    # Read the upstream mean time between failures in hours.
    mttf_um = params["MTTF_um"]
    # Read the downstream mean time between failures in hours.
    mttf_dm = params["MTTF_dm"]
    # Read the upstream mean time to repair in hours.
    mttr_um = params["MTTR_um"]
    # Read the downstream mean time to repair in hours.
    mttr_dm = params["MTTR_dm"]
    # Read the required probability that a buffer is replenished before the next machine failure.
    rho = params["rho"]
    # Convert the upstream exponential reliability requirement into an allowable replenishment time.
    q_um = -mttf_um * math.log(rho)
    # Convert the downstream exponential reliability requirement into an allowable replenishment time.
    q_dm = -mttf_dm * math.log(rho)
    # Calculate the effective marginal cost of one unit of upstream buffer after substituting P_um=S_ub/q_um.
    k_um = params["cub"] + params["Cum"] / q_um
    # Calculate the effective marginal cost of one unit of downstream buffer after substituting P_dm=S_db/q_dm.
    k_dm = params["cdb"] + params["Cdm"] / q_dm
    # Calculate the bottleneck time that may be lost while still meeting the target utilization.
    delay_budget = (1.0 - params["U"]) * params["W"]
    # Create a list that will store every mathematically possible active-set candidate.
    candidates = []
    # Check whether the safety-time allowance alone can absorb the expected repair delays.
    if mttr_um + mttr_dm <= delay_budget + 1e-12:
        # Add the zero-protection solution when no additional buffer or protective capacity is required.
        candidates.append({"Sub": 0.0, "Sdb": 0.0, "case": "no additional protection"})
    # Calculate the remaining delay allowance available to the upstream machine if the downstream side is left unprotected.
    upstream_remaining = delay_budget - mttr_dm
    # Check whether an upstream-only solution can satisfy the expected-delay constraint.
    if 0.0 < upstream_remaining <= mttr_um:
        # Solve E[D_um]=MTTR_um*exp(-S_ub/(p*MTTR_um)) for the required upstream buffer size.
        sub = -params["p"] * mttr_um * math.log(upstream_remaining / mttr_um)
        # Add the feasible upstream-only candidate to the candidate list.
        candidates.append({"Sub": sub, "Sdb": 0.0, "case": "upstream protection only"})
    # Calculate the remaining delay allowance available to the downstream machine if the upstream side is left unprotected.
    downstream_remaining = delay_budget - mttr_um
    # Check whether a downstream-only solution can satisfy the expected-delay constraint.
    if 0.0 < downstream_remaining <= mttr_dm:
        # Solve E[D_dm]=MTTR_dm*exp(-S_db/(p*MTTR_dm)) for the required downstream buffer size.
        sdb = -params["p"] * mttr_dm * math.log(downstream_remaining / mttr_dm)
        # Add the feasible downstream-only candidate to the candidate list.
        candidates.append({"Sub": 0.0, "Sdb": sdb, "case": "downstream protection only"})
    # Calculate the denominator that appears in the KKT solution when both buffers are strictly positive.
    both_denominator = mttr_um * k_um + mttr_dm * k_dm
    # Calculate the upstream expected-delay allocation implied by the KKT first-order conditions.
    d_um_both = delay_budget * mttr_um * k_um / both_denominator
    # Calculate the downstream expected-delay allocation implied by the KKT first-order conditions.
    d_dm_both = delay_budget * mttr_dm * k_dm / both_denominator
    # Check whether both KKT-implied delay contributions are physically feasible for positive buffers.
    if 0.0 < d_um_both <= mttr_um and 0.0 < d_dm_both <= mttr_dm:
        # Convert the upstream expected-delay contribution into its required buffer size.
        sub = -params["p"] * mttr_um * math.log(d_um_both / mttr_um)
        # Convert the downstream expected-delay contribution into its required buffer size.
        sdb = -params["p"] * mttr_dm * math.log(d_dm_both / mttr_dm)
        # Add the two-sided KKT candidate to the candidate list.
        candidates.append({"Sub": sub, "Sdb": sdb, "case": "both sides protected"})
    # Raise an explicit error if the parameter combination unexpectedly produces no feasible active-set candidate.
    if not candidates:
        # Stop the script because continuing without a feasible design would invalidate the reported results.
        raise RuntimeError("No feasible active-set candidate was found for the supplied parameters.")
    # Loop over every candidate so its protective capacity, expected delay, and total cost can be evaluated consistently.
    for candidate in candidates:
        # Read the upstream buffer size from this active-set candidate.
        sub = candidate["Sub"]
        # Read the downstream buffer size from this active-set candidate.
        sdb = candidate["Sdb"]
        # Set upstream protective capacity to zero when no upstream buffer exists; otherwise bind the chance constraint.
        pum = 0.0 if sub == 0.0 else sub / q_um
        # Set downstream protective capacity to zero when no downstream buffer exists; otherwise bind the chance constraint.
        pdm = 0.0 if sdb == 0.0 else sdb / q_dm
        # Calculate the analytical expected upstream bottleneck delay under exponential repair duration.
        expected_dum = mttr_um * math.exp(-sub / (params["p"] * mttr_um))
        # Calculate the analytical expected downstream bottleneck delay under exponential repair duration.
        expected_ddm = mttr_dm * math.exp(-sdb / (params["p"] * mttr_dm))
        # Calculate the upstream protective-capacity cost.
        protection_um_cost = params["Cum"] * pum
        # Calculate the downstream protective-capacity cost.
        protection_dm_cost = params["Cdm"] * pdm
        # Calculate the upstream buffer cost using the revised per-unit buffer-capacity interpretation.
        buffer_um_cost = params["cub"] * sub
        # Calculate the downstream buffer cost using the revised per-unit buffer-capacity interpretation.
        buffer_dm_cost = params["cdb"] * sdb
        # Sum all four design-cost components to obtain the candidate objective value.
        total_cost = protection_um_cost + protection_dm_cost + buffer_um_cost + buffer_dm_cost
        # Store the upstream protective-capacity decision in the candidate record.
        candidate["Pum"] = pum
        # Store the downstream protective-capacity decision in the candidate record.
        candidate["Pdm"] = pdm
        # Store the expected upstream delay in the candidate record.
        candidate["expected_Dum"] = expected_dum
        # Store the expected downstream delay in the candidate record.
        candidate["expected_Ddm"] = expected_ddm
        # Store the expected total delay in the candidate record.
        candidate["expected_total_delay"] = expected_dum + expected_ddm
        # Store the upstream protective-capacity cost component.
        candidate["protection_um_cost"] = protection_um_cost
        # Store the downstream protective-capacity cost component.
        candidate["protection_dm_cost"] = protection_dm_cost
        # Store the upstream buffer cost component.
        candidate["buffer_um_cost"] = buffer_um_cost
        # Store the downstream buffer cost component.
        candidate["buffer_dm_cost"] = buffer_dm_cost
        # Store the complete objective value.
        candidate["total_cost"] = total_cost
    # Select the globally optimal candidate because the reduced problem is convex and all possible active sets were evaluated.
    best = min(candidates, key=lambda item: item["total_cost"])
    # Add the allowable upstream replenishment time implied by the exponential chance constraint.
    best["q_um"] = q_um
    # Add the allowable downstream replenishment time implied by the exponential chance constraint.
    best["q_dm"] = q_dm
    # Add the expected-delay budget for direct checking against the solution.
    best["delay_budget"] = delay_budget
    # Add the effective upstream buffer cost coefficient used in the reduced convex problem.
    best["k_um"] = k_um
    # Add the effective downstream buffer cost coefficient used in the reduced convex problem.
    best["k_dm"] = k_dm
    # Return the exact optimal design and all quantities required for reporting.
    return best

# Define a Monte Carlo validation function that evaluates a fixed here-and-now design after it has been optimized.
def validate_solution(params, solution, n_replications=100000, seed=RANDOM_SEED):
    # Create four independent random-number streams so validation variables are statistically independent.
    rng_ttf_um = np.random.default_rng(seed + 1)
    # Create the downstream inter-failure random-number stream.
    rng_ttf_dm = np.random.default_rng(seed + 2)
    # Create the upstream repair-duration random-number stream.
    rng_tr_um = np.random.default_rng(seed + 3)
    # Create the downstream repair-duration random-number stream.
    rng_tr_dm = np.random.default_rng(seed + 4)
    # Sample upstream inter-failure times from the assumed exponential distribution.
    ttf_um = rng_ttf_um.exponential(params["MTTF_um"], n_replications)
    # Sample downstream inter-failure times from the assumed exponential distribution.
    ttf_dm = rng_ttf_dm.exponential(params["MTTF_dm"], n_replications)
    # Sample upstream repair durations from the assumed exponential distribution.
    tr_um = rng_tr_um.exponential(params["MTTR_um"], n_replications)
    # Sample downstream repair durations from the assumed exponential distribution.
    tr_dm = rng_tr_dm.exponential(params["MTTR_dm"], n_replications)
    # Define upstream replenishment time as zero when no buffer is installed.
    refill_um = 0.0 if solution["Sub"] == 0.0 else solution["Sub"] / solution["Pum"]
    # Define downstream replenishment time as zero when no buffer is installed.
    refill_dm = 0.0 if solution["Sdb"] == 0.0 else solution["Sdb"] / solution["Pdm"]
    # Estimate the probability that the upstream buffer can be restored before the next upstream failure.
    refill_probability_um = 1.0 if solution["Sub"] == 0.0 else float(np.mean(ttf_um >= refill_um))
    # Estimate the probability that the downstream buffer can be restored before the next downstream failure.
    refill_probability_dm = 1.0 if solution["Sdb"] == 0.0 else float(np.mean(ttf_dm >= refill_dm))
    # Calculate upstream disruption delay in every validation replication.
    dum = np.maximum(0.0, tr_um - solution["Sub"] / params["p"])
    # Calculate downstream disruption delay in every validation replication.
    ddm = np.maximum(0.0, tr_dm - solution["Sdb"] / params["p"])
    # Return empirical validation statistics without re-optimizing after any random realization.
    return {
        # Report the number of independent validation replications.
        "validation_replications": n_replications,
        # Report the empirical upstream replenishment reliability.
        "empirical_refill_probability_um": refill_probability_um,
        # Report the empirical downstream replenishment reliability.
        "empirical_refill_probability_dm": refill_probability_dm,
        # Report empirical expected upstream delay.
        "empirical_mean_Dum": float(np.mean(dum)),
        # Report empirical expected downstream delay.
        "empirical_mean_Ddm": float(np.mean(ddm)),
        # Report empirical expected total bottleneck delay.
        "empirical_mean_total_delay": float(np.mean(dum + ddm)),
        # Report the analytical expected total delay for side-by-side validation.
        "analytical_mean_total_delay": solution["expected_total_delay"],
        # Report the model's required replenishment reliability.
        "target_refill_probability": params["rho"],
        # Report the allowed total bottleneck delay implied by target utilization.
        "delay_budget": solution["delay_budget"],
    }

# Define a helper that writes a list of dictionaries to a CSV file.
def write_csv(path, rows):
    # Do nothing when an empty result list is supplied.
    if not rows:
        # Exit the helper before trying to create an invalid header.
        return
    # Open the output file in text mode with UTF-8 encoding.
    with path.open("w", newline="", encoding="utf-8") as handle:
        # Build the CSV writer from the ordered keys in the first result row.
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        # Write column names to the first row.
        writer.writeheader()
        # Write all result dictionaries below the header.
        writer.writerows(rows)

# Define the numerical base case used in the corrected manuscript.
def baseline_parameters():
    # Return every base-case input with one explicit and internally consistent unit system.
    return {"Cum": 55.0, "Cdm": 96.0, "cub": 12.0, "cdb": 31.0, "p": 36.0, "U": 0.85, "W": 24.0, "MTTF_um": 4.0, "MTTF_dm": 4.0, "MTTR_um": 2.0, "MTTR_dm": 2.0, "rho": 0.80}

# Define the sensitivity experiment that replaces the inconsistent GA scenario runs in the original manuscript.
def run_sensitivity():
    # Create an empty list for the complete sensitivity table.
    rows = []
    # Define the two cost structures discussed in the original paper.
    cost_cases = {
        # Use higher protective-capacity costs and lower buffer costs in the first case.
        "PC_cost_gt_buffer_cost": {"Cum": 55.0, "Cdm": 96.0, "cub": 12.0, "cdb": 31.0},
        # Reverse the relative cost structure in the second case.
        "buffer_cost_gt_PC_cost": {"Cum": 12.0, "Cdm": 31.0, "cub": 55.0, "cdb": 96.0},
    }
    # Loop over both cost structures.
    for cost_case, costs in cost_cases.items():
        # Evaluate a less-disruptive and a more-disruptive reliability setting.
        for mttf, mttr in [(4.0, 2.0), (2.0, 4.0)]:
            # Evaluate low, base, and high bottleneck-utilization targets.
            for utilization in [0.75, 0.85, 0.95]:
                # Start from the common base-case parameters.
                params = baseline_parameters()
                # Replace the four cost parameters with the selected cost-case values.
                params.update(costs)
                # Set the upstream mean time between failures for this disruption scenario.
                params["MTTF_um"] = mttf
                # Set the downstream mean time between failures for this disruption scenario.
                params["MTTF_dm"] = mttf
                # Set the upstream mean repair duration for this disruption scenario.
                params["MTTR_um"] = mttr
                # Set the downstream mean repair duration for this disruption scenario.
                params["MTTR_dm"] = mttr
                # Set the target bottleneck utilization for this sensitivity scenario.
                params["U"] = utilization
                # Solve the exact convex model for this parameter combination.
                result = solve_exact_model(params)
                # Add a readable cost-case label to the result record.
                result["cost_case"] = cost_case
                # Add the MTTF input used in this scenario.
                result["MTTF"] = mttf
                # Add the MTTR input used in this scenario.
                result["MTTR"] = mttr
                # Add the utilization target used in this scenario.
                result["U"] = utilization
                # Add the replenishment-reliability target used in this scenario.
                result["rho"] = params["rho"]
                # Append the complete solution to the sensitivity results.
                rows.append(result)
    # Save the complete sensitivity experiment to CSV.
    write_csv(OUTPUT_DIR / "sensitivity_results.csv", rows)
    # Return the sensitivity rows for use in figure generation.
    return rows

# Define revised Figure 7 showing the effect of disruption severity and target utilization on optimal cost.
def make_cost_figure(rows):
    # Create one standalone figure with dimensions suitable for insertion into the Word manuscript.
    plt.figure(figsize=(7.2, 4.6))
    # Plot the less-disruptive and more-disruptive settings separately.
    for mttf, mttr, label in [(4.0, 2.0, "MTTF=4 h, MTTR=2 h"), (2.0, 4.0, "MTTF=2 h, MTTR=4 h")]:
        # Select the baseline cost structure and the requested reliability setting.
        selected = [row for row in rows if row["cost_case"] == "PC_cost_gt_buffer_cost" and row["MTTF"] == mttf and row["MTTR"] == mttr]
        # Sort the selected observations from low to high utilization.
        selected = sorted(selected, key=lambda row: row["U"])
        # Convert utilization fractions to percentages for reader-friendly axis labels.
        x = [100.0 * row["U"] for row in selected]
        # Extract the exact optimal cost for each utilization level.
        y = [row["total_cost"] for row in selected]
        # Draw the sensitivity line for this disruption setting.
        plt.plot(x, y, marker="o", linewidth=1.8, label=label)
    # Label the horizontal axis.
    plt.xlabel("Target utilization (%)")
    # Label the vertical axis.
    plt.ylabel("Optimal design cost ($ per planning period)")
    # Add a descriptive chart title.
    plt.title("Optimal resilience cost increases with utilization and disruption severity")
    # Display a legend identifying the two disruption settings.
    plt.legend()
    # Add a subtle grid to support visual comparison without dominating the figure.
    plt.grid(True, alpha=0.25)
    # Fit all labels and the legend inside the image boundary.
    plt.tight_layout()
    # Save the revised figure at publication-quality resolution.
    plt.savefig(OUTPUT_DIR / "figure7_cost_sensitivity.png", dpi=300)
    # Close the current figure to release memory.
    plt.close()

# Define revised Figure 8 showing how optimal buffers and protective capacities respond to utilization.
def make_design_figure(rows):
    # Select the baseline cost structure and the less-disruptive MTTF/MTTR setting.
    selected = [row for row in rows if row["cost_case"] == "PC_cost_gt_buffer_cost" and row["MTTF"] == 4.0 and row["MTTR"] == 2.0]
    # Sort the selected observations by utilization.
    selected = sorted(selected, key=lambda row: row["U"])
    # Convert utilization fractions to percentages.
    x = [100.0 * row["U"] for row in selected]
    # Create a second standalone manuscript figure.
    plt.figure(figsize=(7.2, 4.6))
    # Plot optimal upstream buffer capacity.
    plt.plot(x, [row["Sub"] for row in selected], marker="o", linewidth=1.8, label="Upstream buffer S_ub")
    # Plot optimal downstream buffer capacity.
    plt.plot(x, [row["Sdb"] for row in selected], marker="o", linewidth=1.8, label="Downstream buffer S_db")
    # Plot optimal upstream incremental protective capacity.
    plt.plot(x, [row["Pum"] for row in selected], marker="o", linewidth=1.8, label="Upstream protective capacity P_um")
    # Plot optimal downstream incremental protective capacity.
    plt.plot(x, [row["Pdm"] for row in selected], marker="o", linewidth=1.8, label="Downstream protective capacity P_dm")
    # Label the horizontal axis.
    plt.xlabel("Target utilization (%)")
    # Label the common design-level vertical axis.
    plt.ylabel("Optimal design level")
    # Add a title that states the principal corrected managerial result.
    plt.title("Higher utilization requires more buffer and protective capacity")
    # Display a legend for the four design decisions.
    plt.legend()
    # Add a subtle grid to improve readability.
    plt.grid(True, alpha=0.25)
    # Tighten the layout before saving the image.
    plt.tight_layout()
    # Save the revised design-sensitivity figure.
    plt.savefig(OUTPUT_DIR / "figure8_design_sensitivity.png", dpi=300)
    # Close the figure to avoid carrying graphical state into later code.
    plt.close()

# Execute all revised numerical work when this file is run as a script.
if __name__ == "__main__":
    # Load the single corrected base-case parameter set.
    base_params = baseline_parameters()
    # Solve the base case before any random validation is performed.
    base_solution = solve_exact_model(base_params)
    # Validate the fixed base-case solution using 100,000 independently generated stochastic cycles.
    validation = validate_solution(base_params, base_solution, n_replications=100000)
    # Merge the base-case inputs and exact solution into one transparent output row.
    baseline_row = {**base_params, **base_solution}
    # Save the exact base-case solution.
    write_csv(OUTPUT_DIR / "baseline_result.csv", [baseline_row])
    # Save the independent Monte Carlo validation results.
    write_csv(OUTPUT_DIR / "baseline_validation.csv", [validation])
    # Solve all 12 sensitivity combinations used in the revised Results section.
    sensitivity_rows = run_sensitivity()
    # Generate the corrected cost-sensitivity figure.
    make_cost_figure(sensitivity_rows)
    # Generate the corrected design-sensitivity figure.
    make_design_figure(sensitivity_rows)
    # Print the headline exact solution for a quick reproducibility check.
    print("Corrected base-case exact solution:", base_solution)
    # Print the independent stochastic validation results.
    print("Monte Carlo validation:", validation)
    # Print the folder containing every generated result and figure.
    print("Outputs saved to:", OUTPUT_DIR)
