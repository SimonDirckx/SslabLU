import numpy as np
import matplotlib.pyplot as plt
import os

import pickle

plot_directory = "output_kappa10_beta1_plots"  # change as needed
os.makedirs(plot_directory, exist_ok=True)

with open("output_kappa10_beta1.pkl", "rb") as f:
    data = pickle.load(f)

    Nx_list = data["Nx_list"]
    slab_list = data["slab_list"]

    variables = {
        "True Residual (Tridiagonal)": data["true_residual_tridiagonals"],
        "Backward Residual (Tridiagonal)": data["residual_tridiagonals"],
        "Forward Error (Tridiagonal)": data["error_tridiagonals"],
        "Factor Time (Tridiagonal)": data["factor_time_tridiagonals"],
        "Run Time (Tridiagonal)": data["run_time_tridiagonals"],
        "Condition Numbers": data["conds"],
        "True Residual (Red-Black)": data["true_residual_redblacks"],
        "Backward Residual (Red-Black)": data["residual_redblacks"],
        "Forward Error (Red-Black)": data["error_redblacks"],
        "Factor Time (Red-Black)": data["factor_time_redblacks"],
        "Run Time (Red-Black)": data["run_time_redblacks"],
        "Block Condition Number (Tridiagonal)": data["block_cond_tridiagonals"],
        "Block Condition Number (Red-Black)": data["block_cond_redblacks"],
    }

    log_scale = {
        "True Residual (Tridiagonal)": True,
        "Backward Residual (Tridiagonal)": True,
        "Forward Error (Tridiagonal)": True,
        "Factor Time (Tridiagonal)": False,
        "Run Time (Tridiagonal)": False,
        "Condition Numbers": True,
        "True Residual (Red-Black)": True,
        "Backward Residual (Red-Black)": True,
        "Forward Error (Red-Black)": True,
        "Factor Time (Red-Black)": False,
        "Run Time (Red-Black)": False,
        "Block Condition Number (Tridiagonal)": True,
        "Block Condition Number (Red-Black)": True,
    }

    
    for title, data in variables.items():
        fig, ax = plt.subplots()
        for i, Nx in enumerate(Nx_list):
            ax.plot(slab_list, data[i], marker='o', label=f"Nx={Nx}")

        ax.set_xscale("log")
        
        if log_scale[title]:
            ax.set_yscale("log")
        
        ax.set_xlabel("Number of Slabs")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(slab_list)
        ax.legend()
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, f"{title.replace(' ', '_').replace('(', '').replace(')', '')}.png"), dpi=150)
        plt.show()
    

    conds              = variables["Condition Numbers"].flatten()
    wanted_keys        = ["Forward Error (Tridiagonal)", "Forward Error (Red-Black)"]
    filtered_variables = {key: value for key, value in variables.items() if key in wanted_keys}

    max_cond = np.max(conds)

    fig, ax = plt.subplots()
    for title, data in filtered_variables.items():
        ax.scatter(conds, data.flatten(), marker='o', label=title)

    ax.plot([0, max_cond], [0, max_cond * 1e-16], label="kappa x eps_machine")
    ax.set_xscale("log")
    ax.set_yscale("log") 
    ax.set_xlabel("Condition Number of A")
    ax.set_ylabel("Forward Error")
    ax.set_title("Forward Error to kappa(A)")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_directory, "forward_errors_to_cond.png"), dpi=150)
    plt.show()