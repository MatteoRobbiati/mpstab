import os
import json
import matplotlib.pyplot as plt

results_path = "../../../results/floquet/"
# TODO: refactor way less hard-coded
DEF_COLORS = ["red", "orange", "blue"]

def plot_for_qubits_layers(q:int, l:int, ncircs:list):
    """Plot the results for given nqubits and nlayers."""
    plt.figure(figsize=(4, 4 * 6 / 8))
    ql_label = f"{q}q_{l}l"

    for fold in os.listdir(results_path):
        if ql_label in fold:
            for i, n in enumerate(ncircs):
                if f"nc{n}_" in fold:
                    with open(f"{results_path}/{fold}/results.json", 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if i == 0:
                            plt.errorbar(
                                i, 
                                data["median_noisy_value"],
                                data["median_abs_deviation_noisy_value"],
                                color="black",
                                marker="o",
                                capsize=8,                
                                linestyle='-',           
                                linewidth=1.5,            
                                markeredgecolor = "black",         
                                # label="Noisy"
                            )
                            plt.hlines(data["exact_expval"], 0, len(ncircs), color="black", ls="--", label="Exact value")
                        plt.errorbar(
                            i+1, 
                            data["median_mit_value"],
                            data["median_abs_deviation_mit_value"],
                            color=DEF_COLORS[i],
                            marker="o",
                            capsize=8,                
                            linestyle='-',           
                            linewidth=1.5,   
                            markeredgecolor = "black",         
                            # label=f"{n} circuits"
                        )
    plt.legend(loc=4)
    plt.xticks(list(range(len(ncircs)+1)), ["Noisy"] + ncircs)
    plt.ylabel("Expectation value")
    plt.xlabel("Number of circuits (if TNCDR)")
    plt.savefig(f"{q}q_{l}l_scatter.pdf", bbox_inches="tight")


# run the plot
plot_for_qubits_layers(
    q=7, 
    l=4, 
    ncircs=[10,30,100]
)