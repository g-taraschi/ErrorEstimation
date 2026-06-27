import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    uniform_data = np.array(
		[
[544, 1.119e-03],
[2112, 6.063e-05],
[8320, 3.596e-06],
		],
		dtype=float,
	)

    adaptive_data = np.array(
        [
[544, 1.119e-03],
[632, 1.039e-03],
[752, 7.131e-05],
[916, 2.711e-05],
[1128, 7.713e-06],
[1474, 5.857e-06],
[2040, 8.566e-06],
[2918, 2.401e-06],
[4024, 1.047e-06],
[5610, 1.466e-06],
        ],
        dtype=float,
    )

    prager_data = np.array(
        [
[544, 1.119e-03],
[676, 1.149e-03],
[912, 5.489e-05],
[1272, 6.399e-06],
[2136, 8.776e-06],
[3666, 1.019e-06],
[6396, 1.725e-06],
[10954, 6.950e-07],
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(uniform_data[:, 0], uniform_data[:, 1], marker="s", linewidth=2, label="Uniform refinement")
    ax.loglog(adaptive_data[:, 0], adaptive_data[:, 1], marker="o", linewidth=2, label="Goal-oriented")
    ax.loglog(prager_data[:, 0], prager_data[:, 1], marker="^", linewidth=2, label="Prager-Synge")

    ax.set_xlabel("DoFs")
    ax.set_ylabel("Error")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    plt.savefig("error_vs_dofs.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
	main()
