![QuForge Logo](https://github.com/tiago939/QuForge/blob/main/logo.png)

# QuForge v0.3.5
QuForge is a Python-based quantum simulator designed for qudits, extending quantum computing capabilities beyond traditional qubits. It facilitates the simulation of quantum circuits with customizable qudit dimensions, supporting both dense and sparse operations for efficient computation. Built on the differentiable framework PyTorch, QuForge enables execution on accelerating devices such as GPUs and TPUs, enhancing simulation speed.

For detailed documentation, visit [quforge.readthedocs.io](https://quforge.readthedocs.io).

# Installation

Install QuForge using pip:

```bash
pip install quforge
```

To install the latest version directly from the source:


```bash
# Clone the repository
git clone https://github.com/tiago939/QuForge.git

# Navigate to the project directory
cd QuForge

# Install the package
pip install .
```

Ensure that your environment meets the necessary dependencies listed in requirements.txt.

# Usage

Here's an example of how to create and execute a simple quantum circuit using QuForge:

```bash

import quforge.quforge as qf

# Initialize a 2-level (qubit) circuit with 2 wires
circuit = qf.Circuit(dim=2, wires=2)

# Add gates to the circuit
circuit.H(index=[0])           # Hadamard gate on wire 0
circuit.CNOT(index=[0, 1])   # CNOT gate with control wire 0 and target wire 1

# Initialize the state
state = qf.State('0-0', dim=2)

# Execute the circuit
result = circuit(state)

print(result)
```

This script sets up a quantum circuit with a Hadamard gate and a CNOT gate, then executes it on an initial state.


# Directory Structure

The QuForge repository is organized as follows:

```bash
QuForge/
├── docs/                   # Documentation files
├── examples/               # Example scripts demonstrating QuForge usage
├── quforge/                # Main source code for the QuForge library
│   ├── __init__.py         # Initializes the quforge package
│   └── ...                 # Other source files
├── .gitignore              # Specifies files and directories to ignore in Git
├── .readthedocs.yaml       # Configuration for Read the Docs documentation hosting
├── LICENSE                 # License information (Apache-2.0)
├── README.md               # Overview and instructions for the project
├── logo.png                # QuForge logo image
├── requirements.txt        # List of Python dependencies
└── setup.py                # Installation script for the QuForge package
```

# Contributions and Support

We are continuously working on optimizing the library. Please reach out if you have any suggestions or questions!


# Citation

If you use QuForge in your research, please cite our work:

```
@misc{2024quforge,
      title={QuForge: A Library for Qudits Simulation},
      author={Tiago de Souza Farias and Lucas Friedrich and Jonas Maziero},
      year={2024},
      eprint={2409.17716},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2409.17716},
}
```


