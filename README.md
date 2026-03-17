# PyPdbComplex : Protein-Protein Complex Analysis Toolkit

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PyPdbComplex** is a comprehensive Python toolkit for analyzing protein structures, with a focus on antibody-antigen complexes. It provides easy-to-use functions for extracting detailed structural features, analyzing molecular interactions, and comparing multiple structures.

## 🚀 Features

### Core Capabilities
- **PDB Parsing**: Robust parsing with support for altloc, HETATM, and complex structures
- **Selection System**: Intuitive syntax for selecting chains, residues, and atoms
- **Contact Analysis**: Detect H-bonds, salt bridges, hydrophobic contacts, pi-stacking, disulfides
- **Interface Analysis**: Identify interface residues and calculate buried surface area
- **SASA Calculation**: Solvent accessible surface area with bound/unbound comparison
- **Geometry Analysis**: Backbone angles (φ, ψ, ω), sidechain rotamers (χ), Ramachandran plots
- **VdW Energy**: Lennard-Jones energy decomposition
- **Distance Analysis**: Pairwise distances and distance matrices

### Advanced Features
- **Residue Features**: Comprehensive feature extraction combining all analyses
- **Single Chain Analysis**: Detailed characterization of individual protein chains
- **Multi-Complex Comparison**: Side-by-side comparison of multiple structures
- **Hotspot Identification**: Automatic identification of critical binding residues
- **Interactive Visualizations**: HTML dashboards with plotly

## 📦 Installation

### From Source

```bash
# Clone or download the repository
git clone https://github.com/fbabd/PyPdbComplex.git
cd PyPdbComplex

# Install in development mode
pip install -e .

# Or install with all optional dependencies
pip install -e ".[all]"
```

### Requirements

**Core dependencies** (automatically installed):
- numpy >= 1.20.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0

**Optional dependencies**:
- freesasa >= 2.1.0 (for fast SASA calculations)
- plotly >= 5.0.0 (for interactive visualizations)
- seaborn >= 0.11.0 (for statistical plots)

## 🎯 Quick Start

### Basic Usage

```python
from PyPdbComplex.pdbparser import parse_pdb
from PyPdbComplex.interface import compute_interface
from PyPdbComplex.contacts import analyze_contacts

# Parse PDB file
cx = parse_pdb("protein.pdb")

# Analyze interface between chains
interface = compute_interface(
    cx,
    selection_A=["H", "L"],  # Antibody
    selection_B=["A"],        # Antigen
    cutoff=5.0
)

print(f"Interface residues: {interface.total_contacts}")
print(f"Buried surface area: {interface.bsa_total:.2f} Ų")

# Detect molecular interactions
contacts = analyze_contacts(cx, ["H", "L"], ["A"])
print(f"H-bonds: {contacts.get_contact_counts()['hydrogen_bond']}")
```

### Comprehensive Feature Extraction

```python
from PyPdbComplex.residue_features import (
    extract_residue_features,
    features_to_dataframe,
    identify_hotspots
)

# Extract all features
features = extract_residue_features(
    cx,
    selection_A=["H", "L"],
    selection_B=["A"],
    compute_sasa=True,
    compute_geometry=True,
    compute_vdw=True
)

# Convert to DataFrame
df = features_to_dataframe(features)

# Find hotspots
hotspots = identify_hotspots(features, min_interactions=3)
print(f"Identified {len(hotspots)} hotspot residues")
```

### Multi-Complex Comparison

```python
from PyPdbComplex.multicomplex.multi_analysis import (
    MultiComplexAnalyzer,
    AnalysisConfig
)

# Load multiple structures
analyzer = MultiComplexAnalyzer([
    ("Wild-Type", "wt.pdb"),
    ("Mutant", "mutant.pdb")
])

# Configure and run analysis
config = AnalysisConfig(
    calculate_sasa=True,
    calculate_contacts=True,
    calculate_interface=True,
    calculate_vdw=True
)

results = analyzer.run_analysis(config)

# Compare results
for name, data in results.items():
    print(f"{name}: {data.summary}")
```

## 📚 Examples

The `examples/` directory contains comprehensive Jupyter notebooks demonstrating all features:

1. **nb1-pdb_file_processing.ipynb** - PDB parsing and navigation
2. **nb2-contact_analysis.ipynb** - Molecular interactions
3. **nb3-interface_analysis.ipynb** - Interface characterization
4. **nb4-geometry_analysis.ipynb** - Structural geometry
5. **nb5-distance_analysis.ipynb** - Distance calculations
6. **nb6-sasa_analysis.ipynb** - Surface accessibility
7. **nb7-vdw_analysis.ipynb** - Van der Waals energies
8. **nb8-residue_features.ipynb** - Comprehensive feature extraction
9. **nb9-single_chain_features.ipynb** - Single protein analysis
10. **nb10-multi_complex_comparison.ipynb** - Multi-structure comparison

## 🔧 API Overview

### Core Modules

- **`pdbparser`**: Parse PDB files → Complex objects
- **`selection`**: Select specific atoms, residues, chains
- **`contacts`**: Detect molecular interactions
- **`interface`**: Analyze protein-protein interfaces
- **`sasa`**: Calculate solvent accessibility
- **`geometry`**: Compute structural angles
- **`distances`**: Calculate distances and distance matrices
- **`vdw`**: Van der Waals energy calculations

### Advanced Modules

- **`residue_features`**: Unified feature extraction for complexes
- **`single_chain_features`**: Feature extraction for single chains
- **`multicomplex`**: Compare multiple structures

### Visualization

- **`visualization.contacts_viz`**: Contact network plots
- **`visualization.interface_viz`**: Interface heatmaps
- **`visualization.geometry_viz`**: Ramachandran plots
- **`visualization.residue_feat_viz`**: Interactive HTML dashboards

## 📊 Use Cases

- **Antibody Engineering**: Analyze and optimize antibody-antigen binding
- **Mutation Effect Prediction**: Compare wild-type vs mutants
- **Drug Discovery**: Identify binding hotspots and druggable sites
- **Protein Design**: Evaluate designed protein interfaces
- **Structure Quality Assessment**: Validate protein structures
- **Machine Learning**: Generate feature matrices for ML models

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 Citation

If you use PyPdbComplex in your research, please cite:

```bibtex

```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for the structural biology and computational chemistry communities

## 📞 Support

- **Documentation**: See `examples/` for comprehensive tutorials
- **Issues**: Report bugs at https://github.com/fbabd/PyPdbComplex/issues
- **Questions**: Open a discussion on GitHub

---

**Made with ❤️ for protein structure analysis**
