# PyPDBcomplex Example Notebooks

This directory contains comprehensive Jupyter notebooks demonstrating all features of the `PyPDBcomplex` package for protein structure analysis.

## 📚 Notebook Overview

### Basic Analysis
1. **[nb1-pdb_file_processing.ipynb](nb1-pdb_file_processing.ipynb)**
   - PDB file parsing and navigation
   - Hierarchical data model (Complex → Chains → Residues → Atoms)
   - Selection system basics
   - Named groups for regions (CDRs, domains)

2. **[nb2-contact_analysis.ipynb](nb2-contact_analysis.ipynb)**
   - Molecular interaction detection
   - H-bonds, salt bridges, hydrophobic contacts
   - Pi-stacking and disulfide bonds
   - Contact visualization

3. **[nb3-interface_analysis.ipynb](nb3-interface_analysis.ipynb)**
   - Interface residue identification
   - Contact counting and partner tracking
   - Buried surface area calculation
   - Interface visualization

4. **[nb4-geometry_analysis.ipynb](nb4-geometry_analysis.ipynb)**
   - Backbone dihedral angles (φ, ψ, ω)
   - Ramachandran plots
   - Sidechain rotamers (χ1-4)
   - Bend angle analysis

5. **[nb5-distance_analysis.ipynb](nb5-distance_analysis.ipynb)**
   - Distance matrix calculations
   - Pairwise minimum distances
   - CA-CA distance analysis
   - Distance-based filtering

6. **[nb6-sasa_analysis.ipynb](nb6-sasa_analysis.ipynb)**
   - Solvent accessible surface area (SASA)
   - Bound vs unbound SASA comparison
   - Burial fraction calculation
   - Per-residue accessibility

7. **[nb7-vdw_analysis.ipynb](nb7-vdw_analysis.ipynb)**
   - Van der Waals energy calculations
   - Lennard-Jones decomposition
   - Per-residue VdW contributions
   - Energy visualization

### Advanced Features
8. **[nb8-residue_features.ipynb](nb8-residue_features.ipynb)**
   - **Comprehensive residue feature extraction for protein complexes**
   - Combines ALL analyses into unified feature representation
   - Interface-focused analysis
   - Hotspot identification
   - Interactive HTML dashboard generation
   - Machine learning-ready feature matrices


10. **[nb9-multi_complex_comparison.ipynb](nb10-multi_complex_comparison.ipynb)**
    - **Multi-structure comparison and analysis**
    - Wild-type vs mutant comparison
    - Side-by-side metric comparison
    - Interaction change identification
    - Statistical analysis and visualization

## 🚀 Getting Started

### Prerequisites
```bash
pip install PyPDBcomplex pandas matplotlib numpy seaborn
```

### Running the Notebooks

1. **Start with basics**: Begin with `nb1-pdb_file_processing.ipynb` to understand data structures
2. **Explore specific analyses**: Choose notebooks based on your needs
3. **Advanced workflows**: Use nb8-10 for comprehensive analysis

### Example Data
All notebooks use the provided PDB files:
- `5GGS_wt.pdb` - Wild-type antibody-antigen complex
- `212__5GGS__H_100_Y_H.pdb` - H100Y mutant

## 📊 Workflow Recommendations

### For Protein-Protein Interactions:
1. Start with **nb1** (file processing)
2. Run **nb3** (interface analysis)
3. Use **nb2** (contacts) for detailed interactions
4. Apply **nb8** (residue features) for comprehensive analysis

### For Single Protein Analysis:
1. Start with **nb1** (file processing)
2. Run **nb4** (geometry) for structure validation
3. Use **nb6** (SASA) for accessibility
4. Apply **nb9** (single chain features) for comprehensive analysis

### For Comparing Structures:
1. Use **nb10** (multi-complex comparison)
2. Combine with **nb8** (residue features) for detailed differences

## 🎯 Common Use Cases

| Use Case | Recommended Notebooks |
|----------|----------------------|
| Antibody-antigen binding analysis | nb1, nb3, nb2, nb8 |
| Mutation effect prediction | nb10, nb8 |
| Protein stability analysis | nb9, nb4, nb6 |
| Structure quality assessment | nb1, nb4, nb9 |
| Hotspot identification | nb8 |
| High-throughput screening | nb10 |
| Machine learning features | nb8, nb9 |

## 📝 Output Files

Notebooks generate various output files:
- **CSV files**: Tabular data for analysis
- **JSON files**: Structured data export
- **HTML dashboards**: Interactive visualizations
- **PNG/PDF plots**: Publication-ready figures

## 🔧 Customization

All notebooks are fully customizable:
- Adjust analysis parameters
- Modify visualizations
- Change output formats
- Extend with custom analyses

## 📖 Documentation

For detailed API documentation, see the main package documentation.

## 💡 Tips

- **Jupyter Lab recommended**: Better for side-by-side viewing
- **Run cells sequentially**: Notebooks build on previous cells
- **Modify parameters**: Experiment with different settings
- **Check outputs**: Review generated CSV files for detailed data

## 🤝 Contributing

Found an issue or have a suggestion? Please open an issue in the main repository.

---

**Note**: These notebooks are designed to be both educational and production-ready. Feel free to adapt them for your specific research needs!
