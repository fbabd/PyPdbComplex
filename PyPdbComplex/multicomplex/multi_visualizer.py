"""
Modular Visualization System for Multi-Complex Comparison
Generates figures based on selected analyses
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional
import numpy as np


class MultiComplexVisualizer:
    """Generate visualizations for multi-complex comparison results"""
    
    def __init__(self, results: Dict):
        """
        Initialize visualizer with analysis results
        
        Args:
            results: Dictionary from MultiComplexAnalyzer.export_results()
        """
        self.results = results
        self.complexes = results['complexes']
        self.wt_name = results['wt_name']
        self.complex_names = list(self.complexes.keys())
        
        # Color scheme
        self.colors = px.colors.qualitative.Set2
    
    
    def generate_summary_table(self) -> str:
        """Generate HTML table with summary statistics"""
        
        # Collect all summary data
        rows = []
        for name in self.complex_names:
            summary = self.complexes[name]['summary']
            row = {'Complex': name}
            row.update(summary)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Reorder columns
        priority_cols = ['Complex', 'total_buried_sasa', 
                        'interface_residues', 'total_contacts', 'total_vdw_energy']
        other_cols = [c for c in df.columns if c not in priority_cols]
        ordered_cols = [c for c in priority_cols if c in df.columns] + other_cols
        df = df[ordered_cols]
        
        # Format numbers
        for col in df.columns:
            if col != 'Complex' and df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].round(2)
        
        # Highlight WT row
        def highlight_wt(row):
            if row['Complex'] == self.wt_name:
                return ['background-color: #d4edda'] * len(row)
            return [''] * len(row)
        
        styled_df = df.style.apply(highlight_wt, axis=1)
        
        return styled_df.to_html(index=False, classes='table table-striped table-hover')
    
    
 
    
    def plot_sasa_comparison(self) -> Optional[str]:
        """Generate SASA comparison chart"""
        
        # Check if SASA data exists
        if not any('sasa' in self.complexes[name] for name in self.complex_names):
            return None
        
        # Extract buried SASA totals
        names = []
        buried_sasa = []
        
        for name in self.complex_names:
            summary = self.complexes[name]['summary']
            if 'total_buried_sasa' in summary:
                names.append(name)
                buried_sasa.append(summary['total_buried_sasa'])
        
        if not names:
            return None
        
        # Create bar chart
        fig = go.Figure()
        
        colors = [self.colors[0] if name == self.wt_name else self.colors[2] 
                 for name in names]
        
        fig.add_trace(go.Bar(
            x=names,
            y=buried_sasa,
            marker_color=colors,
            text=[f'{v:.1f} Ų' for v in buried_sasa],
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Total Buried Surface Area',
            xaxis_title='Structure',
            yaxis_title='Buried SASA (Ų)',
            height=400,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    
    def plot_contact_comparison(self) -> Optional[str]:
        """Generate contact type comparison chart"""
        
        # Check if contact data exists
        if not any('contacts' in self.complexes[name] for name in self.complex_names):
            return None
        
        # Collect contact data
        contact_types = set()
        data = {name: {} for name in self.complex_names}
        
        for name in self.complex_names:
            summary = self.complexes[name]['summary']
            for key, value in summary.items():
                if key.startswith('contacts_'):
                    contact_type = key.replace('contacts_', '')
                    contact_types.add(contact_type)
                    data[name][contact_type] = value
        
        if not contact_types:
            return None
        
        # Create grouped bar chart
        fig = go.Figure()
        
        for i, contact_type in enumerate(sorted(contact_types)):
            values = [data[name].get(contact_type, 0) for name in self.complex_names]
            
            fig.add_trace(go.Bar(
                name=contact_type,
                x=self.complex_names,
                y=values,
                marker_color=self.colors[i % len(self.colors)],
                text=values,
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Inter-chain Contacts by Type',
            xaxis_title='Structure',
            yaxis_title='Number of Contacts',
            barmode='group',
            height=500,
            template='plotly_white',
            legend_title='Contact Type'
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    
    def plot_interface_comparison(self) -> Optional[str]:
        """Generate interface metrics comparison"""
        
        # Check if interface data exists
        if not any('interface' in self.complexes[name] for name in self.complex_names):
            return None
        
        # Extract data
        names = []
        interface_residues = []
        buried_area = []
        
        for name in self.complex_names:
            summary = self.complexes[name]['summary']
            if 'interface_residues' in summary and 'buried_surface_area' in summary:
                names.append(name)
                interface_residues.append(summary['interface_residues'])
                buried_area.append(summary['buried_surface_area'])
        
        if not names:
            return None
        
        # Create subplot with two y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(
                name='Interface Residues',
                x=names,
                y=interface_residues,
                marker_color=self.colors[3],
                text=interface_residues,
                textposition='outside'
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                name='Buried Surface Area',
                x=names,
                y=buried_area,
                mode='lines+markers',
                marker=dict(size=10, color=self.colors[4]),
                line=dict(width=3)
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title='Interface Properties',
            height=500,
            template='plotly_white',
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text='Structure')
        fig.update_yaxes(title_text='Number of Residues', secondary_y=False)
        fig.update_yaxes(title_text='Buried Area (Ų)', secondary_y=True)
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    
    def plot_vdw_comparison(self) -> Optional[str]:
        """Generate VDW energy comparison"""
        
        # Check if VDW data exists
        if not any('vdw' in self.complexes[name] for name in self.complex_names):
            return None
        
        # Extract VDW data
        names = []
        total_vdw = []
        favorable_vdw = []
        
        for name in self.complex_names:
            summary = self.complexes[name]['summary']
            if 'total_vdw_energy' in summary:
                names.append(name)
                total_vdw.append(summary['total_vdw_energy'])
                favorable_vdw.append(summary.get('favorable_vdw_energy', 0))
        
        if not names:
            return None
        
        # Create grouped bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Total VDW',
            x=names,
            y=total_vdw,
            marker_color=self.colors[5],
            text=[f'{v:.1f}' for v in total_vdw],
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name='Favorable VDW',
            x=names,
            y=favorable_vdw,
            marker_color=self.colors[6],
            text=[f'{v:.1f}' for v in favorable_vdw],
            textposition='outside'
        ))
        
        fig.update_layout(
            title='VDW Energy Comparison',
            xaxis_title='Structure',
            yaxis_title='Energy (kcal/mol)',
            barmode='group',
            height=500,
            template='plotly_white',
            legend_title='Energy Type'
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    
    def plot_heatmap_comparison(self, metric: str = 'buried_sasa') -> Optional[str]:
        """
        Generate heatmap comparing residue-level metrics across structures
        
        Args:
            metric: 'buried_sasa', 'vdw_energy', or 'interface'
        """
        
        # Collect residue data
        all_residues = set()
        data_matrix = {}
        
        for name in self.complex_names:
            if metric == 'buried_sasa':
                res_data = self.complexes[name].get('sasa', {})
                for res_id, values in res_data.items():
                    all_residues.add(res_id)
                    if name not in data_matrix:
                        data_matrix[name] = {}
                    data_matrix[name][res_id] = values.get('buried_sasa', 0)
            
            elif metric == 'vdw_energy':
                res_data = self.complexes[name].get('vdw', {})
                for res_id, energy in res_data.items():
                    all_residues.add(res_id)
                    if name not in data_matrix:
                        data_matrix[name] = {}
                    data_matrix[name][res_id] = energy
        
        if not all_residues:
            return None
        
        # Create matrix for plotting
        residues_sorted = sorted(all_residues)
        z_matrix = []
        
        for name in self.complex_names:
            row = [data_matrix.get(name, {}).get(res, 0) for res in residues_sorted]
            z_matrix.append(row)
        
        # Limit to top N residues for readability
        max_residues = 50
        if len(residues_sorted) > max_residues:
            # Sort by variance across structures
            variances = []
            for i, res in enumerate(residues_sorted):
                values = [z_matrix[j][i] for j in range(len(self.complex_names))]
                variances.append((np.var(values), res, i))
            
            variances.sort(reverse=True)
            top_indices = sorted([idx for _, _, idx in variances[:max_residues]])
            
            z_matrix = [[row[i] for i in top_indices] for row in z_matrix]
            residues_sorted = [residues_sorted[i] for i in top_indices]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z_matrix,
            x=residues_sorted,
            y=self.complex_names,
            colorscale='RdBu_r' if metric == 'vdw_energy' else 'Viridis',
            hovertemplate='Structure: %{y}<br>Residue: %{x}<br>Value: %{z:.2f}<extra></extra>'
        ))
        
        metric_titles = {
            'buried_sasa': 'Buried SASA (Ų)',
            'vdw_energy': 'VDW Energy (kcal/mol)',
            'interface': 'Interface Score'
        }
        
        fig.update_layout(
            title=f'{metric_titles.get(metric, metric)} Comparison (Top {len(residues_sorted)} Residues)',
            xaxis_title='Residue',
            yaxis_title='Structure',
            height=400 + len(self.complex_names) * 30,
            template='plotly_white'
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    
    def generate_all_plots(self) -> Dict[str, Optional[str]]:
        """
        Generate all available plots based on data present
        
        Returns:
            Dictionary of plot_name -> HTML string (or None if data not available)
        """
        plots = {
            'summary_table': self.generate_summary_table(),
            'sasa_comparison': self.plot_sasa_comparison(),
            'contact_comparison': self.plot_contact_comparison(),
            'interface_comparison': self.plot_interface_comparison(),
            'vdw_comparison': self.plot_vdw_comparison(),
            'sasa_heatmap': self.plot_heatmap_comparison('buried_sasa'),
            'vdw_heatmap': self.plot_heatmap_comparison('vdw_energy')
        }
        
        return plots
    