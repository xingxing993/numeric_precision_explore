import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from numeric_format import (
    INTFormat, FPFormat, FPFNFormat, FPFNUZFormat, HIF8Format,
    NumericValue
)
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import re

# Set page configuration
st.set_page_config(
    page_title="Numeric Format Visualizer",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .format-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .comparison-section {
        border: 2px solid #1f77b4;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .disabled-format {
        opacity: 0.5;
        pointer-events: none;
    }
    .format-selector {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .custom-format-section {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_numeric_formats() -> Dict[str, object]:
    """Create all available numeric formats."""
    formats = {}
    
    # Integer formats
    formats.update({
        'INT8': INTFormat('INT8', 8, signed=True),
        'UINT8': INTFormat('UINT8', 8, signed=False),
        'INT16': INTFormat('INT16', 16, signed=True),
        'UINT16': INTFormat('UINT16', 16, signed=False),
        'INT32': INTFormat('INT32', 32, signed=True),
        'UINT32': INTFormat('UINT32', 32, signed=False),
    })
    
    # Standard floating-point formats
    formats.update({
        'FP4_E2M1 (IEEE754)': FPFormat('FP4_E2M1 (IEEE754)', 2, 1),
        'FP8_E4M3 (IEEE754)': FPFormat('FP8_E4M3 (IEEE754)', 4, 3),
        'FP8_E5M2 (IEEE754)': FPFormat('FP8_E5M2 (IEEE754)', 5, 2),
        'FP16 (IEEE754)': FPFormat('FP16 (IEEE754)', 5, 10),
        'BF16 (IEEE754)': FPFormat('BF16 (IEEE754)', 8, 7),
        'FP32 (IEEE754)': FPFormat('FP32 (IEEE754)', 8, 23),
        'FP64 (IEEE754)': FPFormat('FP64 (IEEE754)', 11, 52),
    })
    
    # Special FP8 formats
    formats.update({
        'FP8_E4M3FN': FPFNFormat('FP8_E4M3FN', 4, 3),
        'FP8_E4M3FNUZ': FPFNUZFormat('FP8_E4M3FNUZ', 4, 3),
    })
    
    # HiFloat8 format (special non-IEEE format)
    formats.update({
        'HiFloat8': HIF8Format('HiFloat8'),
    })
    
    return formats

def create_custom_format(format_type: str, **kwargs) -> Optional[object]:
    """Create a custom numeric format based on user specifications."""
    try:
        if format_type == "Integer":
            bits = kwargs.get('bits', 8)
            signed = kwargs.get('signed', True)
            if bits < 4 or bits > 64:
                raise ValueError(f"Bits must be between 4 and 64, got {bits}")
            return INTFormat(f"Custom_INT{bits}", bits, signed)
        elif format_type == "Floating Point":
            exp_bits = kwargs.get('exp_bits', 4)
            mantissa_bits = kwargs.get('mantissa_bits', 3)
            if exp_bits < 2 or exp_bits > 15:
                raise ValueError(f"Exponent bits must be between 2 and 15, got {exp_bits}")
            if mantissa_bits < 1 or mantissa_bits > 52:
                raise ValueError(f"Mantissa bits must be between 1 and 52, got {mantissa_bits}")
            return FPFormat(f"Custom_FP{exp_bits}M{mantissa_bits}", exp_bits, mantissa_bits)
        elif format_type == "HiFloat8":
            # HiFloat8 is a fixed 8-bit format, no customization needed
            return HIF8Format("Custom_HiFloat8")
        else:
            raise ValueError(f"Unknown format type: {format_type}")
    except Exception as e:
        st.error(f"Error creating custom format: {str(e)}")
        return None

def get_format_values(format_obj, max_values: int = 10000) -> Tuple[np.ndarray, List[str]]:
    """Get representable values for a format, with sampling for large formats."""
    try:
        if hasattr(format_obj, 'enumerate_values'):
            values = []
            categories = []
            
            for i, numeric_val in enumerate(format_obj.enumerate_values()):
                if i >= max_values:  # Limit for very large formats
                    break
                if numeric_val.category in ['normal', 'subnormal', 'zero', 'integer']:
                    values.append(numeric_val.value)
                    categories.append(numeric_val.category)
            
            return np.array(values), categories
        else:
            return np.array([]), []
    except Exception as e:
        st.error(f"Error getting values for {format_obj.name}: {str(e)}")
        return np.array([]), []

def get_format_values_optimized(format_obj, max_values: int = 10000) -> Tuple[np.ndarray, List[str]]:
    """Get representable values for a format with memory optimization for wide bit widths."""
    try:
        if format_obj.total_bits > 16:
            # Use analytical counting for wide formats to save memory
            return get_format_values_analytical(format_obj, max_values)
        else:
            # Use enumeration for narrow formats
            return get_format_values(format_obj, max_values)
    except Exception as e:
        st.error(f"Error getting values for {format_obj.name}: {str(e)}")
        return np.array([]), []

def get_format_values_analytical(format_obj, max_values: int = 10000) -> Tuple[np.ndarray, List[str]]:
    """Get representative values using analytical methods for wide formats.
    
    For wide formats, this returns minimal placeholder data since the plotting functions
    will use analytical counting via get_analytical_histogram_data instead.
    """
    try:
        # For wide formats, return minimal placeholder data
        # The plotting functions will use analytical counting instead
        return np.array([0.0]), ['zero']
    except Exception as e:
        st.error(f"Error in analytical value generation: {str(e)}")
        return np.array([]), []


def get_analytical_histogram_data(format_obj, bins: int = 100) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Get histogram data using analytical counting for wide formats."""
    try:
        min_val = format_obj.min_value
        max_val = format_obj.max_value
        
        # Create bins that are symmetric around zero
        max_abs_val = max(abs(min_val), abs(max_val))
        
        if min_val < 0 and max_val > 0:
            # For formats spanning both positive and negative
            bin_edges = np.linspace(-max_abs_val, max_abs_val, bins + 1)
        else:
            # For formats with only positive or only negative values
            bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        # Calculate counts for each bin using analytical method
        counts = np.zeros(bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        for i in range(bins):
            bin_min = bin_edges[i]
            bin_max = bin_edges[i + 1]
            counts[i] = format_obj.count_values_between(bin_min, bin_max)
        
        # Assign categories to bin centers
        categories = []
        for center in bin_centers:
            if center == 0:
                categories.append('zero')
            elif hasattr(format_obj, 'get_min_normal'):
                if abs(center) < format_obj.get_min_normal():
                    categories.append('subnormal')
                else:
                    categories.append('normal')
            else:
                categories.append('integer')
        
        return bin_centers, counts, categories
    except Exception as e:
        st.error(f"Error in analytical histogram generation: {str(e)}")
        return np.array([]), np.array([]), []

def create_scatter_plot(format_obj, values: np.ndarray, categories: List[str], 
                       title: str, color_scheme: str = "default", ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Create a scatter plot for numeric values."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        return_fig = True
    else:
        return_fig = False
    
    # Color mapping
    if color_scheme == "default":
        colors = {'normal': 'blue', 'subnormal': 'green', 'zero': 'red', 'integer': 'purple'}
    elif color_scheme == "viridis":
        colors = {'normal': 'darkblue', 'subnormal': 'green', 'zero': 'red', 'integer': 'purple'}
    else:  # rainbow
        colors = {'normal': 'red', 'subnormal': 'orange', 'zero': 'yellow', 'integer': 'green'}
    
    # Plot each category
    for cat in set(categories):
        mask = [c == cat for c in categories]
        if any(mask):
            ax.scatter(values[mask], [0]*sum(mask), 
                      c=colors.get(cat, 'gray'), label=cat, alpha=0.7, s=30)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('')
    ax.set_ylim(-0.5, 0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add format info
    info_text = f'Range: [{format_obj.min_value:.3e}, {format_obj.max_value:.3e}]\n'
    info_text += f'Total values: {len(values)}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    if return_fig:
        return fig
    return None

def create_histogram_plot(format_obj, values: np.ndarray, categories: List[str], 
                         title: str, bins: int = 100, ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Create a histogram plot with bins centered at zero for symmetry."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        return_fig = True
    else:
        return_fig = False
    
    if len(values) == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        if return_fig:
            return fig
        return None
    
    # Use analytical counting for wide formats, regular histogram for narrow formats
    if format_obj.total_bits > 16:
        # Use analytical method for wide formats
        bin_centers, hist, bin_categories = get_analytical_histogram_data(format_obj, bins)
        bin_width = (bin_centers[1] - bin_centers[0]) if len(bin_centers) > 1 else 1.0
    else:
        # Use regular histogram for narrow formats
        max_abs_val = max(abs(format_obj.min_value), abs(format_obj.max_value))
        
        # Create bins that are symmetric around zero
        if format_obj.min_value < 0 and format_obj.max_value > 0:
            # For formats spanning both positive and negative
            bin_edges = np.linspace(-max_abs_val, max_abs_val, bins + 1)
        else:
            # For formats with only positive or only negative values
            bin_edges = np.linspace(format_obj.min_value, format_obj.max_value, bins + 1)
        
        # Create histogram
        hist, bin_edges = np.histogram(values, bins=bin_edges, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        bin_categories = categories
    
    # Plot histogram bars
    ax.bar(bin_centers, hist, width=bin_width, 
           alpha=0.7, color='skyblue', edgecolor='navy')
    
    # Add category markers
    for cat in set(bin_categories):
        cat_values = [v for v, c in zip(bin_centers, bin_categories) if c == cat]
        if cat_values:
            ax.scatter(cat_values, [0]*len(cat_values), 
                      c='red' if cat == 'zero' else 'green' if cat == 'subnormal' else 'blue',
                      marker='|', s=100, linewidth=2, label=f'{cat} values')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add format statistics
    stats_text = f'Min: {format_obj.min_value:.3e}\n'
    stats_text += f'Max: {format_obj.max_value:.3e}\n'
    if format_obj.total_bits > 16:
        stats_text += f'Total values: {sum(hist)} (analytical)'
    else:
        stats_text += f'Values: {len(values)}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    if return_fig:
        return fig
    return None

def create_density_plot(format_obj, values: np.ndarray, categories: List[str], 
                       title: str, bins: int = 50, ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Create a density plot with color coding on the curve itself (no underfill)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        return_fig = True
    else:
        return_fig = False
    
    if len(values) == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        if return_fig:
            return fig
        return None
    
    # Use analytical counting for wide formats, regular histogram for narrow formats
    if format_obj.total_bits > 16:
        # Use analytical method for wide formats
        bin_centers, counts, bin_categories = get_analytical_histogram_data(format_obj, bins)
        # Convert counts to density (normalize by total count and bin width)
        total_count = sum(counts)
        bin_width = (bin_centers[1] - bin_centers[0]) if len(bin_centers) > 1 else 1.0
        hist = counts / (total_count * bin_width) if total_count > 0 else counts
    else:
        # Use regular histogram for narrow formats
        max_abs_val = max(abs(format_obj.min_value), abs(format_obj.max_value))
        
        # Create bins that are symmetric around zero
        if format_obj.min_value < 0 and format_obj.max_value > 0:
            # For formats spanning both positive and negative
            bin_edges = np.linspace(-max_abs_val, max_abs_val, bins + 1)
        else:
            # For formats with only positive or only negative values
            bin_edges = np.linspace(format_obj.min_value, format_obj.max_value, bins + 1)
        
        # Calculate density using histogram
        hist, bin_edges = np.histogram(values, bins=bin_edges, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_categories = categories
    
    # Create color-coded density curve
    colors = plt.cm.viridis(np.linspace(0, 1, len(hist)))
    
    # Plot the curve with color coding
    for i in range(len(hist) - 1):
        ax.plot([bin_centers[i], bin_centers[i+1]], [hist[i], hist[i+1]], 
                color=colors[i], linewidth=3, alpha=0.8)
    
    # Add category markers
    for cat in set(bin_categories):
        cat_values = [v for v, c in zip(bin_centers, bin_categories) if c == cat]
        if cat_values:
            ax.scatter(cat_values, [0]*len(cat_values), 
                      c='red' if cat == 'zero' else 'green' if cat == 'subnormal' else 'blue',
                      marker='|', s=100, linewidth=2, label=f'{cat} values')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add format statistics
    stats_text = f'Min: {format_obj.min_value:.3e}\n'
    stats_text += f'Max: {format_obj.max_value:.3e}\n'
    if format_obj.total_bits > 16:
        stats_text += f'Total values: {sum(counts)} (analytical)'
    else:
        stats_text += f'Values: {len(values)}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    if return_fig:
        return fig
    return None

def create_shared_plot(format_obj1, values1: np.ndarray, categories1: List[str],
                       format_obj2: Optional[object] = None, values2: Optional[np.ndarray] = None, 
                       categories2: Optional[List[str]] = None,
                       plot_type: str = "Scatter Plot", bins: int = 50, 
                       color_scheme: str = "default") -> plt.Figure:
    """Create a shared plot that can show single format or comparison."""
    if plot_type == "Scatter Plot":
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot primary format
        create_scatter_plot(format_obj1, values1, categories1, 
                           f"{format_obj1.name} - Scatter Plot", color_scheme, ax)
        
        # Plot secondary format if available
        if format_obj2 is not None and values2 is not None and categories2 is not None:
            # Shift secondary format up slightly for visibility
            for cat in set(categories2):
                mask = [c == cat for c in categories2]
                if any(mask):
                    ax.scatter(values2[mask], [0.3]*sum(mask), 
                              c='red', alpha=0.7, s=30, label=f"{format_obj2.name} - {cat}")
            
            ax.set_ylim(-0.5, 0.8)
            ax.set_yticks([0, 0.3])
            ax.set_yticklabels([format_obj1.name, format_obj2.name])
            ax.set_title(f"Format Comparison: {format_obj1.name} vs {format_obj2.name}", fontsize=14, fontweight='bold', y=1.02)
        else:
            ax.set_title(f"{format_obj1.name} - Scatter Plot", fontsize=14, fontweight='bold')
    
    elif plot_type == "Histogram Plot":
        if format_obj2 is not None and values2 is not None and categories2 is not None:
            # Comparison mode - create stacked subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Primary format histogram
            create_histogram_plot(format_obj1, values1, categories1, 
                                 f"{format_obj1.name} - Histogram Plot", bins, ax1)
            
            # Secondary format histogram
            create_histogram_plot(format_obj2, values2, categories2, 
                                 f"{format_obj2.name} - Histogram Plot", bins, ax2)
            
            fig.suptitle(f"Format Comparison: {format_obj1.name} vs {format_obj2.name}", 
                        fontsize=16, fontweight='bold', y=1.02)
        else:
            # Single format mode
            fig, ax = plt.subplots(figsize=(12, 6))
            create_histogram_plot(format_obj1, values1, categories1, 
                                 f"{format_obj1.name} - Histogram Plot", bins, ax)
        
        plt.tight_layout()
    
    elif plot_type == "Density Plot":
        if format_obj2 is not None and values2 is not None and categories2 is not None:
            # Comparison mode - create stacked subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Primary format density
            create_density_plot(format_obj1, values1, categories1, 
                               f"{format_obj1.name} - Density Plot", bins, ax1)
            
            # Secondary format density
            create_density_plot(format_obj2, values2, categories2, 
                               f"{format_obj2.name} - Density Plot", bins, ax2)
            
            fig.suptitle(f"Format Comparison: {format_obj1.name} vs {format_obj2.name}", 
                        fontsize=16, fontweight='bold', y=1.02)
        else:
            # Single format mode
            fig, ax = plt.subplots(figsize=(12, 6))
            create_density_plot(format_obj1, values1, categories1, 
                               f"{format_obj1.name} - Density Plot", bins, ax)
        
        plt.tight_layout()
    
    else:  # All plots
        if format_obj2 is not None and values2 is not None and categories2 is not None:
            # Comparison mode - create stacked subplots for all types
            fig, axes = plt.subplots(6, 1, figsize=(12, 20))
            
            # Scatter plots (stacked)
            create_scatter_plot(format_obj1, values1, categories1, 
                               f"{format_obj1.name} - Scatter Plot", color_scheme, axes[0])
            create_scatter_plot(format_obj2, values2, categories2, 
                               f"{format_obj2.name} - Scatter Plot", color_scheme, axes[1])
            
            # Histogram plots (stacked)
            create_histogram_plot(format_obj1, values1, categories1, 
                                 f"{format_obj1.name} - Histogram Plot", bins, axes[2])
            create_histogram_plot(format_obj2, values2, categories2, 
                                 f"{format_obj2.name} - Histogram Plot", bins, axes[3])
            
            # Density plots (stacked)
            create_density_plot(format_obj1, values1, categories1, 
                               f"{format_obj1.name} - Density Plot", bins, axes[4])
            create_density_plot(format_obj2, values2, categories2, 
                               f"{format_obj2.name} - Density Plot", bins, axes[5])
            
            fig.suptitle(f"Format Comparison: {format_obj1.name} vs {format_obj2.name}", 
                        fontsize=16, fontweight='bold', y=1.02)
        else:
            # Single format mode - create subplots for all three types
            fig, axes = plt.subplots(3, 1, figsize=(12, 15))
            
            # Scatter plot
            create_scatter_plot(format_obj1, values1, categories1, 
                               f"{format_obj1.name} - Scatter Plot", color_scheme, axes[0])
            
            # Histogram plot
            create_histogram_plot(format_obj1, values1, categories1, 
                                 f"{format_obj1.name} - Histogram Plot", bins, axes[1])
            
            # Density plot
            create_density_plot(format_obj1, values1, categories1, 
                               f"{format_obj1.name} - Density Plot", bins, axes[2])
        
        plt.tight_layout()
    
    return fig

def get_format_summary(format_obj) -> str:
    """Generate a comprehensive summary of the numeric format in table style."""
    try:
        summary = f"**{format_obj.name}** ({format_obj.total_bits}-bit)\n\n"
        
        if hasattr(format_obj, 'signed'):
            # Integer format table
            summary += "| Property | Value |\n"
            summary += "|----------|-------|\n"
            summary += f"| **Type** | {'Signed' if format_obj.signed else 'Unsigned'} Integer |\n"
            summary += f"| **Range** | [{format_obj.min_value:,}, {format_obj.max_value:,}] |\n"
            summary += f"| **Total Codes** | {format_obj.total_codes:,} |\n"
        elif isinstance(format_obj, HIF8Format):
            # HiFloat8 format table
            summary += "| Property | Value |\n"
            summary += "|----------|-------|\n"
            summary += "| **Type** | HiFloat8 (Non-IEEE) |\n"
            summary += "| **Sign Bit** | 1 |\n"
            summary += "| **Dot Field** | Variable-length prefix codes |\n"
            summary += "| **Exponent** | Sign-magnitude with implicit '1' |\n"
            summary += "| **Mantissa** | Variable width (1-3 bits) |\n"
            summary += f"| **Range** | [{format_obj.min_value:.3e}, {format_obj.max_value:.3e}] |\n"
            summary += f"| **Total Codes** | {format_obj.total_codes:,} |\n"
            summary += "| **Special Values** | Zero (0x00), NaN (0x80), ±Inf (0x6F, 0xEF) |\n"
            summary += "| **Denormal** | Special encoding (Dot='0000') |\n"
        else:
            # Standard floating point format table
            summary += "| Property | Value |\n"
            summary += "|----------|-------|\n"
            summary += "| **Type** | Floating Point (IEEE754) |\n"
            summary += f"| **Exponent Bits** | {format_obj.exponent_bits} |\n"
            summary += f"| **Mantissa Bits** | {format_obj.mantissa_bits} |\n"
            summary += f"| **Bias** | {format_obj.bias} |\n"
            summary += f"| **Range** | [{format_obj.min_value:.3e}, {format_obj.max_value:.3e}] |\n"
            
            if hasattr(format_obj, 'get_min_normal'):
                summary += f"| **Min Normal** | {format_obj.get_min_normal():.3e} |\n"
                summary += f"| **Max Normal** | {format_obj.get_max_normal():.3e} |\n"
                summary += f"| **Min Subnormal** | {format_obj.get_min_subnormal():.3e} |\n"
                summary += f"| **Max Subnormal** | {format_obj.get_max_subnormal():.3e} |\n"
            
            if hasattr(format_obj, 'machine_epsilon'):
                summary += f"| **Machine Epsilon** | {format_obj.machine_epsilon:.3e} |\n"
                summary += f"| **ULP at 1.0** | {format_obj.ulp_at_1:.3e} |\n"
        
        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def validate_numeric_input(value: str, min_val: float, max_val: float) -> Tuple[bool, str, Optional[float]]:
    """Validate numeric input and return (is_valid, error_message, parsed_value)."""
    if not value.strip():
        return False, "Value cannot be empty", None
    
    try:
        parsed = float(value)
        if parsed < min_val or parsed > max_val:
            return False, f"Value must be between {min_val:.3e} and {max_val:.3e}", None
        return True, "", parsed
    except ValueError:
        return False, "Please enter a valid number", None

def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">🔢 Numeric Format Visualizer</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Format selection
    formats = create_numeric_formats()
    format_names = list(formats.keys()) + ["Custom Format"]
    
    # Primary format selection
    selected_format = st.sidebar.selectbox(
        "Select Primary Format:",
        format_names,
        index=format_names.index('FP8_E4M3 (IEEE754)')
    )
    
    # Primary custom format creation
    primary_custom_format = None
    if selected_format == "Custom Format":
        st.sidebar.markdown('<div class="custom-format-section">', unsafe_allow_html=True)
        st.sidebar.subheader("Primary Custom Format")
        
        custom_type1 = st.sidebar.selectbox("Format Type:", ["Integer", "Floating Point", "HiFloat8"], key="primary_type")
        
        if custom_type1 == "Integer":
            custom_bits1 = st.sidebar.number_input("Bits:", min_value=4, max_value=64, value=8, key="primary_bits")
            custom_signed1 = st.sidebar.checkbox("Signed", value=True, key="primary_signed")
            
            if st.sidebar.button("Create Primary Format", key="create_primary"):
                primary_custom_format = create_custom_format("Integer", bits=custom_bits1, signed=custom_signed1)
                if primary_custom_format:
                    st.sidebar.success(f"Created {primary_custom_format.name}")
        elif custom_type1 == "Floating Point":
            custom_exp_bits1 = st.sidebar.number_input("Exponent Bits:", min_value=2, max_value=15, value=4, key="primary_exp")
            custom_mantissa_bits1 = st.sidebar.number_input("Mantissa Bits:", min_value=1, max_value=52, value=3, key="primary_mantissa")
            
            if st.sidebar.button("Create Primary Format", key="create_primary"):
                primary_custom_format = create_custom_format("Floating Point", exp_bits=custom_exp_bits1, mantissa_bits=custom_mantissa_bits1)
                if primary_custom_format:
                    st.sidebar.success(f"Created {primary_custom_format.name}")
        elif custom_type1 == "HiFloat8":
            st.sidebar.info("HiFloat8 is a fixed 8-bit format with variable-length prefix codes. No customization needed.")
            
            if st.sidebar.button("Create Primary Format", key="create_primary"):
                primary_custom_format = create_custom_format("HiFloat8")
                if primary_custom_format:
                    st.sidebar.success(f"Created {primary_custom_format.name}")
        
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Secondary format selection (enabled only when comparison mode is active)
    comparison_mode = st.sidebar.checkbox("Enable Comparison Mode", value=False)
    
    if comparison_mode:
        selected_format2 = st.sidebar.selectbox(
            "Select Secondary Format:",
            format_names,
            index=format_names.index('FP8_E5M2 (IEEE754)'),
            key="secondary_format"
        )
        
        # Secondary custom format creation
        secondary_custom_format = None
        if selected_format2 == "Custom Format":
            st.sidebar.markdown('<div class="custom-format-section">', unsafe_allow_html=True)
            st.sidebar.subheader("Secondary Custom Format")
            
            custom_type2 = st.sidebar.selectbox("Format Type:", ["Integer", "Floating Point", "HiFloat8"], key="secondary_type")
            
            if custom_type2 == "Integer":
                custom_bits2 = st.sidebar.number_input("Bits:", min_value=4, max_value=64, value=8, key="secondary_bits")
                custom_signed2 = st.sidebar.checkbox("Signed", value=True, key="secondary_signed")
                
                if st.sidebar.button("Create Secondary Format", key="create_secondary"):
                    secondary_custom_format = create_custom_format("Integer", bits=custom_bits2, signed=custom_signed2)
                    if secondary_custom_format:
                        st.sidebar.success(f"Created {secondary_custom_format.name}")
            elif custom_type2 == "Floating Point":
                custom_exp_bits2 = st.sidebar.number_input("Exponent Bits:", min_value=2, max_value=15, value=4, key="secondary_exp")
                custom_mantissa_bits2 = st.sidebar.number_input("Mantissa Bits:", min_value=1, max_value=52, value=3, key="secondary_mantissa")
                
                if st.sidebar.button("Create Secondary Format", key="create_secondary"):
                    secondary_custom_format = create_custom_format("Floating Point", exp_bits=custom_exp_bits2, mantissa_bits=custom_mantissa_bits2)
                    if secondary_custom_format:
                        st.sidebar.success(f"Created {secondary_custom_format.name}")
            elif custom_type2 == "HiFloat8":
                st.sidebar.info("HiFloat8 is a fixed 8-bit format with variable-length prefix codes. No customization needed.")
                
                if st.sidebar.button("Create Secondary Format", key="create_secondary"):
                    secondary_custom_format = create_custom_format("HiFloat8")
                    if secondary_custom_format:
                        st.sidebar.success(f"Created {secondary_custom_format.name}")
            
            st.sidebar.markdown('</div>', unsafe_allow_html=True)
    else:
        selected_format2 = None
        secondary_custom_format = None
    
    # Plot type selection
    plot_type = st.sidebar.selectbox(
        "Plot Type:",
        ["Scatter Plot", "Histogram Plot", "Density Plot", "All"],
        index=3
    )
    
    # Color scheme
    color_scheme = st.sidebar.selectbox(
        "Color Scheme:",
        ["default", "viridis", "rainbow"],
        index=0
    )
    
    # Customization parameters using edit boxes
    st.sidebar.subheader("Customization")
    
    # Bins parameter
    bins_input = st.sidebar.text_input(
        "Number of Bins:",
        value="100",
        help="Enter number of bins for histogram and density plots"
    )
    
    # Validate bins input
    try:
        bins = int(bins_input)
    except ValueError:
        st.sidebar.error("Please enter a valid number for bins")
        bins = 100
    
    # Max values parameter
    max_values_input = st.sidebar.text_input(
        "Max Values to Display:",
        value="65535",
        help="Enter maximum number of values to display (1000-100000)"
    )
    
    # Validate max values input
    try:
        max_values = int(max_values_input)
    except ValueError:
        st.sidebar.error("Please enter a valid number for max values")
        max_values = 10000
    
    # Main content area - Shared plot area
    st.header("📊 Visualization")
    
    # Get primary format object and values
    if selected_format == "Custom Format" and primary_custom_format is not None:
        format_obj1 = primary_custom_format
    elif selected_format == "Custom Format":
        st.warning("Please create a custom format first.")
        format_obj1 = None
    else:
        format_obj1 = formats[selected_format]
    
    if format_obj1 is not None:
        values1, categories1 = get_format_values_optimized(format_obj1, max_values)
        
        # Get secondary format object and values if in comparison mode
        format_obj2 = None
        values2 = None
        categories2 = None
        
        if comparison_mode and selected_format2 is not None:
            if selected_format2 == "Custom Format" and secondary_custom_format is not None:
                format_obj2 = secondary_custom_format
            elif selected_format2 == "Custom Format":
                st.warning("Please create a secondary custom format first.")
            else:
                format_obj2 = formats[selected_format2]
            
            if format_obj2 is not None:
                values2, categories2 = get_format_values_optimized(format_obj2, max_values)
        
        # Create shared plot
        if len(values1) > 0:
            fig = create_shared_plot(format_obj1, values1, categories1,
                                   format_obj2, values2, categories2,
                                   plot_type, bins, color_scheme)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("No data available for the primary format.")
    
    # Format summaries
    if comparison_mode and format_obj1 is not None and format_obj2 is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("📋 Primary Format Summary")
            summary1 = get_format_summary(format_obj1)
            st.markdown(summary1)
        
        with col2:
            st.header("📋 Secondary Format Summary")
            summary2 = get_format_summary(format_obj2)
            st.markdown(summary2)
    
    elif format_obj1 is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📋 Format Summary")
            summary = get_format_summary(format_obj1)
            st.markdown(summary)
        
        with col2:
            # Empty column for better layout balance
            pass
    
    # Footer
    st.markdown("---")
    st.markdown("**Numeric Format Visualizer** - Explore the precision and distribution of various numeric formats!")

if __name__ == "__main__":
    main()
