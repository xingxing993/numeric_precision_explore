# 🔢 Numeric Format Visualizer

A comprehensive web-based visualization tool for exploring numeric format distributions, precision characteristics, and format comparisons. Built with Streamlit and Python.

## ✨ Features

### 📊 Visualization Styles
- **Scatter Plot**: Data points plotted on a single x-axis with color-coded categories
- **Density Plot**: Histogram-based density curves with color-coded underfill
- **Comparison View**: Side-by-side comparison of any two numeric formats

### 🔧 Supported Formats
- **Integer Formats**: INT8, UINT8, INT16, UINT16, INT32, UINT32
- **Standard Floating Point**: FP8_E4M3, FP8_E5M2, FP16, BF16, FP32, FP64
- **Special FP8 Formats**: FP8_E4M3FN, FP8_E4M3FNUZ

### 🎛️ Interactive Controls
- Format selection dropdown
- Plot style switching (Scatter, Density, or Both)
- Color scheme customization
- Bin count adjustment for density plots
- Maximum value limit controls for large formats

### 🔍 Analysis Tools
- **Format Summary**: Comprehensive technical details for each format
- **Value Statistics**: Category breakdown and counts
- **Format Comparison**: Visual comparison between any two formats
- **Custom Format Creator**: Build and explore custom numeric formats

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run visualize.py
```

### 3. Open Your Browser
The application will automatically open at `http://localhost:8501`

## 📱 Usage Guide

### Main Visualization
1. **Select Format**: Choose from the sidebar dropdown
2. **Choose Plot Style**: Scatter, Density, or Both
3. **Customize Appearance**: Adjust colors, bins, and display limits
4. **View Summary**: See detailed format information on the right

### Format Comparison
1. **Select Two Formats**: Use the comparison section dropdowns
2. **View Side-by-Side**: See scatter and density comparisons
3. **Analyze Differences**: Identify precision and range variations

### Custom Format Creation
1. **Choose Type**: Integer or Floating Point
2. **Set Parameters**: Adjust bits, signed/unsigned, exponent/mantissa
3. **Create Format**: Generate and explore custom numeric formats

## 🏗️ Architecture

- **Frontend**: Streamlit web interface
- **Backend**: Python with matplotlib/seaborn for plotting
- **Data**: Custom numeric format classes from `numeric_format.py`
- **Visualization**: Interactive plots with real-time updates

## 🔧 Customization

### Adding New Formats
Extend the `create_numeric_formats()` function in `visualize.py`:

```python
formats.update({
    'NEW_FORMAT': NewFormatClass('NEW_FORMAT', params...)
})
```

### Modifying Plot Styles
Edit the plotting functions:
- `create_scatter_plot()`: Customize scatter plot appearance
- `create_density_plot()`: Modify density plot behavior
- `create_comparison_plot()`: Change comparison visualization

### Styling
Modify the CSS in the `st.markdown()` section for custom appearance.

## 📊 Technical Details

### Performance Considerations
- Large formats (>16 bits) are sampled to prevent memory issues
- Configurable `max_values` parameter controls display limits
- Efficient value enumeration with early termination

### Memory Management
- Matplotlib figures are properly closed after display
- NumPy arrays are used for efficient data handling
- Streamlit's caching can be added for repeated computations

## 🌐 Deployment

### Local Development
```bash
streamlit run visualize.py --server.port 8501
```

### Production Deployment
1. **Streamlit Cloud**: Free hosting for Streamlit apps
2. **Docker**: Containerize the application
3. **Cloud Platforms**: Deploy on AWS, GCP, or Azure

### Docker Example
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "visualize.py", "--server.port=8501"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your enhancements
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

For issues or questions:
1. Check the existing documentation
2. Review the code comments
3. Open an issue on GitHub

---

**Happy Exploring! 🎉**

Discover the fascinating world of numeric precision and format distributions with this interactive visualization tool.
