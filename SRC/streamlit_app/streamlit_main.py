import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Streamlit Demo App", layout="centered")

# Title and description
st.title("📊 Streamlit Demo App")
st.markdown("""
This demo app showcases a few of Streamlit's core features:
- Widgets like sliders and checkboxes
- Charts and plots (Matplotlib & Plotly)
- Interactive data display
""")

# Sidebar controls
st.sidebar.header("Controls")
num_points = st.sidebar.slider("Number of Data Points", 10, 1000, 100)
noise_level = st.sidebar.slider("Noise Level", 0.0, 2.0, 0.5)
show_table = st.sidebar.checkbox("Show Data Table", value=True)

# Generate data
x = np.linspace(0, 10, num_points)
y = np.sin(x) + np.random.normal(0, noise_level, size=num_points)
data = pd.DataFrame({"x": x, "y": y})

# Show dataframe
if show_table:
    st.subheader("Generated Data")
    st.dataframe(data)

# Matplotlib plot
st.subheader("Matplotlib Plot")
fig, ax = plt.subplots()
ax.plot(x, y, label="Noisy Sine Wave")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Matplotlib Line Plot")
ax.legend()
st.pyplot(fig)

# Plotly plot
st.subheader("Plotly Plot")
fig2 = px.scatter(data, x="x", y="y", title="Interactive Plotly Scatter Plot", labels={"x": "X", "y": "Y"})
st.plotly_chart(fig2, use_container_width=True)

# Checkbox and button
if st.checkbox("Show summary statistics"):
    st.write(data.describe())

if st.button("Re-run App"):
    st.experimental_rerun()
