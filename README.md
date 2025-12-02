# Digital Twin-Based Channel Generation Toolchain for Low-Altitude XL-MIMO

**Official repository for the paper:**
> **Digital Twin-Based Channel Generation Toolchain and Foundation Model for Low-Altitude XL-MIMO**
Paper will be public at arxiv soon.
---

## 📂 Repository Structure & Methodology

This repository implements the three-stage generation pipeline described in our paper: **Static Environment Construction**, **UAV Trajectory Simulation**, and **XL-MIMO Channel Generation**.

| File Name | Methodological Description |
| :--- | :--- |
| `LAE_Urban_OSM_to_Sionna_scene.ipynb` | **3D Static Environment & Trajectory Construction (Urban):** <br> 1. **Scene:** Converts 2D OpenStreetMap (OSM) data into 3D scenarios, reconstructing meshes and **injecting ITU-compliant material properties** (e.g., concrete, metal). <br> 2. **Mobility:** Integrated generation of **UAV trajectories aligned with road networks**. It supports defining flight paths that follow street topology with **configurable altitude parameters** to simulate diverse low-altitude flight levels. |
| `LAE_Suburban_OSM_to_Sionna_scene.ipynb` | **3D Static Environment & Trajectory Construction (Suburban):** <br> Similar to the urban toolchain but optimized for suburban topologies. It features the generation of **long-distance linear trajectories** along the horizontal direction (cruising mode) with **adjustable vertical altitude** settings, specifically designed to simulate high-speed UAV operations in open suburban areas. |
| `LAE_data_batch_generation.py` | **XL-MIMO Channel Simulation (Ray Tracing):** <br> Loads the generated 3D scenes and trajectories. It executes **Sionna RT** to perform **per-antenna channel generation**, setting `synthetic_array=False` to strictly capture **spherical wave characteristics** in the near-field region ($r < D_R$). |
| `Split_and_merge_single_scene_channels.py` | **Data Management Utility:** <br> Due to the massive scale of XL-MIMO channels (e.g., $64 \times 64$ antennas), this script manages memory by splitting large generation tasks and merging partial results for individual scenes. |
| `merge_all_scene_channels.py` | **Final Dataset Aggregation:** <br> The final step in the pipeline. It aggregates channel data, UAV positions, and labels from all generated dynamic scenes into a unified, comprehensive dataset structure, ready for training the Foundation Model. |
---

## 🚀 Key Features

* **From 2D Maps to 3D Digital Twins**: Automated pipeline to fetch OSM data and semantic descriptions, converting them into renderable 3D assets.
* **Material-Aware Simulation**: Assigns electromagnetic properties (permittivity, conductivity) to scene objects based on ITU recommendations.
* **Dynamic 3D Mobility**: Supports traffic-constrained UAV trajectories (mimicking road networks) and random cruising patterns.
* **Near-Field High-Fidelity**: Generates channels using per-element ray tracing to preserve spherical wavefront curvature, essential for XL-MIMO analysis.

---

## 🛠️ Usage Guide

### Prerequisites
* Python == 3.10.18
* **NVIDIA Sionna**
* sionna-rt 1.1.0
* Mitsuba 3
* `osmnx`, `geopandas`
* SUMO
  

### Workflow

**1. Construct the Digital Twin Scene**
Open `LAE_Urban_OSM_to_Sionna_scene.ipynb`. Configure the target ROI coordinates. Run the notebook to download map data and export the scene file (e.g., `.xml` or `.obj`).

**2. Run Batch Generation**
Once the scene file is ready, configure the simulation parameters (number of UAVs, time slots, antenna array size) in `LAE_data_batch_generation.py` and run:
python LAE_data_batch_generation.py

**3. Merge data**
run:
python Split_and_merge_single_scene_channels.py
python merge_all_scene_channels.py


### Our dataset is released at:
https://drive.google.com/drive/folders/1PnjIwXSudX33-_4c_0Es1WZ10MXFgw82?usp=drive_link
