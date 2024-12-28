# Orbital Path Visualizer

A Python-based 3D visualization tool for tracking and displaying satellite orbits, including the ISS, Hubble Space Telescope, Landsat 8, and StarLink satellites. The tool provides real-time position updates, interactive 3D visualization, and Earth texture mapping.

## Features

- Real-time 3D visualization of satellite orbits
- Interactive camera controls for rotation and zoom
- Earth texture mapping with realistic lighting
- Moon position visualization
- Position export to CSV
- Satellite switching on the fly

## Prerequisites

- Python 3.8+
- A virtual environment (recommended)

## Installation

1. Clone the repository and install required packages:
```bash
git clone https://github.com/MitanshuKumawat/VSSC_ISRO.git
cd VSSC_ISRO
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # On macOS, use: source .venv/bin/activate
```

3. Install required packages:
```bash
pip install numpy matplotlib pandas pillow ephem sgp4
```

## Usage

1. Run the visualizer:
```bash
python ./code/orbital_path_prediction.py
```

2. Controls:
   - Use mouse to rotate the view
   - Keys 1-4 switch between satellites:
     - 1: ISS
     - 2: Hubble
     - 3: Landsat 8
     - 4: StarLink
   - 'm' toggles moon visualization
   - 'e' exports current positions to CSV
   - Use the "Reset View" button to return to default view

## Configuration

The tool uses a JSON configuration file (`orbit_config.json`) for customizing:
- Update interval
- Trail length
- Texture resolution
- Label visibility
- Earth transparency
- Position cache duration

## Notes

- The visualization updates in real-time based on TLE data
- Position data is cached to improve performance
- Exported CSV files include time, position, and lat/long data

## Troubleshooting

1. If you want to improve the texture quality, try increasing the 'texture_quality' in the config file(DEFAULT_CONFIG).
