import os
import numpy as np
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.image as mpimg
from matplotlib.colors import LightSource
from PIL import Image
import ephem
import json
from threading import Thread
from queue import Queue
import warnings
import logging
from matplotlib.widgets import Button

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

class ConfigManager:
    """Manage configuration settings for the visualizer"""
    DEFAULT_CONFIG = {
        'update_interval': 1000,  # ms
        'trail_length': 50,
        'texture_size': 2048,
        'texture_quality': 50,         # texture quality for Earth
        'show_labels': True,
        'earth_alpha': 0.9,
        'position_cache_time': 24,  # hours
        'default_view_mode': 'full'
    }

    def __init__(self, config_file='orbit_config.json'):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        """Load configuration from file or create default"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    return {**self.DEFAULT_CONFIG, **loaded_config}
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
        return self.DEFAULT_CONFIG.copy()

    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            logger.error(f"Could not save config: {e}")

    def get(self, key):
        """Get configuration value"""
        return self.config.get(key, self.DEFAULT_CONFIG.get(key))

class TLEManager:
    """Manage TLE data for satellites"""
    def __init__(self):
        self.tle_data = {
            "ISS": [
                "1 25544U 98067A   22325.59167824  .00016717  00000+0  30729-3 0  9993",
                "2 25544  51.6431 226.7758 0005941 113.7952 246.3823 15.50087640370124"
            ],
            "Hubble": [
                "1 20580U 90037B   22326.14821306  .00001264  00000+0  60660-4 0  9991",
                "2 20580  28.4692 341.3025 0002725  99.7240 260.4518 15.09257836345634"
            ],
            "Landsat 8": [
                "1 39084U 13008A   22326.29217208  .00000068  00000+0  18063-4 0  9991",
                "2 39084  98.2083 322.0110 0001335  90.0123 270.0100 14.57190296322473"
            ],
            "StarLink-32637": [
                "1 62032U 24216A   24355.91667824  .00004160  00000+0  10686-3 0  9994",
                "2 62032  42.9978 264.4632 0001924 273.7191  58.1736 15.40684592  5668"
            ]
        }
        self.satellites = {}
        self._initialize_satellites()

    def _initialize_satellites(self):
        """Initialize satellite objects from TLE data"""
        for name, tle in self.tle_data.items():
            try:
                self.satellites[name] = twoline2rv(tle[0], tle[1], wgs84)
            except Exception as e:
                logger.error(f"Error initializing satellite {name}: {e}")

    def get_satellite(self, name):
        """Get satellite object by name"""
        return self.satellites.get(name)

class OrbitCalculator:
    """Handle orbital calculations and caching"""
    def __init__(self, config_manager):
        self.config = config_manager
        self._position_cache = {}
        self._cache_time = {}

    def calculate_positions(self, satellite, start_time, duration_hours=24, step_minutes=1):
        """Calculate satellite positions with caching"""
        cache_key = (satellite.satnum, start_time, duration_hours, step_minutes)

        if cache_key in self._position_cache:
            cache_time = self._cache_time[cache_key]
            if datetime.now() - cache_time < timedelta(hours=1):
                return self._position_cache[cache_key]

        positions = []
        times = []
        lat_long_alt = []

        for minutes in range(0, duration_hours * 60, step_minutes):
            current_time = start_time + timedelta(minutes=minutes)
            try:
                position, velocity = satellite.propagate(
                    current_time.year, current_time.month, current_time.day,
                    current_time.hour, current_time.minute, current_time.second)
                
                # Calculate latitude, longitude, and altitude
                r = np.sqrt(position[0]**2 + position[1]**2 + position[2]**2)
                lat = np.arcsin(position[2]/r)
                lon = np.arctan2(position[1], position[0])
                alt = r - 6371  # Earth radius subtracted to get altitude

                positions.append(position)
                times.append(current_time)
                lat_long_alt.append((np.degrees(lat), np.degrees(lon), alt))
            except Exception as e:
                logger.error(f"Error calculating position: {e}")

        result = (np.array(positions), times, lat_long_alt)
        self._cache_position(cache_key, result)
        return result

    def _cache_position(self, cache_key, result):
        """Cache calculated positions"""
        self._position_cache[cache_key] = result
        self._cache_time[cache_key] = datetime.now()

        current_time = datetime.now()
        old_keys = [k for k, t in self._cache_time.items() 
                   if (current_time - t).total_seconds() > 3600]
        for k in old_keys:
            del self._position_cache[k]
            del self._cache_time[k]

class OrbitVisualizer:
    """Main visualization class"""
    def __init__(self):
        self.config = ConfigManager()
        self.tle_manager = TLEManager()
        self.orbit_calculator = OrbitCalculator(self.config)

        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(15, 15), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.current_satellite = "ISS"
        self.current_mode = self.config.get('default_view_mode')
        self.show_moon = True

        # Create text elements for live information
        self.info_text = self.fig.text(0.75, 0.95, "", fontsize=10, color='white',
                                     fontfamily='monospace', verticalalignment='top')

        self.earth_texture = self.load_earth_texture()
        self.setup_ui()

        self.update_queue = Queue()
        self.update_thread = Thread(target=self._update_thread, daemon=True)
        self.update_thread.start()

    def load_earth_texture(self):
        """Load and process Earth texture"""
        try:
            img = Image.open("8081_earthmap4k.jpg")
            max_size = self.config.get('texture_size')
            if img.size[0] > max_size:
                img = img.resize((max_size, max_size//2), Image.LANCZOS)
            return np.array(img) / 255.0
        except Exception as e:
            logger.warning(f"Could not load Earth texture: {e}")
            return None

    def setup_ui(self):
        """Set up user interface elements"""
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.add_instructions()
        button_ax = self.fig.add_axes([0.02, 0.4, 0.1, 0.04])
        self.reset_button = Button(button_ax, 'Reset View', color='black', hovercolor='#5d5f61')
        self.reset_button.on_clicked(self.reset_view)

    def add_instructions(self):
        """Add instruction text to visualization"""
        instructions = """
Controls:
- Use mouse to rotate/zoom main view
- Press keys 1-4 to switch satellites:
  1: ISS (altitude: ~400km)
  2: Hubble (altitude: ~540km)
  3: Landsat 8 (altitude: ~705km)
  4: StarLink (altitude: ~550km)
- Press 'm' to toggle moon
- Press 'e' to export positions to CSV
"""
        self.fig.text(0.02, 0.98, instructions,
                     fontsize=10,
                     transform=self.fig.transFigure,
                     verticalalignment='top',
                     fontfamily='monospace',
                     color='white')

    def on_key(self, event):
        """Handle keyboard events"""
        if event.key in ['1', '2', '3', '4']:
            satellites = ['ISS', 'Hubble', 'Landsat 8', 'StarLink-32637']
            self.current_satellite = satellites[int(event.key) - 1]
            self.update_queue.put('update_plot')
        elif event.key == 'm':
            self.show_moon = not self.show_moon
            self.update_queue.put('update_plot')
        elif event.key == 'e':
            self.export_positions()

    def reset_view(self, event=None):
        """Reset view to default position"""
        self.ax.view_init(elev=20, azim=45)
        self.update_queue.put('update_plot')

    def _update_thread(self):
        """Background thread for updates"""
        while True:
            try:
                action = self.update_queue.get()
                if action == 'update_plot':
                    self._update_plot()
                self.update_queue.task_done()
            except Exception as e:
                logger.error(f"Error in update thread: {e}")

    def calculate_moon_position(self, time, scale_factor=1.0):
        """Calculate moon position"""
        moon = ephem.Moon()
        moon.compute(time)
        distance = moon.earth_distance * 149597870.691 * scale_factor
        ra = float(moon.ra)
        dec = float(moon.dec)
        x = distance * np.cos(dec) * np.cos(ra)
        y = distance * np.cos(dec) * np.sin(ra)
        z = distance * np.sin(dec)
        return np.array([x, y, z])

    def create_earth(self):
        """Create Earth representation"""
        r_earth = 6371
        phi = np.linspace(0, np.pi, self.config.get('texture_quality'))
        theta = np.linspace(0, 2 * np.pi, self.config.get('texture_quality'))
        phi, theta = np.meshgrid(phi, theta)

        x = r_earth * np.sin(phi) * np.cos(theta)
        y = r_earth * np.sin(phi) * np.sin(theta)
        z = r_earth * np.cos(phi)

        if self.earth_texture is not None:
            colors = self._apply_texture(phi, theta, z)
        else:
            colors = plt.cm.terrain((z + r_earth) / (2 * r_earth))

        self.ax.plot_surface(x, y, z,
                            facecolors=colors,
                            alpha=0.55,
                            rstride=1, cstride=1,
                            shade=True)

    def _apply_texture(self, phi, theta, z):
        """Apply texture mapping with improved interpolation"""
        height, width = self.earth_texture.shape[:2]
        u = (theta / (2 * np.pi))
        v = (phi / np.pi)

        u_float = u * (width - 1)
        v_float = v * (height - 1)

        u1 = np.floor(u_float).astype(int)
        v1 = np.floor(v_float).astype(int)
        u2 = np.minimum(u1 + 1, width - 1)
        v2 = np.minimum(v1 + 1, height - 1)

        fu = u_float - u1
        fv = v_float - v1

        c1 = self.earth_texture[v1, u1] * (1 - fu)[..., np.newaxis] * (1 - fv)[..., np.newaxis]
        c2 = self.earth_texture[v1, u2] * fu[..., np.newaxis] * (1 - fv)[..., np.newaxis]
        c3 = self.earth_texture[v2, u1] * (1 - fu)[..., np.newaxis] * fv[..., np.newaxis]
        c4 = self.earth_texture[v2, u2] * fu[..., np.newaxis] * fv[..., np.newaxis]

        colors = c1 + c2 + c3 + c4

        light_source = LightSource(azdeg=315, altdeg=45)
        z_normalized = (z - z.min()) / (z.max() - z.min())
        illuminated = light_source.shade(z_normalized, cmap=plt.cm.gray)
        colors = colors * (0.7 + 0.3 * np.expand_dims(illuminated[..., 0], axis=2))

        return np.clip(colors, 0, 1)

    def _update_plot(self):
        """Update the visualization with all elements"""
        try:
            self.ax.clear()

            # Get satellite positions
            satellite = self.tle_manager.get_satellite(self.current_satellite)
            positions, times, lat_long_alt = self.orbit_calculator.calculate_positions(
                satellite, datetime.now())

            # Update live information
            current_time = datetime.now()
            current_lat, current_lon, current_alt = lat_long_alt[-1]
            info_text = f"Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            info_text += f"Latitude: {current_lat:.2f}°\n"
            info_text += f"Longitude: {current_lon:.2f}°\n"
            info_text += f"Altitude: {current_alt:.2f} km"
            self.info_text.set_text(info_text)

            # Calculate view limits
            view_limits = self._calculate_view_limits(positions)

            # Set plot limits
            self.ax.set_xlim(-view_limits, view_limits)
            self.ax.set_ylim(-view_limits, view_limits)
            self.ax.set_zlim(-view_limits, view_limits)

            # Create Earth
            self.create_earth()

            # Plot satellite orbit
            self.ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                        label=f'{self.current_satellite} Orbit',
                        color='deepskyblue',
                        linewidth=2)

            # Plot current satellite position
            self.ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
                          color='red', s=100, label='Current Position')

            # Handle moon visualization if enabled
            if self.show_moon:
                moon_pos = self.calculate_moon_position(times[-1])
                self._add_moon_visualization(moon_pos)

            # Set default view angle if needed
            if not plt.get_fignums():  # Only set if figure is not being interacted with
                self.ax.view_init(elev=20, azim=45)

            # Add title and labels
            self._add_plot_labels(positions)

            # Update display
            self.ax.legend(loc='upper right')
            self.ax.grid(True, alpha=0.2)
            self.ax.set_box_aspect([1, 1, 1])

            self.fig.canvas.draw_idle()

        except Exception as e:
            logger.error(f"Error updating plot: {e}")

    def _calculate_view_limits(self, positions):
        """Calculate view limits"""
        if self.show_moon:
            moon_pos = self.calculate_moon_position(datetime.now())
            return max(np.max(np.abs(positions)) * 1.2,
                      np.max(np.abs(moon_pos)) * 1.2)
        return np.max(np.abs(positions)) * 1.2

    def _add_moon_visualization(self, moon_pos):
        """Add moon visualization elements"""
        self.ax.scatter(*moon_pos, color='white', s=300,
                       label='Moon', alpha=0.8)
        self.ax.plot([0, moon_pos[0]], [0, moon_pos[1]], [0, moon_pos[2]],
                    'w--', alpha=0.2)

        moon_distance = np.linalg.norm(moon_pos)
        text_offset = moon_pos * 1.1
        self.ax.text(text_offset[0], text_offset[1], text_offset[2],
                    f'Moon Distance: {moon_distance:.3f}km',
                    color='white', alpha=0.7)

    def _add_plot_labels(self, positions):
        """Add labels and information to plot"""
        altitude = np.mean(np.sqrt(np.sum(positions**2, axis=1))) - 6371
        title = f'Orbital Path of {self.current_satellite}\n' \
                f'Average Altitude: {altitude:.1f} km'

        self.ax.set_title(title)
        self.ax.set_xlabel('X (km)')
        self.ax.set_ylabel('Y (km)')
        self.ax.set_zlabel('Z (km)')

    def export_positions(self):
        """Export satellite positions to CSV"""
        try:
            satellite = self.tle_manager.get_satellite(self.current_satellite)
            positions, times, lat_long_alt = self.orbit_calculator.calculate_positions(
                satellite, datetime.now())

            df = pd.DataFrame({
                'Time': times,
                'X (km)': positions[:, 0],
                'Y (km)': positions[:, 1],
                'Z (km)': positions[:, 2],
                'Latitude': [lla[0] for lla in lat_long_alt],
                'Longitude': [lla[1] for lla in lat_long_alt],
                'Altitude (km)': [lla[2] for lla in lat_long_alt]
            })

            filename = f'{self.current_satellite}_positions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            df.to_csv(filename, index=False)
            logger.info(f"Exported positions to {filename}")

        except Exception as e:
            logger.error(f"Error exporting positions: {e}")

def main():
    """Main function to run the visualizer"""
    try:
        visualizer = OrbitVisualizer()
        visualizer.update_queue.put('update_plot')
        plt.show()
    except Exception as e:
        logger.error(f"Error running visualizer: {e}")
        raise

if __name__ == "__main__":
    main()