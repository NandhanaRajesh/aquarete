import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm  # For colormap

class ParticleAnalyzer:
    def __init__(self, image_size=(1024, 1024)):
        self.image_size = image_size

    def detect_and_analyze_particles(self, image_path):
        """Detect and analyze particles in the holographic image."""
        output_path = 'detected_particles.png'
        img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        _, thresh_img = cv2.threshold(img_cv, 50, 255, cv2.THRESH_BINARY)
        blurred = cv2.GaussianBlur(thresh_img, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_contour_area = 20
        filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

        img_contours = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_contours, filtered_contours, -1, (0, 255, 0), 1)

        cv2.imwrite(output_path, img_contours)

        # Calculate areas of detected particles
        areas = [cv2.contourArea(c) for c in filtered_contours]

        # Exclude the largest particle
        if areas:
            largest_area = max(areas)
            areas = [a for a in areas if a != largest_area]
            filtered_contours = [c for c in filtered_contours if cv2.contourArea(c) != largest_area]

        # Display the contours using Matplotlib
        plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
        plt.title('Detected Particles')
        plt.axis('off')
        plt.show()

        num_detected_particles = len(filtered_contours)
        surface_area_cm2 = (self.image_size[0] / 100.0) * (self.image_size[1] / 100.0)  # Convert pixels to cm²
        density_particles_per_cm2 = num_detected_particles / surface_area_cm2

        # Display results
        print(f"Number of detected particles: {num_detected_particles}")
        print(f"Density of particles: {density_particles_per_cm2:.2f} particles/cm²")

        return areas, density_particles_per_cm2

    def visualize_particle_distribution(self, areas):
        """Visualize particle size distribution with markers for different particle types."""
        particle_types = ['microplastic', 'algae', 'other']
        particle_size = {
            'microplastic': (2, 5),
            'algae': (15, 30),
            'other': (5, 20)
        }

        type_labels = []
        for area in areas:
            if area <= particle_size['microplastic'][1] ** 2:
                type_labels.append('microplastic')
            elif area <= particle_size['algae'][1] ** 2:
                type_labels.append('algae')
            else:
                type_labels.append('other')

        type_counts = {typ: type_labels.count(typ) for typ in particle_types}
        print("Particle type counts:", type_counts)

        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))

        for typ in particle_types:
            typ_areas = [a for a, l in zip(areas, type_labels) if l == typ]
            plt.hist(typ_areas, bins=20, alpha=0.5, label=f'{typ} particles')

        plt.title('Size Distribution of Detected Particles')
        plt.xlabel('Area (pixels)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    def visualize_3d_particle_distribution(self, areas):
        """Visualize particle size distribution in 3D with a colorful display."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        x = np.arange(len(areas))
        y = np.zeros(len(areas))  # All bars start from the base (y=0)
        z = np.zeros(len(areas))  # All bars start from the base (z=0)
        dx = np.ones_like(x)  # Bar width
        dy = np.ones_like(y)  # Bar depth
        dz = areas  # Bar height

        # Use a colormap for colors
        colors = cm.viridis(np.linspace(0, 1, len(areas)))

        # Create the 3D bars with colormap
        ax.bar3d(x, y, z, dx, dy, dz, color=colors, alpha=0.8)

        ax.set_xlabel('Index')
        ax.set_ylabel('Depth')
        ax.set_zlabel('Area (pixels)')
        plt.title('3D Visualization of Particle Sizes')

        plt.show()

# Example usage for Part 2
image_path = input("Enter the path to the holographic image: ")
analyzer = ParticleAnalyzer(image_size=(1024, 1024))
areas, density_particles_per_cm2 = analyzer.detect_and_analyze_particles(image_path)
analyzer.visualize_particle_distribution(areas)
analyzer.visualize_3d_particle_distribution(areas)
