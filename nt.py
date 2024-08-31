import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm  # For colormap


class ParticleDetectorApp:
    def __init__(self, num_particles=500, image_size=(512, 512)):
        self.num_particles = num_particles
        self.image_size = image_size  # Updated image size

    def generate_particles_image(self):
        """Generate an image with simulated particles including small, large particles, living organisms, and other types."""
        img = Image.new('L', self.image_size, color=0)
        draw = ImageDraw.Draw(img)

        # Define particle sizes and types with distinct colors
        particle_types = ['small', 'medium', 'large', 'microbe', 'algae', 'other']
        particle_size = {
            'small': 2,
            'medium': 5,
            'large': 10,
            'microbe': 15,
            'algae': 20,
            'other': 25
        }
        colors = {
            'small': 100,
            'medium': 150,
            'large': 200,
            'microbe': 255,  # For contrast
            'algae': 125,
            'other': 50
        }

        for _ in range(self.num_particles):
            x, y = np.random.randint(0, self.image_size[0], 2)
            particle_type = np.random.choice(particle_types)
            size = particle_size[particle_type]
            color = colors[particle_type]
            draw.ellipse((x, y, x + size, y + size), fill=color)

        img.save('simulated_particles.png')
        img.show()

    def create_holographic_effect(self, image_path, output_path):
        """Create a holographic effect with interference patterns."""
        img = Image.open(image_path).convert('L')
        img_np = np.array(img)

        # Add interference patterns
        height, width = img_np.shape
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)

        interference = (np.sin(X / 20.0) + np.cos(Y / 20.0)) * 0.5
        holographic_img_np = np.clip(img_np + 50 * interference, 0, 255).astype(np.uint8)
        holographic_img = Image.fromarray(holographic_img_np)
        holographic_img = holographic_img.filter(ImageFilter.GaussianBlur(2))

        holographic_img.save(output_path)
        plt.imshow(holographic_img, cmap='gray')
        plt.title('Holographic Effect with Interference Patterns')
        plt.axis('off')
        plt.show()

        return output_path

    def detect_and_analyze_particles(self, image_path, output_path):
        """Detect and analyze particles in the holographic image."""
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

        # Display the contours using Matplotlib
        plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
        plt.title('Detected Particles')
        plt.axis('off')
        plt.show()

        num_detected_particles = len(filtered_contours)
        surface_area_cm2 = (self.image_size[0] / 100.0) * (self.image_size[1] / 100.0)  # Convert pixels to cm²
        density_particles_per_cm2 = num_detected_particles / surface_area_cm2

        # Calculate areas of detected particles
        areas = [cv2.contourArea(c) for c in filtered_contours]

        # Display results
        print(f"Number of detected particles: {num_detected_particles}")
        print(f"Density of particles: {density_particles_per_cm2:.2f} particles/cm²")

        return areas, density_particles_per_cm2

    def visualize_particle_distribution(self, areas):
        """Visualize particle size distribution with markers for different particle types."""
        # Simulate particle types based on area size
        particle_types = ['small', 'medium', 'large', 'microbe', 'algae', 'other']
        particle_size = {
            'small': 2,
            'medium': 5,
            'large': 10,
            'microbe': 15,
            'algae': 20,
            'other': 25
        }

        type_labels = []
        for area in areas:
            if area <= particle_size['small'] ** 2:
                type_labels.append('small')
            elif area <= particle_size['medium'] ** 2:
                type_labels.append('medium')
            elif area <= particle_size['large'] ** 2:
                type_labels.append('large')
            elif area <= particle_size['microbe'] ** 2:
                type_labels.append('microbe')
            elif area <= particle_size['algae'] ** 2:
                type_labels.append('algae')
            else:
                type_labels.append('other')

        # Convert labels to categorical data for plotting
        type_counts = {typ: type_labels.count(typ) for typ in particle_types}
        print("Particle type counts:", type_counts)

        # Plot size distribution with different colors and markers
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


# Example usage
app = ParticleDetectorApp(num_particles=500, image_size=(1024, 1024))  # Increased image size

# Generate a high number of particles for testing
app.generate_particles_image()

# Create and analyze holographic effect
holographic_image_path = app.create_holographic_effect('simulated_particles.png',
                                                       'holographic_particles_with_interference.png')
areas, density_particles_per_cm2 = app.detect_and_analyze_particles(holographic_image_path, 'detected_particles.png')

# Visualize size distribution
app.visualize_particle_distribution(areas)

# Visualize size distribution in 3D
app.visualize_3d_particle_distribution(areas)
