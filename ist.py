import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm  # For colormap

class ParticleImageGenerator:
    def __init__(self, image_size=(1024, 1024)):
        self.image_size = image_size

    def generate_particles_image(self, num_microplastics, num_nutrients, num_microorganisms, num_sediments):
        """Generate an image with user-defined numbers of particles."""
        img = Image.new('L', self.image_size, color=0)
        draw = ImageDraw.Draw(img)

        # Define particle types, sizes, and colors in order of increasing size
        particle_types = {
            'microplastic': {'size': (5, 10), 'count': num_microplastics, 'color': 100},  # Increased size
            'microorganism': {'size': (10, 25), 'count': num_microorganisms, 'color': 150},  # Increased size
            'nutrient': {'size': (20, 35), 'count': num_nutrients, 'color': 120},  # Increased size
            'sediment': {'size': (40, 60), 'count': num_sediments, 'color': 180}  # Increased size
        }

        for particle_type, properties in particle_types.items():
            for _ in range(properties['count']):
                x, y = np.random.randint(0, self.image_size[0], 2)
                size = np.random.randint(properties['size'][0], properties['size'][1])
                color = properties['color']
                draw.ellipse((x, y, x + size, y + size), fill=color)

        img.save('simulated_particles.png')
        img.show()

    def create_holographic_effect(self, image_path, output_path):
        """Create a holographic effect with different interference patterns for each particle type."""
        img = Image.open(image_path).convert('L')
        img_np = np.array(img)

        height, width = img_np.shape
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)

        # Define different interference patterns
        interference_patterns = {
            'microplastic': (np.sin(X / 20.0) + np.cos(Y / 20.0)) * 0.5,
            'microorganism': (np.sin(X / 30.0) + np.cos(Y / 30.0)) * 0.7,
            'nutrient': (np.sin(X / 25.0) + np.cos(Y / 25.0)) * 0.6,
            'sediment': (np.sin(X / 35.0) + np.cos(Y / 35.0)) * 0.8
        }

        # Apply the interference patterns to the image
        holographic_img_np = np.copy(img_np)
        for i in range(height):
            for j in range(width):
                pixel_value = img_np[i, j]
                if pixel_value == 100:  # Microplastic
                    holographic_img_np[i, j] = np.clip(pixel_value + 50 * interference_patterns['microplastic'][i, j], 0, 255)
                elif pixel_value == 150:  # Microorganism
                    holographic_img_np[i, j] = np.clip(pixel_value + 50 * interference_patterns['microorganism'][i, j], 0, 255)
                elif pixel_value == 120:  # Nutrient
                    holographic_img_np[i, j] = np.clip(pixel_value + 50 * interference_patterns['nutrient'][i, j], 0, 255)
                else:  # Sediment
                    holographic_img_np[i, j] = np.clip(pixel_value + 50 * interference_patterns['sediment'][i, j], 0, 255)

        holographic_img = Image.fromarray(holographic_img_np)
        holographic_img = holographic_img.filter(ImageFilter.GaussianBlur(2))

        holographic_img.save(output_path)
        plt.imshow(holographic_img, cmap='gray')
        plt.title('Holographic Effect with Particle-Specific Interference Patterns')
        plt.axis('off')
        plt.show()

        # Print and return the path of the holographic image
        print(f'Holographic image saved at: {output_path}')
        return output_path

class ParticleAnalyzer:
    def __init__(self, image_size=(1024, 1024)):
        self.image_size = image_size

    def detect_and_analyze_particles(self, image_path):
        """Detect and analyze particles in the simulated image."""
        output_path = 'detected_particles.png'
        img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Apply edge detection on the original particle image
        _, thresh_img = cv2.threshold(img_cv, 50, 255, cv2.THRESH_BINARY)
        blurred = cv2.GaussianBlur(thresh_img, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_contour_area = 50  # Adjusted for larger particles
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
        particle_types = ['microplastic', 'nutrient', 'microorganism', 'sediment']
        particle_size = {
            'microplastic': (5, 10),
            'microorganism': (10, 25),
            'nutrient': (20, 35),
            'sediment': (40, 60)
        }

        # Define specific colors for each particle type
        particle_colors = {
            'microplastic': 'blue',
            'microorganism': 'green',
            'nutrient': 'orange',
            'sediment': 'red'
        }

        type_labels = []
        for area in areas:
            if area <= particle_size['microplastic'][1] ** 2:
                type_labels.append('microplastic')
            elif area <= particle_size['microorganism'][1] ** 2:
                type_labels.append('microorganism')
            elif area <= particle_size['nutrient'][1] ** 2:
                type_labels.append('nutrient')
            else:
                type_labels.append('sediment')

        type_counts = {typ: type_labels.count(typ) for typ in particle_types}
        print("Particle type counts:", type_counts)

        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))

        for typ in particle_types:
            typ_areas = [a for a, l in zip(areas, type_labels) if l == typ]
            plt.hist(typ_areas, bins=20, alpha=0.5, label=f'{typ} particles', color=particle_colors[typ])

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
        dy = np.ones_like(x)  # Bar depth
        dz = np.array(areas)

        cmap = plt.get_cmap('viridis')
        colors = cmap(dz / np.max(dz))

        ax.bar3d(x, y, z, dx, dy, dz, color=colors, zsort='average')

        ax.set_xlabel('Particle Index')
        ax.set_ylabel('Depth')
        ax.set_zlabel('Area (pixels)')
        ax.set_title('3D Visualization of Particle Size Distribution')

        plt.show()


# Sample usage:
generator = ParticleImageGenerator()
generator.generate_particles_image(num_microplastics=100, num_nutrients=150, num_microorganisms=120, num_sediments=80)

analyzer = ParticleAnalyzer()
areas, density = analyzer.detect_and_analyze_particles('simulated_particles.png')
analyzer.visualize_particle_distribution(areas)
analyzer.visualize_3d_particle_distribution(areas)

