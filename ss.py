import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt

class ParticleImageGenerator:
    def __init__(self, image_size=(1024, 1024)):
        self.image_size = image_size

    def generate_particles_image(self, num_microplastics, num_nutrients, num_microorganisms, num_sediments):
        """Generate an image with user-defined numbers of particles."""
        img = Image.new('L', self.image_size, color=0)
        draw = ImageDraw.Draw(img)

        # Define particle types, sizes, and colors in order of increasing size
        particle_types = {
            'microplastic': {'size': (2, 5), 'count': num_microplastics, 'color': 100},
            'microorganism': {'size': (5, 15), 'count': num_microorganisms, 'color': 150},
            'nutrient': {'size': (10, 20), 'count': num_nutrients, 'color': 120},
            'sediment': {'size': (20, 40), 'count': num_sediments, 'color': 180}
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

# Example usage for Part 1
num_microplastics = int(input("Enter the number of microplastic particles: "))
num_nutrients = int(input("Enter the number of nutrient particles: "))
num_microorganisms = int(input("Enter the number of microorganism particles: "))
num_sediments = int(input("Enter the number of sediment particles: "))

generator = ParticleImageGenerator(image_size=(1024, 1024))
generator.generate_particles_image(num_microplastics, num_nutrients, num_microorganisms, num_sediments)
holographic_image_path = generator.create_holographic_effect('simulated_particles.png',
                                                           'holographic_particles_with_interference.png')

# Print the path of the holographic image
print(f'Holographic image path: {holographic_image_path}')



