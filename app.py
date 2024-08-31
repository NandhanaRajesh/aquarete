import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageTk
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import tkinter as tk
from tkinter import filedialog, messagebox
from io import BytesIO


class ParticleDetectorApp:
    def __init__(self, num_particles=500, image_size=(512, 512)):
        self.num_particles = num_particles
        self.image_size = image_size

    def generate_particles_image(self):
        img = Image.new('L', self.image_size, color=0)
        draw = ImageDraw.Draw(img)

        particle_types = ['Microplastics', 'MICROORGANISMS', 'OrganicMatter', 'CHEMICALS', 'SEDIMENTS']
        particle_size = {
            'Microplastics': 20,
            'MICROORGANISMS': 16,
            'OrganicMatter': 8,
            'CHEMICALS': 5,
            'SEDIMENTS': 22
        }
        colors = {
            'Microplastics': 1000,
            'MICROORGANISMS': 400,
            'OrganicMatter': 200,
            'CHEMICALS': 2550,
            'SEDIMENTS': 125
        }

        for _ in range(self.num_particles):
            x, y = np.random.randint(0, self.image_size[0], 2)
            particle_type = np.random.choice(particle_types)
            size = particle_size[particle_type]
            color = colors[particle_type]
            draw.ellipse((x, y, x + size, y + size), fill=color)

        img_path = 'simulated_particles.png'
        img.save(img_path)
        return img_path

    def create_holographic_effect(self, image_path, output_path):
        img = Image.open(image_path).convert('L')
        img_np = np.array(img)

        height, width = img_np.shape
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)

        interference = (np.sin(X / 20.0) + np.cos(Y / 20.0)) * 0.5
        holographic_img_np = np.clip(img_np + 50 * interference, 0, 255).astype(np.uint8)
        holographic_img = Image.fromarray(holographic_img_np)
        holographic_img = holographic_img.filter(ImageFilter.GaussianBlur(2))

        holographic_img.save(output_path)
        return output_path

    def detect_and_analyze_particles(self, image_path, output_path):
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

        num_detected_particles = len(filtered_contours)
        surface_area_cm2 = (self.image_size[0] / 100.0) * (self.image_size[1] / 100.0)
        density_particles_per_cm2 = num_detected_particles / surface_area_cm2

        areas = [cv2.contourArea(c) for c in filtered_contours]

        return areas, density_particles_per_cm2, output_path

    def visualize_particle_distribution(self, areas):
        particle_types = ['Microplastics', 'MICROORGANISMS', 'OrganicMatter', 'CHEMICALS', 'SEDIMENTS']
        particle_size = {
            'Microplastics': 20,
            'MICROORGANISMS': 16,
            'OrganicMatter': 8,
            'CHEMICALS': 5,
            'SEDIMENTS': 22
        }

        type_labels = []
        for area in areas:
            if area <= particle_size['Microplastics'] ** 2:
                type_labels.append('Microplastics')
            elif area <= particle_size['MICROORGANISMS'] ** 2:
                type_labels.append('MICROORGANISMS')
            elif area <= particle_size['OrganicMatter'] ** 2:
                type_labels.append('OrganicMatter')
            elif area <= particle_size['CHEMICALS'] ** 2:
                type_labels.append('CHEMICALS')
            elif area <= particle_size['SEDIMENTS'] ** 2:
                type_labels.append('SEDIMENTS')

        type_counts = {typ: type_labels.count(typ) for typ in particle_types}

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
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        return img

    def visualize_3d_particle_distribution(self, areas):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        x = np.arange(len(areas))
        y = areas
        z = np.zeros_like(x)
        dx = np.ones_like(x) * 0.5
        dy = np.ones_like(y) * 0.5
        dz = areas

        colors = cm.viridis(np.linspace(0, 1, len(areas)))

        for i in range(len(areas)):
            ax.bar3d(x[i], y[i], z[i], dx[i], dy[i], dz[i], color=colors[i], alpha=0.8)

        ax.set_xlabel('Index')
        ax.set_ylabel('Area (pixels)')
        ax.set_zlabel('Frequency')
        plt.title('3D Visualization of Particle Sizes')
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        return img

class AppUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Particle Detector App")
        self.geometry("600x500")

        self.app = ParticleDetectorApp()
        self.create_widgets()

    def create_widgets(self):
        tk.Button(self, text="Generate Particles Image", command=self.generate_particles_image).pack(pady=10)
        tk.Button(self, text="Create Holographic Effect", command=self.create_holographic_effect).pack(pady=10)
        tk.Button(self, text="Detect and Analyze Particles", command=self.detect_and_analyze_particles).pack(pady=10)
        tk.Button(self, text="Visualize Particle Distribution", command=self.visualize_particle_distribution).pack(pady=10)
        tk.Button(self, text="Visualize 3D Particle Distribution", command=self.visualize_3d_particle_distribution).pack(pady=10)

        self.output_label = tk.Label(self, text="", wraplength=500)
        self.output_label.pack(pady=10)

        self.img_label = tk.Label(self)
        self.img_label.pack(pady=10)

    def update_output(self, message):
        self.output_label.config(text=message)

    def show_image(self, image):
        image_tk = ImageTk.PhotoImage(image)
        self.img_label.config(image=image_tk)
        self.img_label.image = image_tk

    def generate_particles_image(self):
        img_path = self.app.generate_particles_image()
        self.update_output(f"Particles image generated: {img_path}")
        img = Image.open(img_path)
        self.show_image(img)

    def create_holographic_effect(self):
        input_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if input_path:
            output_path = 'holographic_particles_with_interference.png'
            output_path = self.app.create_holographic_effect(input_path, output_path)
            self.update_output(f"Holographic effect created: {output_path}")
            img = Image.open(output_path)
            self.show_image(img)

    def detect_and_analyze_particles(self):
        input_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if input_path:
            output_path = 'detected_particles.png'
            areas, density, contours_path = self.app.detect_and_analyze_particles(input_path, output_path)
            self.update_output(f"Particles detected. Density: {density:.2f} particles/cm²")
            img = Image.open(contours_path)
            self.show_image(img)
            self.areas = areas

    def visualize_particle_distribution(self):
        if hasattr(self, 'areas'):
            img = self.app.visualize_particle_distribution(self.areas)
            self.update_output("Particle distribution visualization")
            self.show_image(img)
        else:
            messagebox.showwarning("No Data", "Please run the particle detection step first.")

    def visualize_3d_particle_distribution(self):
        if hasattr(self, 'areas'):
            img = self.app.visualize_3d_particle_distribution(self.areas)
            self.update_output("3D particle distribution visualization")
            self.show_image(img)
        else:
            messagebox.showwarning("No Data", "Please run the particle detection step first.")

if __name__ == "__main__":
    app = AppUI()
    app.mainloop()
