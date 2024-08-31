import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

# Global variables
points = []
water_source = None
town_border_limits = (0, 0, 0, 0)


def preprocess_image(image_path):
    image = cv2.imread(image_path.strip())
    if image is None:
        raise FileNotFoundError(f"Failed to load image from: {image_path.strip()}")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    return blurred_image, image


def detect_borders(preprocessed_image):
    edges = cv2.Canny(preprocessed_image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def find_centroids(contours):
    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))
    return centroids


def get_town_border_limits(town_contour, image):
    x, y, w, h = cv2.boundingRect(town_contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return (x, y, x + w, y + h)


def on_click(event):
    global points, water_source
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        if town_border_limits[0] <= x <= town_border_limits[2] and town_border_limits[1] <= y <= town_border_limits[3]:
            if water_source is None:
                water_source = (x, y)
                plt.scatter(x, y, color='blue', label='Water Source')
            else:
                points.append((x, y))
                plt.scatter(x, y, color='red', label='Building')
            plt.legend()
            plt.draw()
        else:
            print("Coordinate is out of bounds. Please select a point within the town's border.")


def select_points_on_image(image, title):
    global points, water_source
    points = []
    water_source = None

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('on')
    plt.connect('button_press_event', on_click)
    plt.show()

    return points, water_source


def connect_buildings_to_network(image, points, water_source, scale_km_per_cm):
    points_with_source = points + [water_source]
    distances = distance_matrix(points_with_source, points_with_source)

    mst = minimum_spanning_tree(distances).toarray().astype(float)

    image_with_connections = image.copy()
    total_distance_km = 0

    for i in range(len(points_with_source)):
        for j in range(len(points_with_source)):
            if mst[i, j] > 0:
                cv2.line(image_with_connections, tuple(map(int, points_with_source[i])),
                         tuple(map(int, points_with_source[j])), (0, 255, 0), 2)
                dist_cm = mst[i, j]
                dist_km = dist_cm * scale_km_per_cm
                total_distance_km += dist_km

    for point in points:
        cv2.circle(image_with_connections, tuple(map(int, point)), 5, (0, 0, 255), -1)
    if water_source:
        cv2.circle(image_with_connections, tuple(map(int, water_source)), 5, (255, 0, 0), -1)

    return image_with_connections, total_distance_km


def draw_and_save_contours_with_connections(image_path, contours, output_path, points, water_source,
                                            water_body_contour):
    _, image = preprocess_image(image_path)

    if contours:
        for contour in contours:
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            cv2.drawContours(image, [contour], -1, color, 2)

    image_with_connections, total_distance_km = connect_buildings_to_network(image, points, water_source,
                                                                             scale_km_per_cm)
    if water_body_contour is not None:
        cv2.drawContours(image_with_connections, [water_body_contour], -1, (255, 0, 0), 2)
    cv2.imwrite(output_path, image_with_connections)
    return total_distance_km


def display_image_with_contours(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title('Detected Town Borders with Water Supply System')
    plt.show()


def main():
    global scale_km_per_cm

    image_path = input("Enter the path to the image: ").strip()
    scale_km_per_cm = float(input("Enter the scale (km per cm): ").strip())
    contours_output_path = "town_with_water_supply.jpeg"
    preprocessed_image, image = preprocess_image(image_path)

    contours = detect_borders(preprocessed_image)
    town_contour = max(contours, key=cv2.contourArea) if contours else None

    global town_border_limits
    if town_contour is not None:
        town_border_limits = get_town_border_limits(town_contour, image)
        print(f"Town border limits: {town_border_limits}")
    else:
        print("No town border detected.")

    print("Select the coordinates for buildings and the water source by clicking on the map.")
    points, water_source = select_points_on_image(image, "Select Buildings and Water Source")

    # Replace with actual water body detection if needed
    water_body_contour = None  # Modify this if water body detection is needed

    total_distance_km = draw_and_save_contours_with_connections(image_path, contours, contours_output_path, points,
                                                                water_source, water_body_contour)
    display_image_with_contours(contours_output_path)
    print(f"Total distance of the network: {total_distance_km:.2f} km")


if __name__ == "__main__":
    main()


