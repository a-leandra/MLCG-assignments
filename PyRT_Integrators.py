from PyRT_Common import *
from PyRT_Core import *
from random import randint, uniform
from AppWorkbench import *
import numpy as np


# -------------------------------------------------
# Integrator Classes
# -------------------------------------------------
# The integrators also act like a scene class in that-
# it stores all the primitives that are to be ray traced.
# -------------------------------------------------
class Integrator(ABC):
    # Initializer - creates object list
    def __init__(self, filename_, experiment_name=''):
        # self.primitives = []
        self.filename = filename_ + experiment_name
        # self.env_map = None  # not initialized
        self.scene = None

    @abstractmethod
    def compute_color(self, ray):
        pass

    # def add_environment_map(self, env_map_path):
    #    self.env_map = EnvironmentMap(env_map_path)
    def add_scene(self, scene):
        self.scene = scene

    def get_filename(self):
        return self.filename

    # Simple render loop: launches 1 ray per pixel
    def render(self):
        # YOU MUST CHANGE THIS METHOD IN ASSIGNMENTS 1.1 and 1.2:
        cam = self.scene.camera  # camera object
        # ray = Ray()
        print('Rendering Image: ' + self.get_filename())
        for x in range(0, cam.width):
            for y in range(0, cam.height):
                # changes for the ASSIGNMENT 1.1
                #pixel = RGBColor(x / cam.width, y / cam.height, 0)
                #self.scene.set_pixel(pixel, x, y)  # save pixel to pixel array
                
                # changes for the ASSIGNMENT 1.2
                ray_origin = Vector3D(0, 0, 0)
                ray_direction = cam.get_direction(x, y)
                ray = Ray(ray_origin, ray_direction)

                color = self.compute_color(ray)
                self.scene.set_pixel(color, x, y)  # save pixel to pixel array
            progress = (x / cam.width) * 100
            print('\r\tProgress: ' + str(progress) + '%', end='')
        # save image to file
        print('\r\tProgress: 100% \n\t', end='')
        full_filename = self.get_filename()
        self.scene.save_image(full_filename)


class LazyIntegrator(Integrator):
    def __init__(self, filename_):
        super().__init__(filename_ + '_Lazy')

    def compute_color(self, ray):
        return BLACK


class IntersectionIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Intersection')

    def compute_color(self, ray):
        # ASSIGNMENT 1.2: PUT YOUR CODE HERE
        if self.scene.any_hit(ray):
            return RGBColor(1, 0, 0)  # Red color if there is an intersection
        else:
            return RGBColor(0, 0, 0)  # Black color if there is no intersection


class DepthIntegrator(Integrator):

    def __init__(self, filename_, max_depth_=5):
        super().__init__(filename_ + '_Depth')
        self.max_depth = max_depth_

    def compute_color(self, ray):
        # ASSIGNMENT 1.3: PUT YOUR CODE HERE
        hit_data = self.scene.closest_hit(ray)  # Get closest intersection data
        if hit_data.has_hit:  # Check if there's an intersection
            # Calculate gray-scale value based on distance
            depth_color = max(1 - hit_data.hit_distance / self.max_depth, 0)
            # Use the gray-scale value for all color components
            return RGBColor(depth_color, depth_color, depth_color)
        else:
            return BLACK  # Black color if no intersection


class NormalIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Normal')

    def compute_color(self, ray):
        # ASSIGNMENT 1.3: PUT YOUR CODE HERE
        hit_data = self.scene.closest_hit(ray)  # Get closest intersection data
        if hit_data.has_hit:  # Check if there's an intersection
            # Calculate color based on the normal vector
            normal = hit_data.normal
            color = (normal + Vector3D(1, 1, 1)) / 2
            return RGBColor(color.x, color.y, color.z)
        else:
            return BLACK  # Black color if no intersection


class PhongIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Phong')
        self.specular_exponent = 50  # Shininess coefficient

    def compute_color(self, ray):
        # ASSIGNMENT 1.4: PUT YOUR CODE HERE
        color = RGBColor(0.0, 0.0, 0.0)  # Start with black, add light contributions
        hit_data = self.scene.closest_hit(ray)

        if hit_data.has_hit:
            # Start with ambient light
            color += self.scene.i_a

            # Compute for each light
            for light in self.scene.pointLights:
                # Check for shadows first
                shadow_ray = Ray(hit_data.hit_point, Normalize(light.pos - hit_data.hit_point))
                if self.scene.any_hit(shadow_ray):
                    continue  # Skip this light source if shadowed

                # Calculate vectors needed for Phong model
                L = Normalize(light.pos - hit_data.hit_point)
                N = hit_data.normal
                V = Normalize(Vector3D(-ray.d.x, -ray.d.y, -ray.d.z))
                R = Normalize(N * 2 * Dot(N, L) - L)

                # Diffuse component using Lambertian reflection
                diffuse_intensity = max(Dot(N, L), 0)
                kd = RGBColor(1.0, 1.0, 1.0)
                color += light.intensity.multiply(kd) * diffuse_intensity

                # Specular component
                specular_intensity = max(Dot(V, R), 0) ** self.specular_exponent
                ks = RGBColor(1.0, 1.0, 1.0)
                color += light.intensity.multiply(ks) * specular_intensity

        else:
            color = BLACK

        return color

class CMCIntegrator(Integrator):  # Classic Monte Carlo Integrator

    def __init__(self, n, filename_, experiment_name=''):
        filename_mc = filename_ + '_MC_' + str(n) + '_samples' + experiment_name
        super().__init__(filename_mc)
        self.n_samples = n

    def compute_color(self, ray):
        color = RGBColor(0.0, 0.0, 0.0)  # Start with black, add light contributions

        hit_data = self.scene.closest_hit(ray)
        if not hit_data.has_hit:
            # If the ray doesn't hit anything, return the background color
            return self.scene.env_map.getValue(ray.d) if self.scene.env_map else WHITE

        # Assume that all objects in the scene have a BRDF directly assigned
        hit_brdf = self.scene.object_list[hit_data.primitive_index].BRDF

        # Generate a set of samples over the hemisphere
        sample_set, sample_prob = sample_set_hemisphere(self.n_samples, uniform_pdf)
        for omega_j, prob in zip(sample_set, sample_prob):
            # Rotate sample to be around the normal at the hit point
            omega_j_prime = center_around_normal(omega_j, hit_data.normal)

            # Shoot a secondary ray from the hit point in the direction of the sample
            secondary_ray = Ray(hit_data.hit_point, omega_j_prime)
            secondary_hit_data = self.scene.closest_hit(secondary_ray)

            if secondary_hit_data.has_hit:
                # If the secondary ray hits the scene geometry, use the object's emission
                Li = self.scene.object_list[secondary_hit_data.primitive_index].emission
            else:
                # If there's no hit, use the environment map if available
                Li = self.scene.env_map.getValue(omega_j_prime) if self.scene.env_map else RED

        #cos_theta = max(Dot(hit_data.normal, omega_j_prime), 0)
            
            brdf_value = hit_brdf.get_value(omega_j_prime, Vector3D(-ray.d.x, -ray.d.y, -ray.d.z), hit_data.normal)
            color += Li.multiply(brdf_value) / prob

        # Average the color by the number of samples
        color /= float(self.n_samples)

        return color

def rotate_around_y(pos, angle):
    # Rotate a position vector around the Y-axis by a given angle
    rotation_matrix = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    rotated_pos = np.dot(rotation_matrix, [pos.x, pos.y, pos.z])
    return Vector3D(rotated_pos[0], rotated_pos[1], rotated_pos[2])


def rotate_around_normal(pos, normal):
    # Ensure normal is a numpy array
    normal = np.array([normal.x, normal.y, normal.z])
    pos = np.array([pos.x, pos.y, pos.z])

    # Define the up vector
    up_vector = np.array([0, 1, 0])

    if np.allclose(normal, up_vector):
        return Vector3D(pos[0], pos[1], pos[2])  # No rotation needed if normal is already the up vector.

    rotation_axis = np.cross(up_vector, normal)
    rotation_angle = np.arccos(np.dot(up_vector, normal) / (np.linalg.norm(up_vector) * np.linalg.norm(normal)))
    rotation_matrix = axis_angle_rotation_matrix(rotation_axis, rotation_angle)
    rotated_pos = np.dot(rotation_matrix, pos)

    return Vector3D(rotated_pos[0], rotated_pos[1], rotated_pos[2])

def axis_angle_rotation_matrix(axis, angle):
    # Create a rotation matrix given an axis and an angle
    axis = axis / np.linalg.norm(axis)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    one_minus_cos = 1.0 - cos_angle

    x, y, z = axis
    return np.array([
        [cos_angle + x * x * one_minus_cos, x * y * one_minus_cos - z * sin_angle, x * z * one_minus_cos + y * sin_angle],
        [y * x * one_minus_cos + z * sin_angle, cos_angle + y * y * one_minus_cos, y * z * one_minus_cos - x * sin_angle],
        [z * x * one_minus_cos - y * sin_angle, z * y * one_minus_cos + x * sin_angle, cos_angle + z * z * one_minus_cos]
    ])


class BayesianMonteCarloIntegrator(Integrator):
    def __init__(self, n, gp_list, filename_, experiment_name=''):
        if not isinstance(gp_list, list):
            raise TypeError("gp_list should be a list of Gaussian Process instances.")
        filename_bmc = filename_ + '_BMC_' + str(n) + '_samples' + experiment_name
        super().__init__(filename_bmc)
        self.n_samples = n
        self.gp_list = gp_list

    def compute_color(self, ray):
        color = RGBColor(0.0, 0.0, 0.0)  # Start with black, add light contributions

        hit_data = self.scene.closest_hit(ray)
        if not hit_data.has_hit:
            # If the ray doesn't hit anything, return the background color
            return self.scene.env_map.getValue(ray.d) if self.scene.env_map else WHITE

        # Assume that all objects in the scene have a BRDF directly assigned
        hit_brdf = self.scene.object_list[hit_data.primitive_index].BRDF

        # Select a random GP from the precomputed list
        myGP = self.gp_list[randint(0, len(self.gp_list) - 1)]

        # Apply a random rotation around the Y-axis to the sample set
        rotation_angle = uniform(0, 2 * np.pi)
        rotated_samples_y = [rotate_around_y(pos, rotation_angle) for pos in myGP.samples_pos]

        # Rotate the sample positions to align with the normal at the hit point
        rotated_samples = [rotate_around_normal(pos, hit_data.normal) for pos in rotated_samples_y]

        # Sample the integrand using the rotated sample set
        sample_values = []
        for omega_j_prime in rotated_samples:
            # Shoot a secondary ray from the hit point in the direction of the sample
            secondary_ray = Ray(hit_data.hit_point, omega_j_prime)
            secondary_hit_data = self.scene.closest_hit(secondary_ray)

            if secondary_hit_data.has_hit:
                # If the secondary ray hits the scene geometry, use the object's emission
                Li = self.scene.object_list[secondary_hit_data.primitive_index].emission
            else:
                # If there's no hit, use the environment map if available
                Li = self.scene.env_map.getValue(omega_j_prime) if self.scene.env_map else RGBColor(1.0, 0.0, 0.0)

            brdf_value = hit_brdf.get_value(omega_j_prime, Vector3D(-ray.d.x, -ray.d.y, -ray.d.z), hit_data.normal)
            sample_values.append(Li.multiply(brdf_value))

        # Flatten the sample values to use only the red channel (or the combined color intensity if needed)
        flattened_sample_values = [sv.r for sv in sample_values]

        # Add the sample values to the Gaussian Process
        myGP.add_sample_val(flattened_sample_values)

        # Compute the integral using Bayesian Monte Carlo
        color_value = myGP.compute_integral_BMC()

        return RGBColor(color_value, 0, 0)
