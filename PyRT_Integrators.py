from PyRT_Common import *
from PyRT_Core import *
from random import randint


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

    def compute_color2(self, ray):
        # ASSIGNMENT 1.4: PUT YOUR CODE HERE
        color = RGBColor(0.0, 0.0, 0.0)  # Start with black, add light contributions

        hit_data = self.scene.closest_hit(ray)

        if hit_data.has_hit:
            # Start with ambient light
            color += self.scene.i_a

            # Compute for each light
            for light in self.scene.pointLights:
                # Calculate vectors needed for Phong model
                light_vector = light.pos - hit_data.hit_point
                light_distance = Length(light_vector)
                L = Normalize(light_vector)
                N = hit_data.normal

                # Diffuse component using Lambertian reflection
                kd = self.scene.object_list[hit_data.primitive_index].get_BRDF().get_value(light_vector, None, N)
                color += light.intensity.multiply(kd) / (light_distance * light_distance)

        else:
            color = BLACK

        return color

    def compute_color(self, ray):
        # ASSIGNMENT 1.4: PUT YOUR CODE HERE
        color = RGBColor(0.0, 0.0, 0.0)  # Start with black, add light contributions

        hit_data = self.scene.closest_hit(ray)

        if hit_data.has_hit:
            # Start with ambient light
            color += self.scene.i_a

            # Compute for each light
            for light in self.scene.pointLights:
                light_vector = light.pos - hit_data.hit_point # (d) calculate the distance from the hit point to the light source
                light_distance = Length(light_vector)
                L = Normalize(light_vector)
                N = hit_data.normal

                # Check for shadows first
                shadow_ray = Ray(hit_data.hit_point, L, tmax=light_distance) # set tmax to d
                if self.scene.any_hit(shadow_ray):
                    continue  # Skip this light source if shadowed

                # Diffuse component using Lambertian reflection
                kd = self.scene.object_list[hit_data.primitive_index].get_BRDF().get_value(light_vector, None, N)
                color += light.intensity.multiply(kd) / (light_distance * light_distance) # Li / d^2
        else:
            # Return black if no hit
            color = BLACK

        return color

class CMCIntegrator(Integrator):  # Classic Monte Carlo Integrator

    def __init__(self, n, filename_, experiment_name=''):
        filename_mc = filename_ + '_MC_' + str(n) + '_samples' + experiment_name
        super().__init__(filename_mc)
        self.n_samples = n

    def compute_color(self, ray):
        pass


class BayesianMonteCarloIntegrator(Integrator):
    def __init__(self, n, myGP, filename_, experiment_name=''):
        filename_bmc = filename_ + '_BMC_' + str(n) + '_samples' + experiment_name
        super().__init__(filename_bmc)
        self.n_samples = n
        self.myGP = myGP

    def compute_color(self, ray):
        pass