import numpy as np

from utils import *

"""
Core implementation of the ray tracer.  This module contains the classes (Sphere, Mesh, etc.)
that define the contents of scenes, as well as classes (Ray, Hit) and functions (shade) used in
the rendering algorithm, and the main entry point `render_image`.

In the documentation of these classes, we indicate the expected types of arguments with a
colon, and use the convention that just writing a tuple means that the expected type is a
NumPy array of that shape.  Implementations can assume these types are preconditions that
are met, and if they fail for other type inputs it's an error of the caller.  (This might
not be the best way to handle such validation in industrial-strength code but we are adopting
this rule to keep things simple and efficient.)
"""


class Ray:

    def __init__(self, origin, direction, start=0., end=np.inf):
        """Create a ray with the given origin and direction.

        Parameters:
          origin : (3,) -- the start point of the ray, a 3D point
          direction : (3,) -- the direction of the ray, a 3D vector (not necessarily normalized)
          start, end : float -- the minimum and maximum t values for intersections
        """
        # Convert these vectors to double to help ensure intersection
        # computations will be done in double precision
        self.origin = np.array(origin, np.float64)
        self.direction = np.array(direction, np.float64)
        self.start = start
        self.end = end


class Material:

    def __init__(self, k_d, k_s=0., p=20., k_m=0., k_a=None):
        """Create a new material with the given parameters.

        Parameters:
          k_d : (3,) -- the diffuse coefficient
          k_s : (3,) or float -- the specular coefficient
          p : float -- the specular exponent
          k_m : (3,) or float -- the mirror reflection coefficient
          k_a : (3,) -- the ambient coefficient (defaults to match diffuse color)
        """
        self.k_d = k_d
        self.k_s = k_s
        self.p = p
        self.k_m = k_m
        self.k_a = k_a if k_a is not None else k_d


class Hit:

    def __init__(self, t, point=None, normal=None, material=None):
        """Create a Hit with the given data.

        Parameters:
          t : float -- the t value of the intersection along the ray
          point : (3,) -- the 3D point where the intersection happens
          normal : (3,) -- the 3D outward-facing unit normal to the surface at the hit point
          material : (Material) -- the material of the surface
        """
        self.t = t
        self.point = point
        self.normal = normal
        self.material = material

# Value to represent absence of an intersection
no_hit = Hit(np.inf)


class Sphere:

    def __init__(self, center, radius, material):
        """Create a sphere with the given center and radius.

        Parameters:
          center : (3,) -- a 3D point specifying the sphere's center
          radius : float -- a Python float specifying the sphere's radius
          material : Material -- the material of the surface
        """
        self.center = center
        self.radius = radius
        self.material = material

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and this sphere.

        Parameters:
          ray : Ray -- the ray to intersect with the sphere
        Return:
          Hit -- the hit data
        """
        # TODO A4 implement this function
        a = np.dot(ray.direction, ray.direction)
        b = 2*np.dot(ray.origin - self.center, ray.direction)
        c = np.dot(ray.origin - self.center, ray.origin - self.center) - self.radius**2
        if (b**2 - 4*a*c) < 0:
            return no_hit
        t1 = (-b+np.sqrt(b**2 - 4*a*c))/(2*a)
        t2 = (-b-np.sqrt(b**2 - 4*a*c))/(2*a)

        if t1 < t2:
            if ray.start <= t1 <= ray.end:
                normal = normalize(ray.origin + ray.direction*t1 - self.center)
                return Hit(t1, ray.origin + ray.direction*t1, normal, self.material)
            else:
                if ray.start <= t2 <= ray.end:
                    normal = normalize(ray.origin + ray.direction*t2 - self.center)
                    return Hit(t2, ray.origin + ray.direction*t2, normal, self.material)
        else:
            if ray.start <= t2 <= ray.end:
                normal = normalize(ray.origin + ray.direction * t1 - self.center)
                return Hit(t2, ray.origin + ray.direction*t2, normal, self.material)
            else:
                if ray.start <= t1 <= ray.end:
                    normal = normalize(ray.origin + ray.direction * t1 - self.center)
                    return Hit(t1, ray.origin + ray.direction * t1, normal, self.material)
        return no_hit


class Triangle:

    def __init__(self, vs, material):
        """Create a triangle from the given vertices.

        Parameters:
          vs (3,3) -- an arry of 3 3D points that are the vertices (CCW order)
          material : Material -- the material of the surface
        """
        self.vs = vs
        self.material = material

    def intersect(self, ray):
        """Computes the intersection between a ray and this triangle, if it exists.

        Parameters:
          ray : Ray -- the ray to intersect with the triangle
        Return:
          Hit -- the hit data
        """
        # TODO A4 implement this function
        A = np.matrix(self.vs[0] - self.vs[1], self.vs[0] - self.vs[2], ray.direction)
        b = np.matrix(self.vs[0] - ray.origin)
        x = np.linalg.solve(A, b)
        if x[0] > 0 and x[1] > 0 and (x[0] + x[1]) < 1:
            if ray.start <= x[2] <= ray.end:
                normal = normalize(np.cross(self.vs[0] - self.vs[1], self.vs[0] - self.vs[2]))
                return Hit(x[2], ray.origin + ray.direction*x[2], normal, self.material)
        return no_hit


class Camera:

    def __init__(self, eye=vec([0,0,0]), target=vec([0,0,-1]), up=vec([0,1,0]),
                 vfov=90.0, aspect=1.0):
        """Create a camera with given viewing parameters.

        Parameters:
          eye : (3,) -- the camera's location, aka viewpoint (a 3D point)
          target : (3,) -- where the camera is looking: a 3D point that appears centered in the view
          up : (3,) -- the camera's orientation: a 3D vector that appears straight up in the view
          vfov : float -- the full vertical field of view in degrees
          aspect : float -- the aspect ratio of the camera's view (ratio of width to height)
        """
        self.eye = eye
        self.aspect = aspect
        self.f = None; # you should set this to the distance from your center of projection to the image plane
        self.M = np.eye(4);  # set this to the matrix that transforms your camera's coordinate system to world coordinates
        # TODO A4 implement this constructor to store whatever you need for ray generation
        self.target = target
        self.up = up
        self.vfov = vfov

    def generate_ray(self, img_point):
        """Compute the ray corresponding to a point in the image.

        Parameters:
          img_point : (2,) -- a 2D point in [0,1] x [0,1], where (0,0) is the upper left
                      corner of the image and (1,1) is the lower right.
        Return:
          Ray -- The ray corresponding to that image location (not necessarily normalized)
        """
        # TODO A4 implement this function
        return Ray(vec([img_point[0],img_point[1],0]), self.target)


class PointLight:

    def __init__(self, position, intensity):
        """Create a point light at given position and with given intensity

        Parameters:
          position : (3,) -- 3D point giving the light source location in scene
          intensity : (3,) or float -- RGB or scalar intensity of the source
        """
        self.position = position
        self.intensity = intensity

    def illuminate(self, ray, hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        # TODO A4 implement this function
        hit_point = hit.point
        normal = hit.normal
        material = hit.material  # Material properties, such as color and reflectivity

        # Direction from hit point to light
        light_direction = self.position - hit_point
        light_distance = np.sum(light_direction)
        light_direction = light_direction/light_distance # Normalize the light direction

        ray_length = ray.end - ray.start
        v = -ray.direction/np.sum(ray_length)

        h = (v+light_direction)/np.sum(v+light_direction)
        specular = material.k_s*(np.dot(normal, h))**material.p

        # Compute the diffuse component (Lambertian shading)
        irradiance = max(np.dot(normal, light_direction), 0.0)

        shading_contribution = irradiance/(light_distance**2) * self.intensity * (material.k_d + specular)

        # Cast a shadow ray to see if the point is in shadow
        shadow_ray = Ray(hit_point + normal * 0.001, light_direction)  # A small offset to avoid self-intersection

        if not scene.intersect(shadow_ray):  # Check if any object blocks the light
            return shading_contribution
        else:
            return vec([0,0,0])  # Return black (no light) if in shadow

        # Calculate the diffuse color contribution





class AmbientLight:

    def __init__(self, intensity):
        """Create an ambient light of given intensity

        Parameters:
          intensity (3,) or float: the intensity of the ambient light
        """
        self.intensity = intensity

    def illuminate(self, ray, hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        # TODO A4 implement this function

        return self.intensity * hit.material.k_a


class Scene:

    def __init__(self, surfs, bg_color=vec([0.2,0.3,0.5])):
        """Create a scene containing the given objects.

        Parameters:
          surfs : [Sphere, Triangle] -- list of the surfaces in the scene
          bg_color : (3,) -- RGB color that is seen where no objects appear
        """
        self.surfs = surfs
        self.bg_color = bg_color

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and the scene.

        Parameters:
          ray : Ray -- the ray to intersect with the scene
        Return:
          Hit -- the hit data
        """
        # TODO A4 implement this function
        closest_hit = None
        closest_t = np.inf

        for surf in self.surfs:
            # Intersect the ray with the surface (Sphere or Triangle)
            hit = surf.intersect(ray)

            # If the hit is valid and closer than the previous closest, update the closest hit
            if hit and hit.t < closest_t:
                closest_hit = hit
                closest_t = hit.t

        return closest_hit
        # return self.surfs[0].intersect(ray)


MAX_DEPTH = 4

def shade(ray, hit, scene, lights, depth=0):
    """Compute shading for a ray-surface intersection.

    Parameters:
      ray : Ray -- the ray that hit the surface
      hit : Hit -- the hit data
      scene : Scene -- the scene
      lights : [PointLight or AmbientLight] -- the lights
      depth : int -- the recursion depth so far
    Return:
      (3,) -- the color seen along this ray
    When mirror reflection is being computed, recursion will only proceed to a depth
    of MAX_DEPTH, with zero contribution beyond that depth.
    """
    # TODO A4 implement this function
    return vec([0,0,0])


def render_image(camera, scene, lights, nx, ny):
    """Render a ray traced image.

    Parameters:
      camera : Camera -- the camera defining the view
      scene : Scene -- the scene to be rendered
      lights : Lights -- the lights illuminating the scene
      nx, ny : int -- the dimensions of the rendered image
    Returns:
      (ny, nx, 3) float32 -- the RGB image
    """
    # TODO A4 implement this function
    output_image = np.zeros((ny, nx, 3), np.float32)
    for i in range(ny):
        for j in range(nx):
            ray = camera.generate_ray(vec([i, j])) # Generate Ray---we recommend just generating an orthographic ray to start with
            intersection = scene.surfs[0].intersect(ray)  # this will return a Hit object

            # set the output pixel color if an intersection is found
            if intersection is not no_hit:
                output_image[i, j] = vec([1.0, 1.0, 1.0])
            else:
                output_image[i, j] = vec([0.0, 0.0, 0.0])

    return output_image
