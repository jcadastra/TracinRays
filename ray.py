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

    def __init__(self, k_d, k_s=0., p=20., k_m=0., k_a=None, opacity=1.0):
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
        self.opacity = opacity


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

class Cone:

    def __init__(self, center, radius, height, material):
        """Create a cone with the given center, height, and radius.

        Parameters:
          center : (3,) -- a 3D point specifying the cone's center
          radius : float -- a Python float specifying the cone's radius
          angle : float -- a Python float specifying the cone's height
          material : Material -- the material of the surface
          """
        self.center = center
        self.radius = radius
        self.height = height
        self.material = material
        self.angle = np.arctan(radius/height)

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and this cone.

        Parameters:
          ray : Ray -- the ray to intersect with the cone
        Return:
          Hit -- the hit data
        """
        direction = ray.direction
        dir_two = np.array(vec([direction[0], direction[2]]), dtype=np.float64)
        oc = ray.origin - self.center
        oc_two = np.array(vec([oc[0], oc[2]]), dtype=np.float64)

        a = np.dot(dir_two, dir_two) - (np.tan(self.angle) ** 2) * direction[1] ** 2
        b = 2.0 * (np.dot(oc_two, dir_two) - np.tan(self.angle) ** 2 * oc[1] * direction[1])
        c = np.dot(oc_two, oc_two) - (np.tan(self.angle) ** 2) * oc[1] ** 2

        determinant = b**2.0 - 4.0*a*c
        if determinant < 0.0:
            return no_hit

        t1 = (-b+np.sqrt(determinant))/(2.0*a)
        t2 = (-b-np.sqrt(determinant))/(2.0*a)

        if t1 >= t2:
            tmp = t1
            t1 = t2
            t2 = tmp

        for t in [t1, t2]:
            if ray.start <= t <= ray.end:
                hit_pt = ray.origin + t * direction
                if self.center[1] - self.height <= hit_pt[1] <= self.center[1]:
                    #normal = normalize(hit_pt - np.array(vec([hit_pt[0], self.center[1], hit_pt[2]]), dtype=np.float64))
                    normal = normalize(np.tan(self.angle)*(hit_pt - np.array(vec([self.center[0], hit_pt[1], self.center[2]]), dtype=np.float64)))
                    #normal[1] = -np.tan(self.angle)
                    return Hit(t, ray.origin + direction*t1, normal, self.material)

        return no_hit

class Cylinder:

    def __init__(self, center, radius, height, material, orientation=np.array(vec([0,1,0]), dtype=np.float64)):
        """Create a cylinder with the given center, height, and radius.

        Parameters:
          center : (3,) -- a 3D point specifying the cylinder's center
          radius : float -- a Python float specifying the cylinder's radius
          height : float -- a Python float specifying the cylinder's height
          material : Material -- the material of the surface
        """
        self.center = center
        self.radius = radius
        self.height = height
        self.material = material
        self.orientation = orientation
        self.rot_matrix = np.eye(3)

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and this cylinder.

        Parameters:
          ray : Ray -- the ray to intersect with the cylinder
        Return:
          Hit -- the hit data
        """
        direction = ray.direction
        dir_two = np.array(vec([direction[0], direction[2]]), dtype=np.float64)
        oc = ray.origin - self.center
        oc_two = np.array(vec([oc[0], oc[2]]), dtype=np.float64)

        a = np.dot(dir_two, dir_two)
        b = 2.0*np.dot(oc_two, dir_two)
        c = np.dot(oc_two, oc_two) - self.radius**2
        if (b**2.0 - 4.0*a*c) < 0.0:
            return no_hit
        t1 = (-b+np.sqrt(b**2.0 - 4.0*a*c))/(2.0*a)
        t2 = (-b-np.sqrt(b**2.0 - 4.0*a*c))/(2.0*a)

        if t1 >= t2:
            tmp = t1
            t1 = t2
            t2 = tmp

        if t1 <= t2:
            if ray.start <= t1 <= ray.end:
                hit_pt = ray.origin + direction * t1
                if self.center[1] <= hit_pt[1] <= self.center[1]+self.height:
                    normal = normalize(hit_pt - np.array(vec([self.center[0], hit_pt[1], self.center[2]]), dtype=np.float64))
                    return Hit(t1, ray.origin + direction*t1, normal, self.material)
            else:
                if ray.start <= t2 <= ray.end:
                    hit_pt = ray.origin + direction * t2
                    if self.center[1] <= hit_pt[1] <= self.center[1] + self.height:
                        normal = normalize(hit_pt - np.array(vec([self.center[0], hit_pt[1], self.center[2]]), dtype=np.float64))
                        return Hit(t2, ray.origin + direction*t2, normal, self.material)

        return no_hit


class Sphere:

    def __init__(self, center, radius, material, normal_map=None):
        """Create a sphere with the given center and radius.

        Parameters:
          center : (3,) -- a 3D point specifying the sphere's center
          radius : float -- a Python float specifying the sphere's radius
          material : Material -- the material of the surface
        """
        self.center = center
        self.radius = radius
        self.material = material
        self.normal_map = normal_map

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and this sphere.

        Parameters:
          ray : Ray -- the ray to intersect with the sphere
        Return:
          Hit -- the hit data
        """
        # TODO A4 implement this function
        direction = ray.direction

        a = np.dot(direction, direction)
        b = 2.0*np.dot(ray.origin - self.center, direction)
        c = np.dot(ray.origin - self.center, ray.origin - self.center) - self.radius**2
        if (b**2.0 - 4.0*a*c) < 0.0:
            return no_hit
        t1 = (-b+np.sqrt(b**2.0 - 4.0*a*c))/(2.0*a)
        t2 = (-b-np.sqrt(b**2.0 - 4.0*a*c))/(2.0*a)

        if t1 >= t2:
            tmp = t1
            t1 = t2
            t2 = tmp

        if t1 <= t2:
            if ray.start <= t1 <= ray.end:
                normal = normalize(ray.origin + direction*t1 - self.center)
                if self.normal_map is not None:
                    uv = self.get_uv_coordinates(ray.origin + direction*t1)
                    sampled_normal = self.get_normal(uv)
                    normal = normalize(sampled_normal)
                return Hit(t1, ray.origin + direction*t1, normal, self.material)
            else:
                if ray.start <= t2 <= ray.end:
                    normal = normalize(ray.origin + direction*t2 - self.center)
                    if self.normal_map is not None:
                        uv = self.get_uv_coordinates(ray.origin + direction * t2)
                        sampled_normal = self.get_normal(uv)
                        normal = normalize(sampled_normal)
                    return Hit(t2, ray.origin + direction*t2, normal, self.material)

        return no_hit

    def get_uv_coordinates(self, point):

        x = point[0]
        z = point[2]

        u = np.clip((x / (self.radius/20) + 1) / 2, 0, 1)
        v = np.clip((z / (self.radius/2.5) + 1) / 2, 0, 1)

        return u, v # [0, 1]

    def get_normal(self, uv):
        u, v = uv
        u = int(np.floor(u * (self.normal_map.shape[0] - 1)))
        v = int(np.floor(v * (self.normal_map.shape[1] - 1)))

        normal_color = abs(self.normal_map[u, v] / 255.0)

        return normalize(normal_color)

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
        ab = self.vs[0] - self.vs[1]
        ac = self.vs[0] - self.vs[2]
        direction = ray.direction

        A = np.concatenate(([ab], [ac], [direction]), axis=0)
        b = self.vs[0] - ray.origin
        x = np.linalg.solve(A.T, b)
        if x[0] > 0.0 and x[1] > 0.0 and (x[0] + x[1]) < 1.0:
            if ray.start <= x[2] <= ray.end:
                normal = normalize(np.cross(ab, ac))
                return Hit(x[2], ray.origin + direction*x[2], normal, self.material)
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
        self.f = np.linalg.norm(target-eye); # you should set this to the distance from your center of projection to the image plane
        self.M = np.eye(4);  # set this to the matrix that transforms your camera's coordinate system to world coordinates
        # TODO A4 implement this constructor to store whatever you need for ray generation
        self.target = target
        self.up = up
        self.vfov = vfov
        self.height = 2.0*self.f*np.tan(np.radians(vfov)/2.0)
        self.width = aspect*self.height
        self.w = (self.target - self.eye)/self.f # away from viewing direction
        self.u = normalize(np.cross(self.w, self.up)) # right
        self.v = normalize(np.cross(self.w, self.u)) # up

        rotate = np.eye(4)
        rotate[0, :3] = self.u
        rotate[1, :3] = self.v
        rotate[2, :3] = self.w

        transform = np.eye(4)
        transform[:3, 3] = -self.eye

        self.M = rotate@transform


    def generate_ray(self, img_point):
        """Compute the ray corresponding to a point in the image.

        Parameters:
          img_point : (2,) -- a 2D point in [0,1] x [0,1], where (0,0) is the upper left
                      corner of the image and (1,1) is the lower right.
        Return:
          Ray -- The ray corresponding to that image location (not necessarily normalized)
        """
        # TODO A4 implement this function
        x = img_point[0] * 2.0 - 1.0
        y = img_point[1] * 2.0 - 1.0

        d = normalize(self.u * (x * self.width/2.0) + self.v * (y * self.height/2.0) + self.w * self.f)
        return Ray(self.eye, d)

MAX_DEPTH = 4

class PointLight:

    def __init__(self, position, intensity):
        """Create a point light at given position and with given intensity

        Parameters:
          position : (3,) -- 3D point giving the light source location in scene
          intensity : (3,) or float -- RGB or scalar intensity of the source
        """
        self.position = position
        self.intensity = intensity

    def illuminate(self, ray, hit, scene, depth=0):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        # TODO A4 implement this function
        n = hit.normal
        material = hit.material

        r = np.linalg.norm(self.position - hit.point)
        l = normalize(self.position - hit.point)

        irradiance = max(np.dot(n, l), 0.0)
        # lambertian_shading = material.k_d * irradiance/(r**2) * self.intensity
        v = -normalize(ray.direction)
        h = normalize(v+l)
        specular = material.k_s*(np.dot(n, h))**material.p
        specular_light = (material.k_d + specular)*irradiance/(r**2) * self.intensity


        if material.opacity < 1.0:
            color = (1 - material.opacity) * specular_light + material.opacity * scene.bg_color

            refraction_ray = Ray(hit.point + n * 1e-5, normalize(ray.direction))  # Ray continues in the same direction
            new_scene = Scene(scene.surfs[1:], scene.bg_color)
            refraction_hit = new_scene.intersect(refraction_ray)

            if refraction_hit is not no_hit and depth <= MAX_DEPTH:
                # If the refraction ray hits an object behind the transparent object, combine the colors
                behind_color = self.illuminate(refraction_ray, refraction_hit, scene, depth+1)  # Recursive call for the hit object
                color = (1 - material.opacity) * specular_light + material.opacity * behind_color
        else:
        # Fully opaque: just use the computed color
            color = specular_light

        shadow_ray = Ray(hit.point + n * 1e-5, l)
        if scene.intersect(shadow_ray) is no_hit:
            return color
        else:
            return vec([0.0,0.0,0.0])


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

    def __init__(self, surfs, bg_color=vec([0.2,0.3,0.5]), skybox = None):
        """Create a scene containing the given objects.

        Parameters:
          surfs : [Sphere, Triangle] -- list of the surfaces in the scene
          bg_color : (3,) -- RGB color that is seen where no objects appear
        """
        self.surfs = surfs
        self.bg_color = bg_color
        self.skybox = skybox

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and the scene.

        Parameters:
          ray : Ray -- the ray to intersect with the scene
        Return:
          Hit -- the hit data
        """
        # TODO A4 implement this function
        closest_hit = no_hit
        closest_t = np.inf

        for surf in self.surfs:
            hit = surf.intersect(ray)

            if hit and hit.t < closest_t:
                closest_hit = hit
                closest_t = hit.t

        return closest_hit




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

    direct = vec([0.0,0.0,0.0])

    if depth > MAX_DEPTH:
        return vec([0.0,0.0,0.0])
    if hit is no_hit:
        return scene.skybox.get_skybox_color(ray) if scene.skybox else scene.bg_color

    for light in lights:
        direct += light.illuminate(ray, hit, scene)

    n = hit.normal
    v = -normalize(ray.direction)
    r = normalize(2.0 * np.dot(n,v) * n - v)
    reflected = Ray(hit.point+r*1e-6, r)

    new_hit = scene.intersect(reflected)
    mirror = shade(reflected, new_hit, scene, lights, depth+1)
    direct += hit.material.k_m * mirror

    return direct


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
            ray = camera.generate_ray(vec([(j+0.5)/nx, (i+0.5)/ny])) # Generate Ray---we recommend just generating an orthographic ray to start with
            # ray = Ray(vec([1-(2*(i+0.5)/ny), 1-(2*(j+0.5)/nx), 0]), vec([0,0,-1]))
            # intersection = scene.surfs[0].intersect(ray)  # this will return a Hit object
            intersection = scene.intersect(ray)
            output_image[i, j] = shade(ray, intersection, scene, lights)
            # set the output pixel color if an intersection is found
            # if intersection is not no_hit:
            #    output_image[i, j] = vec([1, 1, 1])
            # else:
            #    output_image[i, j] = vec([0,0,0])

    return output_image

class Union:
    def __init__(self, object1, object2):
        self.object1 = object1
        self.object2 = object2

    def intersect(self, ray):
        hit1 = self.object1.intersect(ray)
        hit2 = self.object2.intersect(ray)

        if hit1 is no_hit and hit2 is no_hit:
            return no_hit

        if hit1 is no_hit:
            return hit2
        if hit2 is no_hit:
            return hit1

        if (hit1.t > hit2.t):
            return hit2
        return hit1

class Intersection:

    def __init__(self, object1, object2):
        self.object1 = object1
        self.object2 = object2

    def intersect(self, ray):
        hit1 = self.object1.intersect(ray)
        hit2 = self.object2.intersect(ray)

        if hit1 is no_hit or hit2 is no_hit:
            return no_hit

        if (hit1.t > hit2.t):
            return hit1
        return hit2

class Difference:

    def __init__(self, base, cut):
        self.base = base
        self.cut = cut

    def intersect(self, ray):
        hit1 = self.base.intersect(ray)
        hit2 = self.cut.intersect(ray)

        if hit1 is no_hit:
            return no_hit

        if hit2 is no_hit:
            return hit1

        if hit1.t > hit2.t:
            return no_hit

        return hit1

class Skybox:

    def __init__(self, right, left, up, down, forward, back):
        """
        Initialize the Skybox with textures for each face.

        Parameters:
        - right: Texture for the right face (positive X)
        - left: Texture for the left face (negative X)
        - up: Texture for the up face (positive Y)
        - down: Texture for the down face (negative Y)
        - forward: Texture for the forward face (positive Z)
        - back: Texture for the back face (negative Z)
        """
        self.cube_map = {
            'right': right,
            'left': left,
            'up': up,
            'down': down,
            'forward': forward,
            'back': back
        }

    def get_skybox_color(self, ray):
        """
        Return the color from the skybox based on the ray's direction.

        Parameters:
          ray : Ray -- the ray that hit the surface

        Returns:
          (3,) -- the color seen along this ray
        """

        ray_direction = normalize(ray.direction)

        if abs(ray_direction[0]) > abs(ray_direction[1]) and abs(ray_direction[0]) > abs(ray_direction[2]):
            if ray_direction[0] > 0:
                face = 'right'
                u = -ray_direction[2]
                v = ray_direction[1]
            else:
                face = 'left'
                u = ray_direction[2]
                v = ray_direction[1]
        elif abs(ray_direction[1]) > abs(ray_direction[0]) and abs(ray_direction[1]) > abs(ray_direction[2]):
            if ray_direction[1] > 0:
                face = 'up'
                u = ray_direction[0]
                v = -ray_direction[2]
            else:
                face = 'down'
                u = ray_direction[0]
                v = ray_direction[2]
        else:
            if ray_direction[2] > 0:
                face = 'forward'
                u = ray_direction[0]
                v = ray_direction[1]
            else:
                face = 'back'
                u = -ray_direction[0]
                v = ray_direction[1]

        # Normalize texture coordinates to [0, 1]
        u = (u + 1) / 2
        v = (v + 1) / 2

        # Ensure the coordinates are within bounds [0, 1]
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)

        # Get the texture for the selected face
        texture = np.array(self.cube_map[face])

        # Calculate texture coordinates
        # tex_x = int(u * texture.shape[1])  # Width of texture
        # tex_y = int(v * texture.shape[0])  # Height of texture
        x, y = texture.shape[:2]
        x = int(u*(y-1))
        y = int(v*(x-1))

        # Return the color from the texture at the calculated coordinates
        return texture[y, x]