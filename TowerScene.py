import ray
from ImLite import *
from utils import *
import importlib

class ExampleSceneDef(object):
    def __init__(self, camera, scene, lights):
        self.camera = camera;
        self.scene = scene;
        self.lights = lights;

    def render(self, output_path=None, output_shape=None, gamma_correct=True, srgb_whitepoint=None):
        importlib.reload(ray)
        if(output_shape is None):
            output_shape=[128,128];
        if(srgb_whitepoint is None):
            srgb_whitepoint = 1.0;
        pix = ray.render_image(self.camera, self.scene, self.lights, output_shape[1], output_shape[0]);
        im = None;
        if(gamma_correct):
            cam_img_ui8 = to_srgb8(pix / srgb_whitepoint)
            im = Image(pixels=cam_img_ui8);
        else:
            im = im = Image(pixels=pix);
        if(output_path is None):
            return im;
        else:
            im.writeToFile(output_path);

def Tower():
    importlib.reload(ray)
    tan = ray.Material(vec([0.7, 0.7, 0.4]), 0.6)
    gray = ray.Material(vec([0.2, 0.2, 0.2]))

    g1 = ray.Sphere(vec([0, -45.5015, 0]), 45, gray)
    g2 = ray.Sphere(vec([0, -45, 0]), 44.5, tan)
    ground = ray.Union(g1, g2)

    scene = ray.Scene([
        # ray.Sphere(vec([-0.5, 0, 0]), 0.5, tan),
        # ray.Sphere(vec([0, -0.5, 1]), 0.5, gray),
        ground
    ])

    lights = [
        ray.PointLight(vec([12, 10, 5]), vec([300, 300, 300])),
        ray.AmbientLight(0.1),
    ]

    camera = ray.Camera(vec([0, 0, 8]), target=vec([0, 0, 0]), vfov=25, aspect=16 / 9)
    return ExampleSceneDef(camera=camera, scene=scene, lights=lights);
