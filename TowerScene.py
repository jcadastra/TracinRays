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
    tan = ray.Material(vec([225/255, 204/255, 179/255]), 0.6)
    tri = ray.Material(vec([151/255, 152/255, 144/255]), k_s=0.5, k_m=0.4, opacity=0.5)
    #gray = ray.Material(vec([0.2, 0.2, 0.2]))
    #tan = ray.Material(vec([0.4, 0.4, 0.2]), k_s=0.3, p=90, k_m=0.3)
    gray = ray.Material(vec([0.2, 0.2, 0.2]), k_m=0.4)

    g1 = ray.Sphere(vec([0, -46.0010, 0.6]), 45, gray)
    g2 = ray.Sphere(vec([0, -45.5, 0.6]), 44.5, tan)
    c1 = ray.Cylinder(vec([0, -1, 0.6]), 0.13, 0.6, tan) # bottom cylinder
    c2 = ray.Cylinder(vec([-0.1, -0.4, 0.7]), 0.06, 1.5, tan) # middle left
    c3 = ray.Cylinder(vec([0.1, -0.4, 0.7]), 0.06, 1.5, tan) # middle right
    c4 = ray.Cylinder(vec([0, -0.4, 0.5]), 0.06, 1.5, tan) # middle middle
    c5 = ray.Cylinder(vec([0, 1.1, 0.6]), 0.06, 0.6, tan) # top
    c6 = ray.Cylinder(vec([0, 1.7, 0.6]), 0.04, 0.3, tan)
    cone_1 = ray.Cone(vec([0, 2.5, 0.6]), 0.05, 0.6, tan)
    ground = ray.Union(g1, g2)
    big_sphere = ray.Sphere(vec([0, -0.4, 0.6]), 0.3, tri)
    small_sphere = ray.Sphere(vec([0, 1.1, 0.6]), 0.25, tri)
    smaller_sphere = ray.Sphere(vec([0, 1.7, 0.6]), 0.1, tri)
    tiny_sphere1 = ray.Sphere(vec([0, 0.15, 0.6]), 0.07, tri)
    tiny_sphere2 = ray.Sphere(vec([0, 0.55, 0.6]), 0.07, tri)
    biggest_sphere = ray.Sphere(vec([1.5, -0.4, 2]), 0.8, tri)

    vs_list_1 = 0.4 * read_obj_triangles(open("rectangular_1.obj"))
    vs_list_2 = 0.3 * read_obj_triangles(open("rectangular_2.obj"))
    vs_list_3 = 0.4 * read_obj_triangles(open("rectangular_3.obj"))
    rec_1 = [ray.Triangle(vs, tan) for vs in vs_list_1]
    rec_1_union = rec_1[0]
    for tri in rec_1:
        rec_1_union = ray.Union(rec_1_union, tri)
    rec_2 = [ray.Triangle(vs, tan) for vs in vs_list_2]
    rec_2_union = rec_2[0]
    for tri in rec_2:
        rec_2_union = ray.Union(rec_2_union, tri)
    rec_3 = [ray.Triangle(vs, tan) for vs in vs_list_3]
    rec_3_union = rec_3[0]
    for tri in rec_3:
        rec_3_union = ray.Union(rec_3_union, tri)
    tower = ray.Union(big_sphere, c1)
    tower = ray.Union(tower, c2)
    tower = ray.Union(tower, c3)
    tower = ray.Union(tower, c4)
    tower = ray.Union(tower, c5)
    tower = ray.Union(tower, c6)
    tower = ray.Union(tower, cone_1)
    tower = ray.Union(tower, small_sphere)
    tower = ray.Union(tower, smaller_sphere)
    tower = ray.Union(tower, tiny_sphere1)
    tower = ray.Union(tower, tiny_sphere2)
    main_scene = ray.Union(ground, tower)
    main_scene = ray.Union(main_scene, biggest_sphere)
    main_scene = ray.Union(main_scene, rec_1_union)
    main_scene = ray.Union(main_scene, rec_2_union)
    main_scene = ray.Union(main_scene, rec_3_union)

    scene = ray.Scene([
        #ground,
        # c1,
        # c2,
        # c3,
        # c4,
        # c5,
        # c6,
        # cone_1,
        # big_sphere,
        # small_sphere,
        # smaller_sphere,
        # tiny_sphere1,
        # tiny_sphere2,
        #tower,
        #biggest_sphere,
        main_scene
    ], bg_color=vec([0,0,0]))

    lights = [
        # ray.PointLight(vec([-12, 10, 5]), vec([300, 300, 300])),
        ray.PointLight(vec([12, 10, 5]), vec([235, 48, 133])),
        # ray.PointLight(vec([0, 0.35, 0.6]), vec([52, 249, 56])),
        ray.PointLight(vec([-12, 10, 5]), vec([52, 249, 56])),
        ray.PointLight(vec([0, 10, 5]), vec([52, 237, 249])),
        ray.AmbientLight(0.1),
    ]

    camera = ray.Camera(vec([0, 0, 11.7]), target=vec([0, 0, 0]), vfov=25, aspect=1 / 1)
    return ExampleSceneDef(camera=camera, scene=scene, lights=lights);
