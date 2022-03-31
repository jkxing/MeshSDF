#!/usr/bin/env python3

import sys
sys.path.append("../")
import argparse
from ast import Num
import json
import os
import random
import time
from cv2 import VideoWriter
import torch
import numpy as np

import math
import lib
from lib.workspace import *
from lib.models.decoder import *
from lib.utils import *
from lib.mesh import *

#import neural_renderer as nr
import nvdiffrast.torch as dr
import pdb


import os
import sys
import torch
import pytorch3d
import argparse
import matplotlib.pyplot as plt
from pytorch3d.utils import ico_sphere
import numpy as np
import cv2
from tqdm import tqdm
from pytorch3d.io import load_objs_as_meshes, save_obj

from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras, 
    camera_position_from_spherical_angles,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRendererWithFragments,
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex
)
from pytorch3d.transforms import axis_angle_to_matrix

from PointRenderer.LossFunction import PointLossFunction2
from PointRenderer.NvDiffRastRenderer import NVDiffRastFullRenderer
from PointRenderer.Logger import Logger
import glm
AZIMUTH = 45
ELEVATION = 30
CAMERA_DISTANCE = 2.5


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Demo optimization"
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=1000,
        help="The number of latent code optimization iterations to perform.",
    )
    arg_parser.add_argument(
        "--resolution",
        dest="resolution",
        default=64,
        help="Marching cubes resolution for reconstructed surfaces.",
    )
    arg_parser.add_argument(
        "--image_resolution",
        dest="image_resolution",
        type=int,
        default=256,
        help="Image resolution for differentiable rendering.",
    )
    
    arg_parser.add_argument(
        "--matcher",
        dest="matcher",
        type=str,
        default="bipart",
        help="matcher(bipart/spot)",
    )

    arg_parser.add_argument("--fast", default=False, action="store_true" , help="Run faster iso-surface extraction algorithm presented in main paper.")
    args = arg_parser.parse_args()

    specs_filename = os.path.join(args.experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    latent_size = specs["CodeLength"]

    decoder = DeepSDF(latent_size, **specs["NetworkSpecs"])
    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, model_params_subdir, "latest.pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.module.cuda()

    optimization_meshes_dir = os.path.join(
        args.experiment_directory, optimizations_subdir
    )

    if not os.path.isdir(optimization_meshes_dir):
        os.makedirs(optimization_meshes_dir)

    reconstruction_codes_dir = os.path.join(
        args.experiment_directory, latent_codes_subdir
    )
    latent_filename = os.path.join(
        reconstruction_codes_dir, "latest.pth"
    )
    latent = torch.load(latent_filename)["latent_codes"]["weight"]

    latent_init = latent[1]
    latent_init.requires_grad = True
    latent_target = latent[0]

    # select view point
    

    verts_target, faces_target, _ , _ = lib.mesh.create_mesh(decoder, latent_target, N=args.resolution, output_mesh = True)
    verts_dr = torch.tensor(verts_target[None, :, :].copy(), dtype=torch.float32, requires_grad = False).cuda()  # 
    faces_dr = torch.tensor(faces_target[None, :, :].copy()).cuda()
    textures_dr = 0.7*torch.ones(faces_dr.shape[1], 1, 1, 1, 3, dtype=torch.float32).cuda()
    textures_dr = textures_dr.unsqueeze(0)

    device="cuda"
    def build_mesh(pos,pos_idx):
        mesh = Meshes(verts = pos, faces = pos_idx)
        verts_tex = torch.full([1, pos.shape[1], 3], 0.5, device=device)
        mesh.textures = TexturesVertex(verts_features=verts_tex) 
        return [{"model":mesh}]
    def build_sensors(dist, elev, azim):
        scene={}
        center = [0,0,0]
        width,height=128,128
        fov = 60
        znear = 0.1
        zfar = 1000.0
        perspective = torch.tensor(np.array(glm.perspective(glm.radians(fov), 1.0, znear, zfar)),device=device)
        position = torch.tensor(center,device = device)+camera_position_from_spherical_angles(distance=dist, elevation=elev, azimuth=azim, device = device)
        center=torch.tensor(center,device=device)
        camera_matrix = np.array([np.array(glm.lookAt(position[i].tolist(), center.tolist(), [0,-1,0])) for i in range(num_views)])
        camera_matrix = torch.tensor(camera_matrix).to(device)
        vp = torch.matmul(perspective, camera_matrix).to(device)
        sensors=[{"position":position,"resolution":(width,height),"matrix":vp,"camera_matrix":camera_matrix,"perspective_matrix":perspective}]
        return sensors

    num_views=5
    dist = 2.5
    elev = torch.linspace(0, 360, num_views)
    azim = torch.linspace(-180, 360, num_views)
    Logger.init(exp_name="MeshSDF") 
    renderer = NVDiffRastFullRenderer(device=device,simple=True)
    loss_function = PointLossFunction2(debug=False, resolution=128, matcher="Sinkhorn", device=device, renderer=renderer, num_views=num_views, matching_interval=0, logger=Logger)
    def show_img(img, stop=False, title=""):
        img = img.detach().cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("show_img"+title,img)
        if stop:
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)
        
    verts_dr-=0.3
    scene={}
    scene["sensors"]=build_sensors(dist, elev, azim)
    scene["meshes"]=build_mesh(verts_dr,faces_dr)
    gt = renderer.render(scene)
    verts, faces, samples, next_indices = lib.mesh.create_mesh(decoder, latent_init, N=args.resolution, output_mesh = True)
    verts_dr = torch.tensor(verts[None, :, :].copy(), dtype=torch.float32, requires_grad = False).cuda()
    faces_dr = torch.tensor(faces[None, :, :].copy()).cuda()
    textures_dr = 0.7*torch.ones(faces_dr.shape[1], 1, 1, 1, 3, dtype=torch.float32).cuda()
    textures_dr = textures_dr.unsqueeze(0)
    image_filename = os.path.join(optimization_meshes_dir, "initialization.png")
    #store_image(image_filename, images_out, alpha_out)
    lr= 1e-1
    regl2 = 1000
    decreased_by = 1.5
    adjust_lr_every = 500
    optimizer = torch.optim.Adam([latent_init], lr=lr)
    
    print("Starting optimization:")
    decoder.eval()
    best_loss = None
    sigma = None
    images = []

    for e in range(args.iterations):
        optimizer.zero_grad()
        # first extract iso-surface
        if args.fast:
            verts, faces, samples, next_indices = lib.mesh.create_mesh_optim_fast(samples, next_indices, decoder, latent_init, N=args.resolution)
        else:
            verts, faces, samples, next_indices = lib.mesh.create_mesh(decoder, latent_init, N=args.resolution, output_mesh = True)

        # now assemble loss function
        xyz_upstream = torch.tensor(verts.astype(float), requires_grad = True, dtype=torch.float32, device=torch.device('cuda:0'))
        faces_upstream = torch.tensor(faces.astype(float), requires_grad = False, dtype=torch.float32, device=torch.device('cuda:0'))

        """
        Differentiable Rendering back-propagating to mesh vertices
        """
        textures_dr = 0.7*torch.ones(faces_upstream.shape[0], 1, 1, 1, 3, dtype=torch.float32).cuda()
        scene["meshes"] = build_mesh(xyz_upstream.unsqueeze(0), faces_upstream.unsqueeze(0).int())
        selected_view = random.randint(0,num_views-1)
        if True:
            loss, render_res = loss_function(gt, iteration=e, scene = scene, view=selected_view)
        else:
            render_res = renderer.render(scene, view=selected_view)
            loss = torch.mean((render_res["images"][0]-gt["images"][selected_view])**2)
        
        show_img(gt["images"][selected_view],title="gt")
        show_img(render_res["images"][0],title="render")
        print("Loss at iter {}:".format(e) + ": {}".format(loss.detach().cpu().numpy()))
        
        # now store upstream gradients
        loss.backward()
        dL_dx_i = xyz_upstream.grad

        # use vertices to compute full backward pass
        optimizer.zero_grad()
        xyz = torch.tensor(verts.astype(float), requires_grad = True,dtype=torch.float32, device=torch.device('cuda:0'))
        latent_inputs = latent_init.expand(xyz.shape[0], -1)

        #first compute normals
        pred_sdf = decoder(latent_inputs, xyz)
        loss_normals = torch.sum(pred_sdf)
        loss_normals.backward(retain_graph = True)
        # normalization to take into account for the fact sdf is not perfect...
        normals = xyz.grad/torch.norm(xyz.grad, 2, 1).unsqueeze(-1)
        # now assemble inflow derivative
        optimizer.zero_grad()
        dL_ds_i = -torch.matmul(dL_dx_i.unsqueeze(1), normals.unsqueeze(-1)).squeeze(-1)
        # refer to Equation (4) in the main paper
        loss_backward = torch.sum(dL_ds_i * pred_sdf)
        loss_backward.backward()
        # and update params
        optimizer.step()

        # to visualize gradients first interpolate them on face centroids
        verts_dr = torch.tensor(verts[None, :, :].copy(), dtype=torch.float32, requires_grad = False).cuda()  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
        faces_dr = torch.tensor(faces[None, :, :].copy()).cuda()
        field_faces = interpolate_on_faces(dL_ds_i, faces_dr).squeeze(1)
        # now pick a meaningful normalization, here 30% of initial grad magnitude
        if sigma is None:
            sigma = 0.3*torch.max(torch.abs(field_faces)).cpu().numpy()
        field_min = -sigma
        field_max = sigma
        field_faces = torch.clamp((field_faces-field_min)/(field_max-field_min),0,1)
        textures_dr = torch.ones(faces_dr.shape[1], 1, 1, 1, 3, dtype=torch.float32).cuda()
        # hand crafted color map
        textures_dr[:,0,0,0,0] = field_faces
        textures_dr[:,0,0,0,1] = 1.0-field_faces
        textures_dr[:,0,0,0,2] = 0.7
        textures_dr = textures_dr.unsqueeze(0)

    print("Optimization completed, storing GIF...")
    gif_filename = os.path.join(optimization_meshes_dir, "movie.gif")
    #imageio.mimsave(gif_filename, images)
    print("Done.")
