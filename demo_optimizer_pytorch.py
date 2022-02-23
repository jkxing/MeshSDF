#!/usr/bin/env python3

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

import sys
sys.path.append("../PointRenderer")
sys.path.append("../DenseMatching")
from LossFunction import PointLossFunction
from Logger import Logger

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
    azimuth = AZIMUTH
    elevation = ELEVATION
    camera_distance = CAMERA_DISTANCE
    intrinsic, extrinsic = get_projection(azimuth, elevation, camera_distance, img_w=args.image_resolution, img_h=args.image_resolution)
    # set up renderer
    K_cuda = torch.tensor(intrinsic[np.newaxis, :, :].copy()).float().cuda().unsqueeze(0)
    R_cuda = torch.tensor(extrinsic[np.newaxis, 0:3, 0:3].copy()).float().cuda().unsqueeze(0)
    t_cuda = torch.tensor(extrinsic[np.newaxis, np.newaxis, 0:3, 3].copy()).float().cuda().unsqueeze(0)

    
    glctx = dr.RasterizeGLContext()
    #renderer = nr.Renderer(image_size = args.image_resolution, orig_size = args.image_resolution, K=K_cuda, R=R_cuda, t=t_cuda, anti_aliasing=False)
    verts_target, faces_target, _ , _ = lib.mesh.create_mesh(decoder, latent_target, N=args.resolution, output_mesh = True)

    # visualize target stuff
    verts_dr = torch.tensor(verts_target[None, :, :].copy(), dtype=torch.float32, requires_grad = False).cuda()  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    faces_dr = torch.tensor(faces_target[None, :, :].copy()).cuda()
    textures_dr = 0.7*torch.ones(faces_dr.shape[1], 1, 1, 1, 3, dtype=torch.float32).cuda()
    textures_dr = textures_dr.unsqueeze(0)
    image_filename = os.path.join(optimization_meshes_dir, "target.png")
    if not os.path.exists(os.path.dirname(image_filename)):
        os.makedirs(os.path.dirname(image_filename))
    num_views=1
    device="cuda"
    elev = [30.0]
    azim = [45.0]
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    R, T = look_at_view_transform(dist=2.5, elev=elev, azim=azim)
    cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
    camera = OpenGLPerspectiveCameras(device=device, R=R[None, 0, ...], 
                                    T=T[None, 0, ...]) 
    resolution = args.image_resolution
    raster_settings = RasterizationSettings(
        image_size=resolution, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
        perspective_correct=False,
    )
    renderer_hard = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=camera,
            lights=lights
        )
    )
    
    sigma = 1e-4
    raster_settings_silhouette = RasterizationSettings(
        image_size=resolution, 
        blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
        faces_per_pixel=50, 
    )

    # Silhouette renderer 
    renderer_soft = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings_silhouette
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=camera,
            lights=lights
        )
    ) 

    def renderer(pos, pos_idx, textures, camera, render):
        mesh = Meshes(verts = pos, faces = pos_idx)
        verts_tex = torch.full([1, pos.shape[1], 3], 0.5, device=device)
        mesh.textures = TexturesVertex(verts_features=verts_tex) 
        meshes = mesh.extend(num_views)
        target_images, image_fragments = render(meshes, cameras=camera, lights=lights)
        color_img = target_images[0].detach().cpu().numpy()
        Logger.log("meshsdf_pytorch",content = color_img)
        Logger.step(False)
        return mesh, target_images, image_fragments
        silhouette_images = renderer_silhouette(meshes, cameras=cameras, lights=lights)
        haspos, uvw, tri_id, render_pos = point_renderer.render_point_pytorch(image_fragments, mesh)
        render_pos = point_renderer.point_renderer_pytorch(tri_id, uvw, image_fragments)
        render_rgb = target_images[haspos>0]
        print(render_pos.shape,render_rgb.shape)
        return target_images[0:1],target_images[0:1,...,3],target_images[0:1,...,:3]

    tgt_mesh, tgt_images, tgt_image_fragments = renderer(verts_dr, faces_dr, textures_dr, camera, renderer_soft)

    lr= 1e-2
    regl2 = 1000
    decreased_by = 1.5
    adjust_lr_every = 500
    optimizer = torch.optim.Adam([latent_init], lr=lr)
    
    import win32api
    def my_exit(sig, func=None):
        Logger.exit()
        print("video saved")
    win32api.SetConsoleCtrlHandler(my_exit, 1)
    
    print("Starting optimization:")
    decoder.eval()
    best_loss = None
    sigma = None
    images = []
    loss_function = PointLossFunction(num_views=1, res = resolution, matcher=args.matcher, vis=False)
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
        mesh, images_out, image_fragments = renderer(xyz_upstream.unsqueeze(0), faces_upstream.unsqueeze(0).int(), textures_dr.unsqueeze(0), camera, renderer_soft)
        
        loss =torch.mean((images_out[...,3] - tgt_images[...,3])**2)
        #loss = loss_function(tgt_images[0], images_out, image_fragments, mesh, 0, tgt_mesh, tgt_image_fragments)
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
