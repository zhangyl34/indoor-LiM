import bpy
import numpy as np
from blender_kitti import add_voxels

# read voxels.
file = "/home/neal/projects/ply_process/blender/occupy.txt"
f = open(file)
line = f.readline().split(' ')
dim_x = int(line[1])
dim_y = int(line[2])
dim_z = int(line[3])
f.close()
voxels = np.loadtxt(file,delimiter=',',dtype=bool,comments='#').reshape(dim_x,dim_y,dim_z)

scene = bpy.context.scene
colors = np.random.randint(0,256,size=(dim_x,dim_y,dim_z,3),dtype=np.uint8)
add_voxels(voxels=voxels, colors=colors, scene=scene)



