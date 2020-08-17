import numpy as np
import trimesh
import sys
import os

sys.path.append(os.getcwd())

all_random_colors = (np.random.rand(1000+1, 4)*255).astype(int)
all_random_colors[:, 3] = 255

def build_colored_pointcloud(pc, classes, random=True):
    """
    Builds a trimesh.points.PointCloud with colors according to classes.
    pc has shape (N,3), whereas classes has shape (N,)
    """

    #Create array with random colors
    if random:
        random_colors = (np.random.rand(classes.max()+1, 4)*255).astype(int)
        random_colors[:, 3] = 255
    else:
        random_colors = all_random_colors[0:classes.max()+1, :]

    #One hot encoding for points' classes
    one_hot_classes = np.zeros((classes.shape[0], classes.max()+1)).astype(int)
    one_hot_classes[np.arange(classes.shape[0]), classes] = 1

    #Assign colors
    pc_colors = np.dot(one_hot_classes, random_colors)

    return trimesh.points.PointCloud(vertices=pc, colors=pc_colors)


if __name__ == '__main__':
    print("Usage: python3 viewsegbatch.py <xyzbatchpoints.npy> <classbatchpoints.npy>")
    pcs = np.load(sys.argv[1])
    preds = np.load(sys.argv[2])
    sqsize = int(np.ceil(np.sqrt(pcs.shape[0])))

    colored_pcs = []

    #Go over batch
    nshapes = 0
    stop=False
    for i in range(sqsize):
        for j in range(sqsize):
            #Build colored point cloud
            idx = sqsize*i + j
            cpc = build_colored_pointcloud(pcs[idx, 0:3, :].T, preds[idx, :], random=False)

            #Displace pcs on grid
            cpc.vertices[:, 0] += i*2.0
            cpc.vertices[:, 2] += j*2.0
            colored_pcs.append(cpc)

            nshapes+=1
            #Enough shapes? Finish it!
            if nshapes >= pcs.shape[0]:
                stop = True
                break
        if stop:
            break

    trimesh.Scene(colored_pcs).show()
