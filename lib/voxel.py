import numpy as np
import lib.eval


def evaluate_voxel_prediction(preds, ground_truth, thresh):
    # preds.shape: (batch_size, 32, 32, 32, 2)
    # preds is log prob. convert to prob first
    probs = lib.eval.softmax(preds)
    occupy = probs[:, :, :, :, 1] >= thresh
    # diff = np.sum(np.logical_xor(occupy, ground_truth))
    intersection = np.sum(np.logical_and(occupy, ground_truth), axis=(1, 2, 3))
    union = np.sum(np.logical_or(occupy, ground_truth), axis=(1, 2, 3))
    iou = intersection / union
    iou = np.mean(iou)
    return iou

    # n_false_pos = np.sum(np.logical_and(occupy, ground_truth[:, 0, :, :]))  # false positive
    # n_false_neg = np.sum(np.logical_and(np.logical_not(occupy), ground_truth[:, 1, :, :]))  # false negative
    # return np.array([iou, diff, intersection, union, n_false_pos, n_false_neg])

def voxel2mesh(voxels):
    cube_verts = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0],
                  [1, 1, 1]]  # 8 points

    cube_faces = [[0, 1, 2], [1, 3, 2], [2, 3, 6], [3, 7, 6], [0, 2, 6], [0, 6, 4], [0, 5, 1],
                  [0, 4, 5], [6, 7, 5], [6, 5, 4], [1, 7, 3], [1, 5, 7]]  # 12 face

    cube_verts = np.array(cube_verts)
    cube_faces = np.array(cube_faces) + 1

    l, m, n = voxels.shape

    scale = 0.01
    cube_dist_scale = 1.1
    verts = []
    faces = []
    curr_vert = 0
    for i in range(l):
        for j in range(m):
            for k in range(n):
                # If there is a non-empty voxel
                if voxels[i, j, k] > 0:
                    verts.extend(scale * (cube_verts + cube_dist_scale * np.array([[i, j, k]])))
                    faces.extend(cube_faces + curr_vert)
                    curr_vert += len(cube_verts)

    return np.array(verts), np.array(faces)


def write_obj(filename, verts, faces):
    """ write the verts and faces on file."""
    with open(filename, 'w') as f:
        # write vertices
        f.write('g\n# %d vertex\n' % len(verts))
        for vert in verts:
            f.write('v %f %f %f\n' % tuple(vert))

        # write faces
        f.write('# %d faces\n' % len(faces))
        for face in faces:
            f.write('f %d %d %d\n' % tuple(face))


def voxel2obj(filename, pred):
    verts, faces = voxel2mesh(pred)
    write_obj(filename, verts, faces)
