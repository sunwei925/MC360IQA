import numpy as np
from scipy import ndimage

def gen_xyz(fov, u, v, out_h, out_w):
    out = np.ones((out_h, out_w, 3), np.float32)

    x_rng = np.linspace(-np.tan(fov / 2), np.tan(fov / 2), num=out_w, dtype=np.float32)
    y_rng = np.linspace(-np.tan(fov / 2), np.tan(fov / 2), num=out_h, dtype=np.float32)

    out[:, :, :2] = np.stack(np.meshgrid(x_rng, -y_rng), -1)
    Rx = np.array([[1, 0, 0], [0, np.cos(v), -np.sin(v)], [0, np.sin(v), np.cos(v)]])
    Ry = np.array([[np.cos(u), 0, np.sin(u)], [0, 1, 0], [-np.sin(u), 0, np.cos(u)]])

    R = np.dot(Ry, Rx)
    return out.dot(R.T)

def xyz_to_uv(xyz):
    x, y, z = np.split(xyz, 3, axis=-1)
    u = np.arctan2(x, z)
    c = np.sqrt(x ** 2 + z ** 2)
    v = np.arctan2(y, c)
    return np.concatenate([u, v], axis=-1)

def uv_to_XY(uv, eq_h, eq_w):
    u, v = np.split(uv, 2, axis=-1)
    X = (u / (2 * np.pi) + 0.5) * eq_w - 0.5
    Y = (-v / np.pi + 0.5) * eq_h - 0.5
    return np.concatenate([X, Y], axis=-1)

def eq_to_pers(eqimg, fov, u, v, out_h, out_w):
    xyz = gen_xyz(fov, u, v, out_h, out_w)
    uv  = xyz_to_uv(xyz)

    eq_h, eq_w = eqimg.shape[:2]
    XY = uv_to_XY(uv, eq_h, eq_w)

    X, Y = np.split(XY, 2, axis=-1)
    X = np.reshape(X, (out_h, out_w))
    Y = np.reshape(Y, (out_h, out_w))

    mc0 = ndimage.map_coordinates(eqimg[:, :, 0], [Y, X]) # channel: B
    mc1 = ndimage.map_coordinates(eqimg[:, :, 1], [Y, X]) # channel: G
    mc2 = ndimage.map_coordinates(eqimg[:, :, 2], [Y, X]) # channel: R

    output = np.stack([mc0, mc1, mc2], axis=-1)
    return output