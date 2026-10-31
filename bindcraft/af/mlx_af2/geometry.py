#Vectors, rotations and rigid transforms as components in separate arrays, with the arithmetic and the epsilon clipping of the vendored geometry package (geometry/vector.py, rotation_matrix.py,
#rigid_matrix_vector.py). Indexing applies to every component, so a rigid of shape [N] can be viewed as [N, 1, 1] to broadcast against points of shape [N, heads, points].
from dataclasses import dataclass

import mlx.core as mx

@dataclass(frozen=True)
class Vec3:
    x: mx.array
    y: mx.array
    z: mx.array

    def map(self, function):
        return Vec3(function(self.x), function(self.y), function(self.z))

    def __getitem__(self, index):
        return self.map(lambda component: component[index])

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    def scale(self, factor):
        return Vec3(self.x * factor, self.y * factor, self.z * factor)

    def __truediv__(self, factor):
        return Vec3(self.x / factor, self.y / factor, self.z / factor)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vec3(self.y * other.z - self.z * other.y, self.z * other.x - self.x * other.z, self.x * other.y - self.y * other.x)

    def norm(self, epsilon: float = 1e-6):
        #clipped before the square root, as the JAX version does to keep the gradient finite
        squared = self.dot(self)
        return mx.sqrt(mx.maximum(squared, epsilon ** 2) if epsilon else squared)

    def normalized(self, epsilon: float = 1e-6):
        return self / self.norm(epsilon)

    def to_array(self):
        return mx.stack([self.x, self.y, self.z], axis=-1)

    @classmethod
    def from_array(cls, array):
        return cls(array[..., 0], array[..., 1], array[..., 2])

@dataclass(frozen=True)
class Rot3:
    xx: mx.array
    xy: mx.array
    xz: mx.array
    yx: mx.array
    yy: mx.array
    yz: mx.array
    zx: mx.array
    zy: mx.array
    zz: mx.array

    def components(self):
        return (self.xx, self.xy, self.xz, self.yx, self.yy, self.yz, self.zx, self.zy, self.zz)

    def __getitem__(self, index):
        return Rot3(*[component[index] for component in self.components()])

    def map(self, function):
        return Rot3(*[function(component) for component in self.components()])

    def inverse(self):
        return Rot3(self.xx, self.yx, self.zx, self.xy, self.yy, self.zy, self.xz, self.yz, self.zz)

    def apply_to_point(self, point: Vec3) -> Vec3:
        return Vec3(self.xx * point.x + self.xy * point.y + self.xz * point.z, self.yx * point.x + self.yy * point.y + self.yz * point.z, self.zx * point.x + self.zy * point.y + self.zz * point.z)

    def apply_inverse_to_point(self, point: Vec3) -> Vec3:
        return self.inverse().apply_to_point(point)

    def __matmul__(self, other):
        c0 = self.apply_to_point(Vec3(other.xx, other.yx, other.zx))
        c1 = self.apply_to_point(Vec3(other.xy, other.yy, other.zy))
        c2 = self.apply_to_point(Vec3(other.xz, other.yz, other.zz))
        return Rot3(c0.x, c1.x, c2.x, c0.y, c1.y, c2.y, c0.z, c1.z, c2.z)

    @classmethod
    def identity(cls, shape):
        ones, zeros = mx.ones(shape), mx.zeros(shape)
        return cls(ones, zeros, zeros, zeros, ones, zeros, zeros, zeros, ones)

    @classmethod
    def from_two_vectors(cls, e0: Vec3, e1: Vec3):
        #in the new frame e0 lies on the positive x axis and e1 in the xy plane with positive y
        e0 = e0.normalized()
        e1 = (e1 - e0.scale(e1.dot(e0))).normalized()
        e2 = e0.cross(e1)
        return cls(e0.x, e1.x, e2.x, e0.y, e1.y, e2.y, e0.z, e1.z, e2.z)

    @classmethod
    def from_quaternion(cls, w, x, y, z, normalize: bool = True, epsilon: float = 1e-6):
        if normalize:
            inverse_norm = mx.rsqrt(mx.maximum(w ** 2 + x ** 2 + y ** 2 + z ** 2, epsilon))
            w, x, y, z = w * inverse_norm, x * inverse_norm, y * inverse_norm, z * inverse_norm
        return cls(1 - 2 * (mx.square(y) + mx.square(z)), 2 * (x * y - w * z), 2 * (x * z + w * y), 2 * (x * y + w * z), 1 - 2 * (mx.square(x) + mx.square(z)), 2 * (y * z - w * x),
                   2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (mx.square(x) + mx.square(y)))

    def to_array(self):
        return mx.stack([mx.stack([self.xx, self.xy, self.xz], axis=-1), mx.stack([self.yx, self.yy, self.yz], axis=-1), mx.stack([self.zx, self.zy, self.zz], axis=-1)], axis=-2)

    @classmethod
    def from_array(cls, array):
        return cls(*[array[..., row, column] for row in range(3) for column in range(3)])

@dataclass(frozen=True)
class Rigid3:
    rotation: Rot3
    translation: Vec3

    def __getitem__(self, index):
        return Rigid3(self.rotation[index], self.translation[index])

    def map(self, function):
        return Rigid3(self.rotation.map(function), self.translation.map(function))

    def __matmul__(self, other):
        return Rigid3(self.rotation @ other.rotation, self.apply_to_point(other.translation))

    def inverse(self):
        inverse_rotation = self.rotation.inverse()
        return Rigid3(inverse_rotation, inverse_rotation.apply_to_point(-self.translation))

    def apply_to_point(self, point: Vec3) -> Vec3:
        return self.rotation.apply_to_point(point) + self.translation

    def apply_inverse_to_point(self, point: Vec3) -> Vec3:
        return self.rotation.apply_inverse_to_point(point - self.translation)

    def scale_translation(self, factor):
        return Rigid3(self.rotation, self.translation.scale(factor))

    @classmethod
    def identity(cls, shape):
        return cls(Rot3.identity(shape), Vec3(mx.zeros(shape), mx.zeros(shape), mx.zeros(shape)))

    def to_array(self):
        return mx.concatenate([self.rotation.to_array(), self.translation.to_array()[..., None]], axis=-1)

    @classmethod
    def from_array(cls, array):
        return cls(Rot3.from_array(array[..., :3]), Vec3.from_array(array[..., -1]))

def make_transform_from_reference(a_xyz: Vec3, b_xyz: Vec3, c_xyz: Vec3) -> Rigid3:
    #all_atom_multimer.make_transform_from_reference: the frame of the three backbone atoms N, CA, C (a, b, c)
    return Rigid3(Rot3.from_two_vectors(c_xyz - b_xyz, a_xyz - b_xyz), b_xyz)
