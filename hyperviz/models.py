from copy import deepcopy
import numpy as np
from typing import Union, Tuple, Any
from open3d import geometry as o3d_geometry
from open3d.visualization import rendering as o3d_rendering
from hyperviz.utilities import BoolTrigger, WatchableList


class O3DGeometryWrapper():
    """Wrapper object to allow overwritting/extension of o3d goemetry methods"""
    
    def __init__(self, o3d_geometry_object):
        object.__setattr__(self, '_geometry', o3d_geometry_object)

    def __getattribute__(self, __name: str) -> Any:
        obj = object.__getattribute__(self, '_geometry')
        # give priority to the o3d version of a function/attribute
        if hasattr(obj, __name):
            return getattr(obj, __name)
        # fallback to own version of a function/attribute
        return object.__getattribute__(self, __name)

    def __setattr__(self, __name: str, __value: Any):
        obj = object.__getattribute__(self, '_geometry')
        # assign the value to the o3d geometry if possible
        if hasattr(obj, __name):
            setattr(obj, __name, __value)
        # fallback to the wrapper if o3d geometry does not have attribute
        else:
            object.__setattr__(self, __name, __value)


class O3DBaseModel:
    def __init__(self, name: str, geometry: O3DGeometryWrapper, group: str=None, transparent: bool=False, visible: bool=True):
        """Model from which PointCloudModel and TriangleMeshModel inherit.
        *if you want to keep track of transformation history, do not 
        call O3DBaseModel.geometry.translate or rotate directly,
        instead use the transform function."""

        # run type checks
        if type(self) == O3DBaseModel:
            raise TypeError('BaseModel cannot be called directly, please inherit from it')
        elif not isinstance(geometry, O3DGeometryWrapper):
            raise TypeError('Goemetry has not been wrapped using O3dGeometryWrapper')
        elif geometry.dimension() != 3:
            raise TypeError('Model is not instance of open3d.geometry.Geometry3D')

        # set properties
        self.needs_update = BoolTrigger(True)
        self.delete = False
        self.geometry = geometry
        self.position_locked = False
        self.group = group
        self.transparent = transparent
        self.visible = visible
        self._transformation_matrix = np.eye(4, 4, dtype=np.float64)
        self._cartisian_coordinates = np.zeros((2, 3), dtype=np.float64)  # [0] = Px, Py, Pz, [1] = Rx, Ry, Rz in radians

        # validate name before assigning to read only property name
        if not isinstance(name, (str,)):
            raise TypeError('name must be of type string')
        elif not name:
            raise ValueError('name cannot be empty string')
        else:
            self._name = name
    
    def _cartisian_transform_wrapper(self, position_vector, rotation_vector):
        if not self.position_locked:
            # determine relative movements
            relative_translation = position_vector - self._cartisian_coordinates[0]
            relative_rotation = rotation_vector - self._cartisian_coordinates[1]
            # create transform_matrix
            relative_transform_matrix = np.eye(4, 4)
            relative_transform_matrix[:-1, -1] = relative_translation
            relative_transform_matrix[:-1, :-1] = o3d_geometry.get_rotation_matrix_from_xyz(relative_rotation)
            self._transformation_matrix = np.dot(relative_transform_matrix, self.transformation_matrix)
            self.geometry.transform(relative_transform_matrix)
            # update the model
            self._cartisian_coordinates[0] = position_vector
            self._cartisian_coordinates[1] = rotation_vector
            self.needs_update.true()

    @property
    def cartisian_coordinates(self):
        return self._cartisian_coordinates

    @property
    def transformation_matrix(self):
        return self._transformation_matrix

    def get_positional_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns a models positional data as a
         tuple (cartesian coordinates, transformation matrix)"""
        return self._cartisian_coordinates, self._transformation_matrix

    def set_positional_data(self, positional_data: Tuple[np.ndarray, np.ndarray]):
        """Sets a models positional data from a tuple as returned
        by O3DBaseModel.get_positional_data. This model does not apply 
        the transformation"""
        self._cartisian_coordinates = positional_data[0]
        self._transformation_matrix = positional_data[1]

    def reset_positional_data(self):
        """Convenience function to reset the position data of a model back
        to its initial state."""
        self._cartisian_coordinates = np.zeros((2, 3), dtype=np.float64)
        self._transformation_matrix = np.eye(4, 4, dtype=np.float64)
    
    def asO3Ddict(self):
        """Returns a dictionary compatible with the O3DVisualizer's add_geometry method"""
        this = {
            'name': self.name, 
            'geometry': self.geometry._geometry,
            'material': self._material_record,
            'group': self.group,
            'is_visible': self.visible
        }
        this = {k: v for k, v in this.items() if v is not None}
        return this

    @property
    def geometry(self) -> O3DGeometryWrapper:
        self.needs_update.true()
        return self._geometry

    @geometry.setter
    def geometry(self, geometry: O3DGeometryWrapper):
        try:
            geometry.get_geometry_type()
        except Exception as e:
            raise TypeError('geometry must be of type open3d.geometry.Geometry')
        if not isinstance(O3DGeometryWrapper):
            geometry = O3DGeometryWrapper(geometry)
        geometry.cartisian_transform = self._cartisian_transform_wrapper
        self._geometry = geometry 
        self.needs_update.true()        

    def delete_later(self):
        self.delete = True
        self.needs_update.true()

    @property
    def name(self) -> str:
        return self._name

    @property
    def group(self) -> Union[str, None]:
        return self._group

    @group.setter
    def group(self, group: Union[str, None]):
        if not isinstance(group, (str, type(None))):
            raise TypeError('group must be of type string')
        elif (group is not None) and (not group):
            raise ValueError('group cannot be empty string')
        self._group = group
        self.needs_update.true()

    @property
    def transparent(self) -> bool:
        return self._transparent

    @transparent.setter
    def transparent(self, transparent: bool):
        self._transparent = transparent
        if not self._transparent:
            self._material_record = None
        else:
            self._material_record = o3d_rendering.MaterialRecord()
            self._material_record.shader = 'defaultLitTransparency'
            self._material_record.base_color = [0.467, 0.467, 0.467, 0.5]
            self._material_record.base_roughness = 0.0
            self._material_record.base_reflectance = 0.0
            self._material_record.base_clearcoat = 1.0
            self._material_record.thickness = 1.0
            self._material_record.transmission = 1.0
            self._material_record.absorption_distance = 10
            self._material_record.absorption_color = [0.5, 0.5, 0.5]
        self.needs_update.true()

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, visible: bool):
        if not isinstance(visible, (bool,)):
            raise TypeError('visible must be type bool')
        self._visible = visible
        self.needs_update.true()


class O3DPointCloudModel(O3DBaseModel):

    def __init__(self, name: str, pointcloud=o3d_geometry.PointCloud(), group: str=None, transparent: bool=False, visible: bool=True):
        pointcloud = O3DGeometryWrapper(pointcloud)
        super().__init__(name, pointcloud, group, transparent, visible)

    @property
    def geometry(self) -> O3DGeometryWrapper:
        return self._geometry

    @geometry.setter
    def geometry(self, geometry: O3DGeometryWrapper):
        if geometry.get_geometry_type() != o3d_geometry.Geometry.Type.PointCloud:
            raise TypeError('geometry is not of type open3d.geometry.PointCloud')
        if not isinstance(geometry, O3DGeometryWrapper):
            geometry = O3DGeometryWrapper(geometry)
        geometry.cartisian_transform = self._cartisian_transform_wrapper
        self._geometry = geometry


class O3DTriangleMeshModel(O3DBaseModel):

    def __init__(self, name: str, trianglemesh=o3d_geometry.TriangleMesh(), group: str=None, transparent:bool=False, visible: bool=True):
        trianglemesh = O3DGeometryWrapper(trianglemesh)
        super().__init__(name, trianglemesh, group, transparent, visible)
    
    @property
    def geometry(self) -> O3DGeometryWrapper:
        return self._geometry

    @geometry.setter
    def geometry(self, geometry: O3DGeometryWrapper):
        if geometry.get_geometry_type() != o3d_geometry.Geometry.Type.TriangleMesh:
            raise TypeError('geometry is not of type open3d.geoemtry.TriangleMesh')
        if not isinstance(geometry, O3DGeometryWrapper):
            geometry = O3DGeometryWrapper(geometry)
        geometry.cartisian_transform = self._cartisian_transform_wrapper
        self._geometry = geometry


class O3DTextModel():
    def __init__(self, xyz_point: np.array, text):
        self.needs_update = BoolTrigger(True)
        self.xyz_point = xyz_point
        self.text = text
        self.delete = False

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, value: str):
        self._text = value
        self.needs_update.true()

    @property
    def xyz_point(self) -> np.array:
        return self._xyz_point

    @xyz_point.setter
    def xyz_point(self, value: np.array):
        self._xyz_point = value
        self.needs_update.true()
    
    def delete_later(self):
        self.delete = True
        self.needs_update.true()


class O3DModelList(WatchableList):
    
    def __init__(self):
        super().__init__()
    
    def append(self, model: O3DBaseModel):
        """Custom append method for watchable list
        that validates model meta data before appending"""
        if model in self:
            raise ValueError('Model has already been added to the vizualizer')
        if any([m for m in self if (m.name == model.name)]):
            raise ValueError('Model name is already in use, please change the name before adding the model')
        model.needs_update.true()
        super().append(model)

    def get_model(self, name: str) -> Union[O3DBaseModel, None]:
        results = [m for m in self if m.name == name]
        return results[0] if results else None


if __name__ == '__main__':
    stl = O3DGeometryWrapper(o3d_geometry.TriangleMesh.create_box())