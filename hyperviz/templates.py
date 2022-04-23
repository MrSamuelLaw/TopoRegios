import sys
import asyncio
from pathlib import Path
from copy import deepcopy
from typing import List, Union
import numpy as np
from open3d import pipelines as o3d_pipelines
from open3d import utility as o3d_utils
from open3d import geometry as o3d_geometry
from hyperviz import LJ_X8000
from hyperviz.models import O3DBaseModel, O3DTriangleMeshModel, O3DPointCloudModel, O3DTextModel, O3DModelList
from hyperviz.utilities import BoolTrigger, WatchableList, multidim_xor, unique_rename, aprint
from hyperviz.linear_color_mapper import LinearColorMapper, LinearColorMapperItem
from PySide6.QtCore import Qt, QMimeData
from PySide6.QtGui import QClipboard
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QWidget, QMenu
from PySide6.QtWidgets import QGridLayout, QSizePolicy, QSpacerItem, QHBoxLayout, QColorDialog
from PySide6.QtWidgets import QDialog, QListWidget, QListWidgetItem, QPlainTextEdit, QFileDialog
from PySide6.QtWidgets import QPushButton, QSpinBox, QDoubleSpinBox, QLabel, QCheckBox, QComboBox, QRadioButton


class Standard:

    class Button(QPushButton):
        
        def __init__(self, text, parent=None):
            super().__init__(text, parent=parent)
            size_policy = self.sizePolicy()
            size_policy.setHorizontalPolicy(QSizePolicy.Expanding)
            self.setSizePolicy(size_policy)

    class DoubleSpinbox(QDoubleSpinBox):

        def __init__(self, parent=None):
            super().__init__(parent=parent)
            self.setSingleStep(0.5)
            size_policy = self.sizePolicy()
            size_policy.setHorizontalPolicy(QSizePolicy.Expanding)
            self.setSizePolicy(size_policy)

        def on_value_changed(self, value: float):
            if abs(self.maximum() - value) < 2:
                self.setMaximum(value + 10)
            elif abs(self.minimum() - value) < 2:
                self.setMinimum(value - 10)

    class IntSpinbox(QSpinBox):

        def __init__(self, parent=None):
            super().__init__(parent=parent)
            self.setSingleStep(1)
            size_policy = self.sizePolicy()
            size_policy.setHorizontalPolicy(QSizePolicy.Expanding)
            self.setSizePolicy(size_policy)

        def on_value_changed(self, value: float):
            if abs(self.maximum() - value) < 2:
                self.setMaximum(value + 10)
            elif abs(self.minimum() - value) < 2:
                self.setMinimum(value - 10)

    class Label(QLabel):

        def __init__(self, text, parent=None):
            """sizing = -1 for shrinking, 0 for default, and 1 for expanding"""
            super().__init__(text=text, parent=parent)
            policy = self.sizePolicy()
            policy.setHorizontalPolicy(QSizePolicy.Expanding)
            self.setAlignment(Qt.AlignCenter)
        
    class Dialog(QDialog):
        
        def __init__(self, parent=None):
            super().__init__(parent=parent)

    class AsyncDialog(QDialog):

        def __init__(self, parent=None):
            super().__init__(parent=parent)
            self.running = BoolTrigger(True)
            self.canceled = BoolTrigger(False)

        async def run(self):
            """Method to implement non-blocking dialogs"""
            self.show()
            while self.running() and not self.canceled():
                await asyncio.sleep(1E-9)
            self.hide()
            # allows the window to hide and the event 
            # loop to process before proceeding
            await asyncio.sleep(1E-9)
            return not self.canceled()

        def closeEvent(self, event):
            self.canceled.true()


class BMPImportDialog(Standard.AsyncDialog):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle('Import BMP')
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.resize(200, 100)

        layout = QGridLayout(parent=self)
        nrows, ncols = 3, 2
        cur_row = 0

        # set up the offset
        offset_label = Standard.Label('offset')
        offset_label.setAlignment(Qt.AlignRight)
        self.offset_spinbox = Standard.DoubleSpinbox(self)
        self.offset_spinbox.setToolTip('Use this to cut off the "floor" of a ' + 
                                  'bmp import by <x> number of mm')
        layout.addWidget(offset_label, cur_row, 0)
        layout.addWidget(self.offset_spinbox, cur_row, 1)
        cur_row += 1

        # set up the max number of points
        npoints_label = Standard.Label('max points')
        npoints_label.setAlignment(Qt.AlignRight)
        self.npoints_spinbox = Standard.IntSpinbox(self)
        self.npoints_spinbox.setMaximum(100_000_000)
        self.npoints_spinbox.setToolTip('Use this to limit the number of data'
                                   'points imported to <x> number of points')
        layout.addWidget(npoints_label, cur_row, 0)
        layout.addWidget(self.npoints_spinbox, cur_row, 1)
        cur_row += 1

        import_button = Standard.Button('import')
        import_button.setToolTip('Import the Point Cloud')
        import_button.clicked.connect(self.running.false)
        layout.addWidget(import_button, cur_row, 0, 1, ncols)
        cur_row += 1

        spacer = QSpacerItem(40, 40, QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addItem(spacer, cur_row, 0, 1, 2)

    @property
    def offset(self):
        return self.offset_spinbox.value()

    @property
    def npoints(self):
        return self.npoints_spinbox.value()


class TriangleMeshToPointCloudDialog(Standard.AsyncDialog):

    def __init__(self, model_list: O3DModelList, name=None, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle('STL -> Pointcloud')
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.model_list = model_list
        self.setLayout(QGridLayout(parent=self))
        layout: QGridLayout = self.layout()

        currow = 0
        # drop down to select the model
        label = Standard.Label(text='STL', parent=self)
        self.target_drop_down = QComboBox(parent=self)
        self.target_drop_down.setToolTip('Triangle Mesh to convert to a point cloud')
        self.target_drop_down.addItems([m.name for m in model_list if isinstance(m, O3DTriangleMeshModel)])
        if name is not None:
            self.target_drop_down.setCurrentText(name)
        layout.addWidget(label, currow, 0)
        layout.addWidget(self.target_drop_down, currow, 1)
        currow += 1

        # drop or spinbox for npoints
        label = Standard.Label(text='n-points', parent=self)
        layout.addWidget(label, currow, 0, 2, 1)
        self.npoints_spinbox = Standard.IntSpinbox(parent=self)
        self.npoints_spinbox.setToolTip('Manually enter the number of points to create from the mesh ' + 
                                        'or select an existing point cloud use its point count')
        self.npoints_spinbox.setRange(0, 50E+6)
        layout.addWidget(self.npoints_spinbox, currow, 1)
        currow += 1
        self.npoints_drop_down = QComboBox(parent=self)
        self.npoints_drop_down.setEditable(False)
        self.npoints_drop_down.addItem('<query pointcloud>')
        self.npoints_drop_down.addItems([m.name for m in model_list if isinstance(m, O3DPointCloudModel)])
        self.npoints_drop_down.currentTextChanged.connect(self.get_points_from_model)
        layout.addWidget(self.npoints_drop_down, currow, 1)
        currow += 1

        # button to submit the dialog
        self.apply_button = Standard.Button('Convert', parent=self)
        self.apply_button.setToolTip('Create a new Point Cloud from the STL using n points')
        self.apply_button.clicked.connect(self.running.false)
        layout.addWidget(self.apply_button, currow, 0, 1, 2)

    @property
    def target(self) -> Union[O3DTriangleMeshModel, None]:
        name = self.target_drop_down.currentText()
        return self.model_list.get_model(name)

    @property
    def npoints(self) -> int:
        return self.npoints_spinbox.value()

    def get_points_from_model(self, name: str) -> int:
        model = self.model_list.get_model(name)
        if (model is not None) and model.geometry.has_points():
            npoints = len(model.geometry.points)
            self.npoints_spinbox.setValue(npoints)

    async def run(self):
        await aprint('Awaiting user input')
        completed = await super().run()
        if completed:
            await aprint('Converting STL -> PCD')
            target = self.target
            npoints = self.npoints
            point_cloud = target.geometry.sample_points_uniformly(npoints)
            name = unique_rename([m.name for m in self.model_list], target.name + '_pcd', '_pcd')
            point_cloud = O3DPointCloudModel(name, point_cloud)
            self.model_list.append(point_cloud)
            await aprint('STL -> PCD conversion complete')
        else:
            await aprint('STL to pointcloud conversion canceled')
        return completed       


class OutlierRemovalDialog(Standard.AsyncDialog):

    preview_name = '__removal_preview__'
    algoithms = ['Radius Outlier Removal', 'Statistical Outlier Removal']

    def __init__(self, target: O3DPointCloudModel, model_list: O3DModelList):
        super().__init__(parent=None)
        self.setWindowTitle('Outlier Removal')
        self.setWindowFlags(Qt.WindowStaysOnTopHint)

        self._model_list = model_list
        self._target = target
        self._preview_model = O3DPointCloudModel(self.preview_name)
        self._keep_idx: np.array = None
        self._model_list.append(self._preview_model)

        layout = QGridLayout(parent=self)
        self.setLayout(layout)

        currow = 0
        # set up the drop down for the target selector
        label = Standard.Label('Target PCD', parent=self)
        layout.addWidget(label, currow, 0)
        self.target_dropdown = QComboBox(parent=self)
        self.target_dropdown.addItems([m.name for m in model_list if isinstance(m, O3DPointCloudModel)])
        self.target_dropdown.currentTextChanged.connect(self.on_target_change)
        self.target_dropdown.setCurrentText(target.name)
        layout.addWidget(self.target_dropdown, currow, 1)
        
        currow += 1
        # set up the drop down for the algorithm
        label = Standard.Label('Algorithm', parent=self)
        layout.addWidget(label, currow, 0)
        self.algorithm_dropdown = QComboBox(parent=self)
        self.algorithm_dropdown.setToolTip(
            '\n'.join([
            'Radius Outlier Removal:', 
            '   Function to remove points that have less than n number of points in a given sphere of a given radius',
            '   Parameters:',
            '       n: number of points with the radius',
            '       tolerance: radius of the sphere\n',
            'Statistical Outlier Removal:',
            '   Function to remove points that are further away from their neighbors in average',
            '   Parameters: ',
            '       n: number of neighbors around the target point',
            '       tolerance: standard deviation ratio'
        ]))
        self.algorithm_dropdown.addItems(self.algoithms)
        layout.addWidget(self.algorithm_dropdown, currow, 1)
        
        currow += 1
        # set up the n field
        label = Standard.Label('N')
        layout.addWidget(label, currow, 0)
        self.n_spinbox = Standard.IntSpinbox(parent=self)
        self.n_spinbox.setRange(1, 1_000_000)
        layout.addWidget(self.n_spinbox, currow, 1)

        currow += 1
        # set up the tolerance field
        label = Standard.Label('Tolerance')
        layout.addWidget(label, currow, 0)
        self.tolerance_spinbox = Standard.DoubleSpinbox(parent=self)
        self.tolerance_spinbox.setRange(0, 1_000_000)
        self.tolerance_spinbox.setSingleStep(0.05)
        self.tolerance_spinbox.setValue(1.0)
        layout.addWidget(self.tolerance_spinbox, currow, 1)

        currow += 1
        # set up the preview button
        self.preview_button = Standard.Button('Preview', parent=self)
        self.preview_button.setToolTip('Shows a preview with the points to remove in purple and the points to keep in neon yellow')
        self.preview_button.clicked.connect(self.on_preview)
        layout.addWidget(self.preview_button, currow, 0, 1, 2)

        currow += 1
        # set up the apply button
        self.apply_button = Standard.Button('Apply', parent=self)
        self.apply_button.setToolTip('Removes the points from the selected Point Cloud. This change is not reversable')
        self.apply_button.clicked.connect(self.on_apply)
        layout.addWidget(self.apply_button, currow, 0, 1, 2)

    def on_target_change(self, name: str):
        self._target = self._model_list.get_model(name)

    def on_preview(self):
        index = self.algorithm_dropdown.currentIndex()
        n = self.n_spinbox.value()
        tolerance = self.tolerance_spinbox.value()

        if index == 0: # radius outlier
            pointcloud, keep_idx = self._target.geometry.remove_radius_outlier(n, tolerance)
        elif index == 1:  # statistical outlier removal
            pointcloud, keep_idx = self._target.geometry.remove_statistical_outlier(n, tolerance)
        self._keep_idx = keep_idx
        self._preview_model.geometry = deepcopy(self._target.geometry)

        # paint the model with a removal color
        remove_color = np.array([255, 51, 204], dtype=np.float64)/255
        self._preview_model.geometry.paint_uniform_color(remove_color)

        # grab the array of colors that now exists and pain the keep points
        colors = np.asarray(self._preview_model.geometry.colors)
        keep_color = np.array([146, 255, 8], dtype=np.float64)/255
        colors[keep_idx] = keep_color
        
        # set the update flag
        self._preview_model.needs_update.true()

    def closeEvent(self, event=None):
        super().closeEvent(event)
        self._preview_model.delete_later()

    def on_apply(self):
        self.closeEvent()
        self._target.geometry = self._target.geometry.select_by_index(self._keep_idx)
        self._target.needs_update.true()
        self.running.false()


class ComparisonDialog(Standard.AsyncDialog):

    def __init__(self, target: O3DPointCloudModel, model_list: O3DModelList, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle('Measurement Dialog')
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self._model_list = model_list
        
        # define the color map
        self._cmap = LinearColorMapper()
        self._from_rgb = np.array([0, 0, 1], dtype=np.float64)
        self._to_rgb = np.array([1, 0, 0], dtype=np.float64)
        self._cmap_item = LinearColorMapperItem(self._from_rgb, self._to_rgb, 0, 1)
        self._cmap.add_item(self._cmap_item)

        # define the objects necessary to visualize deviations
        self._default_steps = 20
        self._default_color = np.array([206, 206, 206], dtype=np.float64)/255
        self._scan_cloud: O3DPointCloudModel = target
        self._scan_cloud_origninal_colors = deepcopy(self._scan_cloud.geometry.colors)
        self._scan_cloud.geometry.paint_uniform_color(self._default_color)
        self._scan_cloud_new_colors = np.asarray(self._scan_cloud.geometry.colors)
        self._deltas = None

        layout = QGridLayout(parent=self)
        self.setLayout(layout)   

        currow = 0
        # setup the scan drop down
        label = Standard.Label('Scan', parent=self)
        layout.addWidget(label, currow, 0)
        self.scan_dropdown = QComboBox(parent=self)
        self.scan_dropdown.setToolTip('Point Cloud that represents the scan/actual data')
        self.scan_dropdown.addItems([target.name])
        self.scan_dropdown.setCurrentText(target.name)
        layout.addWidget(self.scan_dropdown, currow, 1, 1, 2)

        currow += 1
        # setup the target drop down
        label = Standard.Label('Target', parent=self)
        layout.addWidget(label, currow, 0)
        self.target_dropdown = QComboBox(parent=self)
        self.target_dropdown.setToolTip('Point Cloud that represents the CAD/desired data')
        layout.addWidget(self.target_dropdown, currow, 1, 1, 2)

        currow += 1
        # set up the labels
        label = Standard.Label('Min', parent=self)
        layout.addWidget(label, currow, 0)
        label = Standard.Label('Steps', parent=self)
        layout.addWidget(label, currow, 1)
        label = Standard.Label('Max', parent=self)
        layout.addWidget(label, currow, 2)

        currow += 1
        # set up spinners
        self.min_spinbox = Standard.DoubleSpinbox(parent=self)
        self.min_spinbox.setToolTip('Minimum deviation to paint')
        self.min_spinbox.setSingleStep(0.10)
        layout.addWidget(self.min_spinbox, currow, 0)
        self.steps_spinbox = Standard.IntSpinbox(parent=self)
        self.steps_spinbox.setToolTip('Number of steps to segment the colors into')
        layout.addWidget(self.steps_spinbox, currow, 1)
        self.max_spinbox = Standard.DoubleSpinbox(parent=self)
        self.max_spinbox.setToolTip('Maximum deviation to paint')
        self.max_spinbox.setSingleStep(0.10)
        layout.addWidget(self.max_spinbox, currow, 2)

        # assignt the values to the combobox
        self.target_dropdown.currentTextChanged.connect(self.on_dropdown_changed)
        self.target_dropdown.addItems([m.name for m in self._model_list if isinstance(m, O3DPointCloudModel) and m is not self._scan_cloud])

        # assign default values to spinners
        self._deltas = self.compute_deltas(self._scan_cloud, self._target_cloud)
        self.min_spinbox.setValue(self._deltas.min())
        self.min_spinbox.valueChanged.connect(self.on_value_changed)
        self.max_spinbox.setValue(self._deltas.max())
        self.max_spinbox.valueChanged.connect(self.on_value_changed)
        self.steps_spinbox.setValue(self._default_steps)
        self.steps_spinbox.valueChanged.connect(self.on_value_changed)

    def compute_deltas(self, scan_cloud: O3DPointCloudModel, target_cloud: O3DPointCloudModel):
        unsigned_deltas = scan_cloud.geometry.compute_point_cloud_distance(target_cloud.geometry)
        return np.asarray(unsigned_deltas)

    def compute_colors(self, min_val, max_val, steps):
        """modifes the colors in place using steps between blue and red"""
        print('Computing color map')
        if (self._deltas is not None) and (self._scan_cloud_new_colors is not None):
            self._scan_cloud_new_colors[:] = self._default_color
            self._cmap_item.recompute(min_val, max_val)
            steps = np.linspace(min_val, max_val, steps+1)
            # color everything within the bounds
            for s1, s0 in zip(steps[1:], steps):
                c = self._cmap.get_color(s0)
                idx = np.where((self._deltas >= s0) & (self._deltas <= s1))
                self._scan_cloud_new_colors[idx] = c
            c = self._cmap.get_color(s1)
            self._scan_cloud_new_colors[idx] = c
        self._scan_cloud.needs_update.true()

    def reset_colors(self, cloud):
        """Resets the colors for the pointcloud"""
        self._scan_cloud.geometry.colors = self._scan_cloud_origninal_colors
        self._scan_cloud.needs_update.true()
        
    def on_dropdown_changed(self, text: str):
        # get the new models
        self._target_cloud = self._model_list.get_model(self.target_dropdown.currentText())
        self._deltas = self.compute_deltas(self._scan_cloud, self._target_cloud)        

        # compute the gradiant colors in place
        self.compute_colors(self._deltas.min(), self._deltas.max(), self._default_steps)

        # refresh the spinboxes
        self.min_spinbox.setValue(self._deltas.min())
        self.steps_spinbox.setValue(self._default_steps)
        self.max_spinbox.setValue(self._deltas.max())

    def on_value_changed(self, *args):
        if any([sb.hasFocus() for sb in (self.min_spinbox, self.steps_spinbox, self.max_spinbox)]):
            # grab the data
            min_val = self.min_spinbox.value()
            max_val = self.max_spinbox.value()
            steps = self.steps_spinbox.value()
            self.compute_colors(min_val, max_val, steps)

    async def run(self):
        completed = await super().run()
        self.reset_colors(self._scan_cloud)
        return completed


class P2PRegistrationDialog(Standard.AsyncDialog):

    def __init__(self, target: O3DPointCloudModel, model_list: O3DModelList, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle('Point to Plane Registration Dialog')
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self._target = target
        self._model_list = model_list

        layout = QGridLayout(parent=self)
        self.setLayout(layout)

        currow = 0
        # drop down to select scan
        label = Standard.Label('Scan', parent=self)
        layout.addWidget(label, currow, 0)
        self.scan_dropdown = QComboBox(parent=self)
        self.scan_dropdown.setToolTip('Point Cloud that represents the scan/actual data')
        self.scan_dropdown.addItems([m.name for m in self._model_list if isinstance(m, O3DPointCloudModel)])
        self.scan_dropdown.setCurrentText(self._target.name)
        layout.addWidget(self.scan_dropdown, currow, 1)

        currow += 1
        # dropdown to select target
        label = Standard.Label('Target', parent=self)
        layout.addWidget(label, currow, 0)
        self.target_dropdown = QComboBox(parent=self)
        self.target_dropdown.setToolTip('Point Cloud that represents the CAD/desired data')
        self.target_dropdown.addItems([m.name for m in self._model_list if isinstance(m, O3DPointCloudModel)])
        # self.target_dropdown.setCurrentText(self._target.name)
        layout.addWidget(self.target_dropdown, currow, 1)

        currow += 1
        # spinbox for convergence threshold
        label = Standard.Label('max distance', parent=self)
        layout.addWidget(label, currow, 0)
        self._max_distance_spinbox = Standard.DoubleSpinbox(parent=self)
        self._max_distance_spinbox.setToolTip('The maximum distance that any one point in the scan ' + 
                                              'cloud may be seperated from any one point in the target cloud.')
        self._max_distance_spinbox.setValue(0.1)
        layout.addWidget(self._max_distance_spinbox)

        currow += 1
        self.apply_button = Standard.Button('Apply', parent=self)
        self.apply_button.setToolTip('Align the scan cloud to the target cloud')
        self.apply_button.clicked.connect(self.on_apply)
        layout.addWidget(self.apply_button, currow, 0, 1, 2)

    def on_apply(self):
        asyncio.create_task(self.apply())

    async def apply(self):
        await aprint('Starting registration')
        await asyncio.sleep(1E-9)
        target: O3DPointCloudModel = self._model_list.get_model(self.target_dropdown.currentText())
        scan: O3DPointCloudModel = self._model_list.get_model(self.scan_dropdown.currentText())
        threshold = self._max_distance_spinbox.value()
        transform_init = np.eye(4, 4)
        point_to_plane_registration = o3d_pipelines.registration.registration_icp(
                scan.geometry, target.geometry, threshold, transform_init,
                o3d_pipelines.registration.TransformationEstimationPointToPlane()
            )
        await aprint(f'Correspondece = {point_to_plane_registration.correspondence_set}: correspondences between source and target.')
        await aprint(f'Fitness = {point_to_plane_registration.fitness}: The overlapping area (# of inlier correspondeces/# of points in source). Higher is better.')
        await aprint(f'RMSE = {point_to_plane_registration.inlier_rmse}: RMSE of all inlier correspondences. Lower is better.')
        matrix = point_to_plane_registration.transformation
        rx = np.arctan2(matrix[2, 1], matrix[2, 2])
        ry = np.arctan2(matrix[2, 0], np.sqrt(matrix[2, 1]**2 + matrix[2, 2]**2))
        rz = np.arctan2(matrix[1, 0], matrix[0, 0])
        dx, dy, dz = matrix[:-1, -1]
        pos = scan.cartisian_coordinates[0] + np.array([dx, dy, dz])
        rot = scan.cartisian_coordinates[1] + np.array([rx, ry, rz])
        await aprint(f'Translation = {pos}: x, y, z, translations used to align scan to target.')
        await aprint(f'Rotation = {rot}: x, y, z, euler angles used to align scan to target.')
        scan.geometry.cartisian_transform(pos, rot)
        await aprint('Registration Complete')
    

class CroppingDialog(Standard.AsyncDialog):

    model_name = '__cropping_volumn__'

    def __init__(self, target: O3DPointCloudModel, model_list: O3DModelList, text_list: WatchableList[O3DTextModel], parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle('Cropping Dialog')
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)

        self.bounding_box = None
        self._target = target
        self._model_list = model_list
        self._text_list = text_list
        self._points: np.array = None
        self._labels: List[O3DTextModel] = None

        layout = QGridLayout(parent=self)
        self.setLayout(layout)

        # list widget to change values
        self.list_widget = CroppingDialog.ListWidget(parent=self)
        self.list_widget.setToolTip('Use ctrl+click to select multiple points to move at once')
        for i, row in enumerate(self.list_widget.spinboxes):
            for j, sb in enumerate(row):                    
                sb.valueChanged.connect(self.on_value_changed_wrapper(sb, i, j))

        currow = 0
        # dropdown to select a clipping target
        label = Standard.Label('Crop Target')
        layout.addWidget(label, currow, 0)
        self.dropdown = QComboBox(parent=self)
        self.dropdown.setToolTip('Model to clip using cropping volume')
        self.dropdown.currentTextChanged.connect(self.on_target_changed)
        self.dropdown.addItems([m.name for m in model_list if isinstance(m, O3DPointCloudModel)])
        self.dropdown.setCurrentText(target.name)
        layout.addWidget(self.dropdown, currow, 1)

        currow += 1
        # add the list widget after the clipping target for callback reasons
        layout.addWidget(self.list_widget, currow, 0, 1, 2)

        currow += 1
        # add the checkbox to invert the clip
        self.keep_inside_radiobutton = QRadioButton(parent=self)
        self.keep_inside_radiobutton.setToolTip('Removes the points outside of the cropping volume')
        self.keep_inside_radiobutton.setText('Keep Inside')    
        self.keep_inside_radiobutton.setChecked(True)
        self.keep_outside_radiobutton = QRadioButton(parent=self)
        self.keep_outside_radiobutton.setToolTip('Removes the points inside of the cropping volume')
        self.keep_outside_radiobutton.setText('Keep Outside')
        layout.addWidget(self.keep_inside_radiobutton, currow, 0)
        layout.addWidget(self.keep_outside_radiobutton, currow, 1)
        
        currow += 1
        # add a button to finalize the clip
        self.apply_button = Standard.Button('Apply Crop')
        self.apply_button.setToolTip('Creates a copy of the target points that ' +
                                     'lie within the cropping volume')
        self.apply_button.clicked.connect(self.running.false)
        layout.addWidget(self.apply_button, currow, 0, 1, 2)

    @property
    def target(self) -> O3DPointCloudModel:
        return self._target 
    
    def on_target_changed(self, name: str):
        self._target = self._model_list.get_model(name)
        # compute the dimensions of the box 
        oriented_bounding_box = self._target.geometry.get_oriented_bounding_box()
        max_bound: np.array = oriented_bounding_box.get_max_bound()
        min_mound: np.array = oriented_bounding_box.get_min_bound()
        dimensions = max_bound - min_mound

        # compute the transformation required to align the box starting at 0,0,0 to the 
        # translation
        origin = np.array([0, 0, 0])
        position = min_mound - origin 

        # rotation
        v2 = max_bound - min_mound  # target box
        v1 = dimensions - origin    # vector of box of the same size aligned at origin
        dv = v1 - v2                # dx, dy, dz between the vectors
        rotation = np.array([np.arcsin(d) for d in dv])  # compute the angles between the orthoginal components

        # create the mesh box
        box = o3d_geometry.TriangleMesh.create_box(*dimensions)

        # create the bounding box if it does not exist
        model = self._model_list.get_model(self.model_name)
        if model is not None:
            model.reset_positional_data()
            model.geometry = box
        else:
            model = O3DTriangleMeshModel(self.model_name, box)
            model.transparent = True
            self._model_list.append(model)
       
        # apply the transformation
        model.geometry.cartisian_transform(position, rotation)
        
        # update the bounding box with the point labels
        self._points = np.asarray(model.geometry.vertices)

        # update the labels from the points
        if self._labels is None:
            self._labels = [O3DTextModel(p, f'P{i+1}') for i, p in enumerate(self._points)]
            self._text_list.extend(self._labels)
        else:
            for l, p in zip(self._labels, self._points):
                l.xyz_point = p
        self.write_to_spinboxes(self._points)

    def on_value_changed_wrapper(self, spinbox: Standard.DoubleSpinbox, row: int, col: int):
        return lambda val: self.on_value_changed(spinbox, row, col, val)

    def on_value_changed(self, spinbox: Standard.DoubleSpinbox , row: int, col: int, value: float):
        if spinbox.hasFocus():
            self._points[row, col] = value
            # update other rows that might be selected
            items = self.list_widget.selectedItems()
            irows = [self.list_widget.row(item) for item in items if self.list_widget.row(item) != row]
            for i in irows:
                sb_row = self.list_widget.spinboxes[i]
                sb_row[col].setValue(value)
                self._points[i, col] = value
            # update the model and query the bounding box
            model = self._model_list.get_model(self.model_name)
            model.needs_update.true()
            self.bounding_box = model.geometry.get_oriented_bounding_box()
            [t.needs_update.true() for t in self._text_list]
            
    def write_to_spinboxes(self, array: np.array):
        for i, row in enumerate(self.list_widget.spinboxes):
            [sb.setValue(v) for sb, v in zip(row, self._points[i])]

    def teardown(self):
        [label.delete_later() for label in self._labels]
        model = self._model_list.get_model(self.model_name)
        if model is not None:
            model.delete_later()

    def closeEvent(self, event=None):
        super().closeEvent(event)
        self.teardown()

    async def run(self):
        completed = await super().run()
        self.teardown()
        if completed:
            await aprint('Beginning Crop')
            # apply the crop
            target = self.target
            bounding_box = self.bounding_box
            name = unique_rename([m.name for m in self._model_list], target.name, '_crop')
            crop = O3DPointCloudModel(name, target.geometry.crop(bounding_box))
            crop.set_positional_data((np.copy(target.cartisian_coordinates), np.copy(target.transformation_matrix)))
            if crop.geometry.has_points() and self.keep_outside_radiobutton.isChecked():
                    # invert the crop if it has any points. In other words, delete the crop, keep the rest
                    idx, *_ = multidim_xor(np.asarray(target.geometry.points), np.asarray(crop.geometry.points))
                    crop.geometry = target.geometry.select_by_index(idx)
            # add the crop to the model list
            self._model_list.append(crop)
            await aprint('Crop completed')
        else:
            await aprint('Crop cancled')
        return completed


    class ListWidget(QListWidget):

        def __init__(self, parent=None):
            super().__init__(parent=parent)
            self.setSelectionMode(QListWidget.ExtendedSelection)

            self.spinboxes: List[List[Standard.DoubleSpinbox]] = []
            for i in range(8):
                item = QListWidgetItem()
                content = CroppingDialog.ListWidget.ListItemWidget(f'P{i+1}')
                self.spinboxes.append(content.spinboxes)
                item.setSizeHint(content.minimumSizeHint())
                self.addItem(item)
                self.setItemWidget(item, content)
            
        class ListItemWidget(QWidget):
            
            def __init__(self, text, parent=None):
                super().__init__(parent=parent) 
                self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

                layout = QHBoxLayout(parent=self)
                layout.setContentsMargins(0, 0, 0, 0)
                self.setLayout(layout)

                label = Standard.Label(text, parent=self)
                layout.addWidget(label)

                self.spinboxes = [Standard.DoubleSpinbox(parent=self) for i in range(3)]
                [sb.setFocusPolicy(Qt.FocusPolicy.StrongFocus) for sb in self.spinboxes]
                [sb.setRange(-10_000_000, 10_000_000) for sb in self.spinboxes]
                [sb.setAlignment(Qt.AlignCenter) for sb in self.spinboxes]
                [layout.addWidget(sb) for sb in self.spinboxes] 


class SidePanel(QWidget):
    
    def __init__(self, model_list: O3DModelList, text_list_ref: WatchableList):
        """
        window is of type O3DVisualizer
        """
        super().__init__(parent=None)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowTitle('SidePanel')
        self.model_list = model_list
        self.text_list = text_list_ref

        # create the layout
        layout = QGridLayout(parent=self)
        layout.setContentsMargins(0, 0, 0, 0)
        [layout.setRowStretch(r, v) for r, v in enumerate((0, 0, 1))]
        self.setLayout(layout)

        # set up the listview widget
        self.listview = SidePanel.ListWidget(parent=self)
        self.listview.setToolTip('Right click on items for more options')
        self.listview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.refresh_list_view(self.model_list)
        self.model_list.subscribe(self.refresh_list_view)
        layout.addWidget(self.listview, 0, 0)

        # set up the rotate/translate widget
        self.pos_rot_widget = SidePanel.PositionRotationWidget(parent=self)
        self.pos_rot_widget.setToolTip('Moves the selected model. Has no effect on locked models')
        self.pos_rot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.listview.currentItemChanged.connect(self.on_current_item_changed)
        layout.addWidget(self.pos_rot_widget, 1, 0)

        # setup the console
        self.console = SidePanel.Console(parent=self)
        self.console.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.console, 2, 0)
    
    # expose child widget functions to the top level for easy access
    def refresh_list_view(self, model_list: List):
        self.listview.clear()
        for m in model_list:
            item = QListWidgetItem()
            content = SidePanel.ListWidget.ListItemWidget(model=m, parent=self)
            item.setSizeHint(content.minimumSizeHint())
            self.listview.addItem(item)
            self.listview.setItemWidget(item, content)

    def refresh_pos_rot_widget(self, model):
        self.pos_rot_widget.refresh_coordinates(model)

    def on_current_item_changed(self, current_item, previous_item):
        widget: SidePanel.ListWidget.ListItemWidget = self.listview.itemWidget(current_item)
        model = widget.model if widget else None
        self.refresh_pos_rot_widget(model)

    @property
    def selected_model(self):
        item = self.listview.currentItem()
        widget: SidePanel.ListWidget.ListItemWidget = self.listview.itemWidget(item)
        model = widget.model
        return model


    class ListWidget(QListWidget):
        
        def __init__(self, parent=None):
            super().__init__(parent=parent)


        class ListItemWidget(QWidget):
            """Represents the model in the side panel"""

            def __init__(self, model: O3DBaseModel, parent=None):
                super().__init__(parent=parent)
                self.side_panel = parent if isinstance(parent, SidePanel) else None
                self.model = model
                self.setLayout(QHBoxLayout(parent=self))
                layout = self.layout()
                layout.setAlignment(Qt.AlignRight)

                # set up the context menu
                self.setContextMenuPolicy(Qt.CustomContextMenu)
                self.customContextMenuRequested.connect(self.showContextMenu)

                # set up the text
                self.text = QLabel(parent=self, text=self.model.name)
                self.text.setAlignment(Qt.AlignLeft)
                self.text.setMaximumWidth(500)
                self.text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
                layout.addWidget(self.text)
                
                # create the binding for the lock check box
                self.locked_checkbox = QCheckBox(parent=self, text='L')
                self.locked_checkbox.setToolTip('Toggle model lock')
                self.locked_checkbox.setChecked(self.model.position_locked)
                self.locked_checkbox.stateChanged.connect(self.on_locked_changed)
                layout.addWidget(self.locked_checkbox)
                
                # create the binding for the visible checkbox
                self.visible_checkbox = QCheckBox(parent=self, text='V')
                self.visible_checkbox.setToolTip('Toggle model visibility')
                self.visible_checkbox.setChecked(self.model.visible)
                self.visible_checkbox.stateChanged.connect(self.on_visible_changed)
                layout.addWidget(self.visible_checkbox)

                # create the binding for the transparent checkbox
                self.transparent_checkbox = QCheckBox(parent=self, text='T')
                self.transparent_checkbox.setToolTip('Toggle model transparency')
                self.transparent_checkbox.setChecked(self.model.transparent)
                self.transparent_checkbox.stateChanged.connect(self.on_transparent_changed)
                layout.addWidget(self.transparent_checkbox)

            def on_locked_changed(self, state: int):
                self.model.position_locked = bool(state)

            def on_visible_changed(self, state: int):
                self.model.visible = bool(state)

            def on_transparent_changed(self, state: int):
                self.model.transparent = bool(state)

            def on_paint_uniform_color(self):
                # get a color from the color dialog
                color_dialog = QColorDialog(parent=None)
                color_dialog.setOption(QColorDialog.DontUseNativeDialog)
                color_dialog.setWindowFlag(Qt.WindowStaysOnTopHint)
                color = color_dialog.getColor()
                if color.isValid():
                    rgb = np.array(color.getRgbF()[:-1])
                    self.model.geometry.paint_uniform_color(rgb)
                    self.model.needs_update.true()

            def on_remove_color(self):
                if isinstance(self.model, O3DPointCloudModel):
                    self.model.geometry.colors = o3d_utils.Vector3dVector()
                elif isinstance(self.model, O3DTriangleMeshModel):
                    self.model.geometry.vertex_colors = o3d_utils.Vector3dVector()
                self.model.needs_update.true()

            def on_copy_coordinates(self):
                # copy coordinates as an array to the global clipboard
                clipboard = QClipboard()
                mimedata = QMimeData()
                # load in the positional data
                cartisian_coordinates, transformation_matrix = self.model.get_positional_data()
                data = cartisian_coordinates.tobytes()
                mimedata.setData('nparray', data)
                # load in the transform matrix
                data = transformation_matrix.tobytes()
                mimedata.setData('npmatrix', data)
                clipboard.setMimeData(mimedata, QClipboard.Clipboard)
                print('Copied positional data to clipboard')

            def on_paste_coordinates(self):
                # retrieve coordinates from global clipboard
                clipboard = QClipboard()
                data = clipboard.mimeData().data('nparray')
                cartisian_coordinates = np.frombuffer(data)
                cartisian_coordinates = cartisian_coordinates.reshape(2, 3)
                # retrieve the matrix from the global clipboard
                data = clipboard.mimeData().data('npmatrix')
                transformation_matrix = np.frombuffer(data)
                transformation_matrix = transformation_matrix.reshape(4, 4)
                # transform back to the origin
                to_origin = np.linalg.inv(self.model.transformation_matrix)
                transformation = np.dot(transformation_matrix, to_origin)
                self.model.geometry.transform(transformation)
                # tell the model where it's now at
                self.model.set_positional_data((cartisian_coordinates, transformation_matrix))
                self.model.needs_update.true()
                if self.side_panel is not None:
                    self.side_panel.refresh_pos_rot_widget(self.model)
                print('Pasted positional data from the clipboard')   
            
            def on_triangle_mesh_to_pointcloud(self):
                asyncio.create_task(self.triangle_mesh_to_pointcloud())

            async def triangle_mesh_to_pointcloud(self):
                dialog = TriangleMeshToPointCloudDialog(self.side_panel.model_list)
                await aprint('Opening STL -> Pointcloud Dialog')
                await dialog.run()

            def on_crop_pointcloud(self):
                asyncio.create_task(self.crop_pointcloud())

            async def crop_pointcloud(self):
                dialog = CroppingDialog(self.model, self.side_panel.model_list, self.side_panel.text_list)
                await aprint('Opening Cropping Dialog')
                await dialog.run()

            def on_remove_outliers(self):
                asyncio.create_task(self.remove_outliers())

            async def remove_outliers(self):
                dialog = OutlierRemovalDialog(self.model, self.side_panel.model_list)
                print('Opening outlier removal dialog')
                if await dialog.run():
                    print('removing outliers')
                else:
                    print('Outlier removal cancled')

            def on_register_pointclouds(self):
                asyncio.create_task(self.register_pointclouds())

            async def register_pointclouds(self):
                dialog = P2PRegistrationDialog(self.model, self.side_panel.model_list)
                await dialog.run()

            def on_compare_pointclouds(self):
                asyncio.create_task(self.compare_pointclouds())      

            async def compare_pointclouds(self):
                dialog = ComparisonDialog(self.model, self.side_panel.model_list)
                await dialog.run()

            def on_export_pointcoud(self):
                asyncio.create_task(self.export_pointcloud())

            async def export_pointcloud(self):
                name = self.model.name + '.txt'
                filename, *_ = QFileDialog.getSaveFileName(None, 'Export File', name)
                if filename:
                    path = Path(filename).resolve()
                    data = np.asarray(self.model.geometry.points)
                    np.savetxt(path, data, delimiter='    ')           
            
            def showContextMenu(self, pos):
                menu = QMenu(parent=self)
                # add the actions that do not depend on the model type
                callbacks = {
                    menu.addAction('Delete'): self.model.delete_later,
                    menu.addAction('Copy Coordinates'): self.on_copy_coordinates,
                    menu.addAction('Paste Coordinates'): self.on_paste_coordinates,
                    menu.addAction('Set Color'): self.on_paint_uniform_color,
                    menu.addAction('Remove Color'): self.on_remove_color,
                }

                # add the stl specific manu items
                if isinstance(self.model, O3DTriangleMeshModel):
                    stl_menu = QMenu(title='STL', parent=self)
                    stl_callbacks = {
                        stl_menu.addAction('STL -> Pointcloud'): self.on_triangle_mesh_to_pointcloud
                    }
                    callbacks.update(stl_callbacks)
                    menu.addMenu(stl_menu)
                # add the pointcloud specific menu items
                elif isinstance(self.model, O3DPointCloudModel):
                    pcd_menu = QMenu(title='PCD', parent=self)
                    pcd_callbacks = {
                        pcd_menu.addAction('Crop Pointcloud'): self.on_crop_pointcloud,
                        pcd_menu.addAction('Remove Outliers'): self.on_remove_outliers,
                        pcd_menu.addAction('Register Clouds'): self.on_register_pointclouds,
                        pcd_menu.addAction('Compare Clouds'):  self.on_compare_pointclouds,
                        pcd_menu.addAction('Export TSV'):      self.on_export_pointcoud,
                    }
                    callbacks.update(pcd_callbacks)
                    menu.addMenu(pcd_menu)
                # run the menu
                action = menu.exec(self.mapToGlobal(pos))
                if action:
                    callbacks.get(action)()


    class PositionRotationWidget(QWidget):

        def __init__(self, parent):
            super().__init__(parent=parent)
            self.side_panel = parent if isinstance(parent, SidePanel) else None
            self.setLayout(QGridLayout(parent=self))
            self.setMinimumWidth(100)
            layout: QGridLayout = self.layout()

            currow = 0
            # set up the header
            spacer = QSpacerItem(40, 0, QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
            layout.addItem(spacer, currow, 0)
            labels = [Standard.Label(char, parent=self) for char in 'X,Y,Z'.split(',')]
            for i, l in enumerate(labels):
                layout.addWidget(l, currow, i+1)

            currow += 1
            # set up the position row
            label = Standard.Label('Pos', parent=self)
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label, currow, 0)
            self.pos_spinboxes = [Standard.DoubleSpinbox(parent=self) for i in range(3)]
            for i, (spinbox, axis) in enumerate(zip(self.pos_spinboxes, 'X,Y,Z'.split(','))):
                spinbox.valueChanged.connect(self.on_coordinate_changed_fast)
                spinbox.editingFinished.connect(self.on_coordinate_changed_slow)
                spinbox.setMaximum(100_000_000.0)
                spinbox.setMinimum(-100_000_000.0)
                spinbox.setToolTip(f'{axis} pos')
                layout.addWidget(spinbox, currow, i+1)

            currow += 1
            # set up the rotation row
            label = Standard.Label('Rot', parent=self)
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label, currow, 0)
            self.rot_spinboxes = [Standard.DoubleSpinbox(parent=self) for i in range(3)]
            for i, (spinbox, axis) in enumerate(zip(self.rot_spinboxes, 'X, Y, Z'.split())):
                spinbox.valueChanged.connect(self.on_coordinate_changed_fast)
                spinbox.editingFinished.connect(self.on_coordinate_changed_slow)
                spinbox.setMaximum(100_000_000.0)
                spinbox.setMinimum(-100_000_000.0)
                spinbox.setToolTip(f'{axis} rot')
                layout.addWidget(spinbox, currow, i+1)

            currow += 1
            # set up the auto update checkbox
            self.fast_updates_checkbox = QCheckBox(parent=self)
            self.fast_updates_checkbox.setText('Fast Updates')
            layout.addWidget(self.fast_updates_checkbox, currow, 3, 1, -4)

        def refresh_coordinates(self, model: O3DBaseModel = None):
            """If the model is not none, the coordinates
            are read from the model and written to the spinboxes.
            If the model is None, the spinboxes are set to zero"""
            if model is not None:
                pos_vec, rot_vec = model.cartisian_coordinates
                [sb.setValue(v) for sb, v in zip(self.pos_spinboxes, pos_vec)]
                [sb.setValue(np.rad2deg(v)) for sb, v in zip(self.rot_spinboxes, rot_vec)]
            else:
                [sb.setValue(0) for sb in self.pos_spinboxes]
                [sb.setValue(0) for sb in self.rot_spinboxes]

        def on_coordinate_changed_fast(self):
            """Writes the coordinate changes to the model, which are updated in the visualizer
            asyncronosly."""
            if any([sb.hasFocus() for sb in (*self.pos_spinboxes, *self.rot_spinboxes)]) and self.fast_updates_checkbox.isChecked():
                position_vector = np.array([sb.value() for sb in self.pos_spinboxes])
                rotation_vector = np.array([np.deg2rad(sb.value()) for sb in self.rot_spinboxes])
                model = self.side_panel.selected_model
                model.geometry.cartisian_transform(position_vector, rotation_vector)

        def on_coordinate_changed_slow(self):
            if not self.fast_updates_checkbox.isChecked():
                position_vector = np.array([sb.value() for sb in self.pos_spinboxes])
                rotation_vector = np.array([np.deg2rad(sb.value()) for sb in self.rot_spinboxes])
                model = self.side_panel.selected_model
                model.geometry.cartisian_transform(position_vector, rotation_vector)


    class Console(QPlainTextEdit):

        def __init__(self, parent=None):
            super().__init__(parent=parent)
            self.setReadOnly(True)
            policy = self.sizePolicy()
            policy.setHorizontalPolicy(QSizePolicy.Expanding)
            policy.setVerticalPolicy(QSizePolicy.Maximum)
            self.setSizePolicy(policy)

        def write(self, text: str):
            self.appendPlainText(text)
    