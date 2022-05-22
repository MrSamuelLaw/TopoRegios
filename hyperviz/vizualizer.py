import sys
import asyncio
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog
from open3d import utility as o3d_utils
from open3d import io as o3d_io
from open3d import visualization as o3d_vis
from open3d.visualization.gui import Application as o3d_app
from hyperviz import LJ_X8000
from hyperviz.models import O3DBaseModel, O3DTriangleMeshModel, O3DPointCloudModel, O3DTextModel, O3DModelList
from hyperviz.utilities import BoolTrigger, WatchableList, R_TRIG, unique_rename, aprint
from hyperviz.templates import BMPImportDialog, SidePanel



class Vizualizer():
    
    model_list = O3DModelList()
    text_list = WatchableList()
    running = BoolTrigger(False)
    _model_count_rtrig = R_TRIG(False)
    _window: o3d_vis.O3DVisualizer = None
    _instance = None
    _sidepanel = None
    _frame_height = 0
    
    def __init__(self):
        pass

    def launch(self):
        """Launches the gui app"""

        # init the o3d window and app
        o3d_app.instance.initialize()
        self._window = o3d_vis.O3DVisualizer('Hyperviz')
        self._window.show_skybox(False)
        self._window.show_axes = False
        self._window.show_settings = False
        o3d_app.instance.add_window(self._window)        
       
        # set up the callbacks
        self._window.add_action('import stl', self.on_import_stl)
        self._window.add_action('import bmp', self.on_import_bmp)

        # get the camera
        camera = self._window.scene.view.get_camera()

        # init the pyside window and app
        self.qapp = QApplication()
        self._sidepanel = SidePanel(self.model_list, self.text_list, camera)
        o3d_utils.set_verbosity_level(o3d_utils.VerbosityLevel.Error)
        sys.stdout = self._sidepanel.console
        sys.stderr = self._sidepanel.console
        self._sidepanel.show()
        self.layout_fullscreen()

        # pass the window to the controller
        self.running.true()

    def layout_fullscreen(self):
        """Lays out the screen and if the sidepanel
        is visible it shows that next to the screen."""
        # inputs 
        screen = self.qapp.primaryScreen()
        if self._sidepanel is not None:
            screen = self._sidepanel.screen()
        ratio = screen.devicePixelRatio()
        qrect = screen.availableGeometry() 
        sp_width = 0.0
        viz_width = 1.0

        rect = self._window.content_rect
        self._frame_height = rect.y
        if self._sidepanel is not None:
            sp_width = 0.25
            viz_width = 0.75
            eps = 3  # px
            content_size = self._sidepanel.size()
            frame_size = self._sidepanel.frameSize()
            x_offset = frame_size.width() - content_size.width()
            y_offset = frame_size.height() - content_size.height() - eps
            qgeo = self._sidepanel.geometry()
            qgeo.setX(x_offset); qgeo.setY(y_offset)
            qgeo.setHeight(int(qrect.height() - y_offset))
            qgeo.setWidth(qrect.width()*0.25)
            self._sidepanel.setGeometry(qgeo)
            self._frame_height = y_offset
        
        frame = self._window.os_frame
        frame.x = int(qrect.width()*ratio*sp_width)
        frame.y = int(self._frame_height*ratio)
        frame.width = int(qrect.width()*ratio*viz_width)
        frame.height = int(qrect.height()*ratio - frame.y)
        self._window.os_frame = frame
        self._window.post_redraw()

    async def cycle_event_loop(self):
        """Runs the event loop one tick"""
        while o3d_app.instance.run_one_tick():
            try:
                self.qapp.processEvents()
                if any([m.needs_update() for m in self.model_list]):
                    model = [m for m in self.model_list if m.needs_update()][0]
                    self.update_model(model)
                if any([t.needs_update() for t in self.text_list]):
                    self.update_text()
                await asyncio.sleep(1E-9)
            except Exception as e:
                print(e)

    def remove_model(self, model: O3DBaseModel):
        """Removes a model from the vizualizer's model list and deletes it from the scene"""
        if model in self.model_list:
            self.model_list.remove(model)
        
    def update_model(self, model):
        if not isinstance(model, O3DBaseModel):
            raise TypeError('Model does not extend from type hyperviz.O3DBaseModel')
        model.needs_update.false()
        self._window.remove_geometry(model.name)
        if model.delete is False and model.visible:
            self._window.add_geometry(model.asO3Ddict())
        elif model.delete is True:
            self.remove_model(model)
        if self._model_count_rtrig(bool(len(self.model_list))):
            self._window.reset_camera_to_default()
        self._window.post_redraw()

    def update_text(self):
        self._window.clear_3d_labels()
        deletion_targets = [t for t in self.text_list if t.delete]
        [self.text_list.remove(t) for t in deletion_targets]
        [self._window.add_3d_label(t.xyz_point, t.text) for t in self.text_list]
        [t.needs_update.false() for t in self.text_list]

    def on_import_stl(self, O3DVisualizer):
        asyncio.create_task(self.import_stl())

    async def import_stl(self):
        file_dialog = QFileDialog(caption='import stl')
        if file_dialog.exec():
            await aprint('Importing STL')
            path = Path(file_dialog.selectedFiles()[0]).resolve()
            stl = o3d_io.read_triangle_mesh(str(path))
            name = unique_rename([m.name for m in self.model_list], path.name, '_copy')
            stl = O3DTriangleMeshModel(name, trianglemesh=stl)
            self.model_list.append(stl)
            await aprint('STL import complete')

    def on_import_bmp(self, O3DVisualizer):
        asyncio.create_task(self.import_bmp())
        
    async def import_bmp(self):
        file_dialog = QFileDialog(caption='import bmp')
        if file_dialog.exec():
            print('Awaiting offset and max_points from user')
            path = Path(file_dialog.selectedFiles()[0]).resolve()
            dialog = BMPImportDialog()
            if await dialog.run():
                await aprint('Importing bmp as point cloud')
                offset, npoints = dialog.offset, dialog.npoints
                point_cloud, text = LJ_X8000.bmp_to_point_cloud(path, offset)
                await aprint(text)
                if npoints and (np.asarray(point_cloud.points).shape[0] > npoints):
                    await aprint('downsampling')
                    ratio = npoints/np.asarray(point_cloud.points).shape[0]
                    point_cloud = point_cloud.random_down_sample(ratio)
                name = unique_rename([m.name for m in self.model_list], path.name, '_copy')
                point_cloud = O3DPointCloudModel(name, pointcloud=point_cloud)
                self.model_list.append(point_cloud)
                await aprint('BMP Import complete')
            else:
                await aprint('BPM Import cancled')






