from __future__ import division

import os
import traceback
os.environ['ETS_TOOLKIT'] = 'qt4'

import argparse
import inspect
import numpy as np
from numpy.linalg import norm
from scipy.interpolate import griddata
import tables

from mayavi import mlab
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene
from tvtk.pyface.scene_editor import SceneEditor
from pyface.action.api import ToolBarManager
from mayavi.tools.camera import deg2rad, rad2deg

from mayavi.modules.api import Outline

from traits.api import Bool, Float, HasTraits, Instance, Range, on_trait_change
from traitsui.api import Item, View, HSplit, VGroup, VSplit
from PyQt4.QtCore import QObject, QTimer, SIGNAL

from tvtk.api import tvtk
from tvtk.util.ctf import ColorTransferFunction, PiecewiseFunction, set_lut
import vtk

from nputil import meshgrid_nd
from prediction import predict_on_volume
import recorder


class RotateAroundZInteractor(tvtk.InteractorStyleTrackballCamera):
    # Definition of required constant from vtkInteractorStyle.h:111.
    # Unfortunately, it is not exported to the Python API.
    VTKIS_ROTATE = 1

    started = Bool(False)
    elevation = Float(135.0)
    roll = Float(180.0)

    def __init__(self):
        tvtk.InteractorStyleTrackballCamera.__init__(self)
        self.add_observer("MouseMoveEvent", self.on_mouse_move)

    def on_mouse_move(self, obj, event):
        if self.state == self.VTKIS_ROTATE:
            self.rotate()
            self.invoke_event(vtk.vtkCommand.InteractionEvent)
        elif self.state != 0:
            tvtk.InteractorStyleTrackballCamera.on_mouse_move(self)
            self.elevation = mlab.view()[1]
            self.roll = self.calc_current_roll()

    def rotate(self):
            rwi = self.interactor
            self.find_poked_renderer(*rwi.event_position)
            camera = self.current_renderer.active_camera

            mouse_move = rwi.event_position - rwi.last_event_position
            size_factor = -20.0 / self.current_renderer.render_window.size
            move = mouse_move * size_factor * self.motion_factor

            view_dir = camera.focal_point - camera.position
            view_dir /= norm(view_dir)
            n = np.cross(view_dir, camera.view_up)
            n /= norm(n)

            u = camera.view_up[2]
            v = n[2]
            w = view_dir[2]
            anorm = norm([u, v, w])

            # Convert angle around z-axis to azimuth, elevation and roll angle.
            theta = deg2rad(move[0])
            elevation = - np.arcsin(
                (u * w * (1 - np.cos(theta)) - v * anorm * np.sin(theta)) /
                anorm ** 2)
            roll = np.arcsin(
                (u * v * (1 - np.cos(theta)) - w * anorm * np.sin(theta)) /
                anorm ** 2 / np.cos(elevation))
            azimuth = np.arcsin(
                (v * w * (1 - np.cos(theta)) - u * anorm * np.sin(theta)) /
                anorm ** 2 / np.cos(elevation))

            camera.azimuth(rad2deg(azimuth))
            camera.elevation(rad2deg(elevation))
            camera.roll(rad2deg(roll))
            camera.orthogonalize_view_up()

            self.elevation += move[1]
            camera.elevation(self.elevation - mlab.view()[1])
            camera.roll(self.roll - self.calc_current_roll())

            rwi.render()

    def calc_current_roll(self):
        camera = self.current_renderer.active_camera
        n = np.cross(camera.focal_point - camera.position, camera.view_up)
        n /= norm(n)
        roll = np.arccos(camera.view_up[2] / norm(np.array(
            [n[2], camera.view_up[2]])))
        roll *= -1 * np.sign(n[2])
        return rad2deg(roll)


class ThinToolbarEditor(MayaviScene):
    def __init__(self, *args, **kwargs):
        MayaviScene.__init__(self, *args, **kwargs)

    def _get_tool_bar_manager(self):
        tbm = ToolBarManager(*self.actions)
        tbm.show_tool_names = False
        return tbm


def current_figure_as_default(fn):
    def with_cf_as_def(*args, **kwargs):
        is_figure_set_in_args = False
        try:
            arg_idx = inspect.getargspec(fn).args.index('figure')
            if arg_idx < len(args):
                is_figure_set_in_args = True
        except:
            pass

        is_figure_set_in_kwargs = 'figure' in kwargs and \
            not kwargs['figure'] is None

        if not is_figure_set_in_args and not is_figure_set_in_kwargs:
            kwargs['figure'] = mlab.gcf()
        return fn(*args, **kwargs)
    return with_cf_as_def


class PreviewEnabledRenderingFunction(object):
    def __init__(self, scene, timeout=250):
        self.scene = scene
        self._fine_rendering_allocated_time = None
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(timeout)
        QObject.connect(self._timer, SIGNAL('timeout()'), self.render)

    def abort_rendering(self):
        self._timer.stop()
        if self._fine_rendering_allocated_time is not None:
            self.scene.renderer.allocated_render_time = \
                self._fine_rendering_allocated_time
            self._fine_rendering_allocated_time = None

    def render(self):
        self.scene.renderer.allocated_render_time = \
            self._fine_rendering_allocated_time
        self.scene.render()

    def render_preview(self):
        self._fine_rendering_allocated_time = \
            self.scene.renderer.allocated_render_time
        self.scene.renderer.allocated_render_time = 0
        self.scene.render()
        self._timer.start()

    def __call__(self):
        self.render_preview()


class PlumeVisualizer(HasTraits):
    trajectory_color = (0, 126 / 255, 203 / 255)
    marker_color = (0, 172 / 255, 1)

    prediction = Instance(MlabSceneModel, ())
    mse = Instance(MlabSceneModel, ())
    truth = Instance(MlabSceneModel, ())

    prediction_cutoff = Range(0.0, 1.0, 0.7)
    mse_cutoff = Range(0.0, 1.0, 0.5)

    view = View(
        HSplit(
            VGroup(
                Item('prediction_cutoff', width=200),
                Item('mse_cutoff', width=200),
            ),
            VSplit(
                Item(
                    'prediction', show_label=False,
                    editor=SceneEditor(scene_class=ThinToolbarEditor)),
                HSplit(
                    Item(
                        'mse', show_label=False,
                        editor=SceneEditor(scene_class=ThinToolbarEditor)),
                    Item(
                        'truth', show_label=False,
                        editor=SceneEditor(scene_class=ThinToolbarEditor))
                )
            )
        ), resizable=True, height=1.0, width=1.0)

    def __init__(self, data, end=None):
        HasTraits.__init__(self)
        self.conf = data.root.conf[0]
        self.render_prediction_with_preview = PreviewEnabledRenderingFunction(
            self.prediction.scene)
        self.render_mse_with_preview = PreviewEnabledRenderingFunction(
            self.mse.scene)
        self.data = data
        if end is None:
            self.end = len(self.data.root.positions)
        else:
            self.end = end

        self._init_scene(self.prediction)
        self._init_scene(self.mse)
        self._init_scene(self.truth)
        self._populate_scene(self.prediction, 'Prediction')
        self._populate_scene(self.mse, 'MSE')
        self._populate_scene(self.truth, 'Truth')

    @staticmethod
    def _init_scene(scene):
        scene.background = (1.0, 1.0, 1.0)
        scene.foreground = (0.0, 0.0, 0.0)

    def _populate_scene(self, scene, title):
        trajectories = self.data.root.positions.read()[:, :self.end, :]
        self.plot_uav_trajectories(trajectories, figure=scene.mayavi_scene)

        area = self.conf['area']
        scene.mayavi_scene.children[0].add_child(
            Outline(manual_bounds=True, bounds=area.flatten()))

        mlab.title(title, figure=scene.mayavi_scene)

    def _plot_fit(self):
        pred, mse, positions = self.calc_estimation(self.data)
        self._prediction_volume = self.plot_volume2(
            positions, pred, self.prediction_cutoff,
            figure=self.prediction.mayavi_scene)
        self._mse_volume = self.plot_volume2(
            positions, mse, self.prediction_cutoff,
            figure=self.mse.mayavi_scene)

    def _plot_plume(self):
        area = self.conf['area']
        ogrid = [np.linspace(*dim, num=res) for dim, res in zip(
            area, (20, 20, 20))]
        x, y, z = meshgrid_nd(*ogrid)
        values = griddata(
            self.data.root.gt_locations.read(),
            self.data.root.gt_samples.read(),
            np.column_stack((x.flat, y.flat, z.flat))).reshape(x.shape)
        self._truth_volume = self.plot_volume2(
            (x, y, z), values, 0.1, figure=self.truth.mayavi_scene)
        mlab.points3d(
            *self.data.root.sample_locations.read().T, scale_factor=5,
            color=(0.7, 0.0, 0.0), figure=self.truth.mayavi_scene)

    # FIXME think of better name
    @classmethod
    @current_figure_as_default
    def plot_volume2(cls, positions, data, cutoff, figure):
        vol = cls.plot_volume(positions, data, figure)
        cls._set_cutoff(vol, cutoff)
        return vol

    @on_trait_change('prediction.activated, mse.activated, truth.activated')
    def init_camera(self):
        all_activated = self.prediction.scene.interactor is not None and \
            self.mse.scene.interactor is not None and \
            self.truth.scene.interactor is not None
        if not all_activated:
            return

        self.prediction.scene.interactor.interactor_style = \
            RotateAroundZInteractor()
        self.mse.scene.interactor.interactor_style = RotateAroundZInteractor()
        self.truth.scene.interactor.interactor_style = \
            RotateAroundZInteractor()

        try:
            self._plot_fit()
        except:
            traceback.print_exc()
        self._plot_plume()
        ax = mlab.axes(
            extent=[-140, 140, -140, 140, -80, 0], xlabel='', ylabel='',
            zlabel='')
        ax.axes.label_format = '%2.0f'

        mlab.sync_camera(self.prediction.mayavi_scene, self.mse.mayavi_scene)
        mlab.sync_camera(self.mse.mayavi_scene, self.prediction.mayavi_scene)
        mlab.sync_camera(self.mse.mayavi_scene, self.truth.mayavi_scene)
        mlab.sync_camera(self.truth.mayavi_scene, self.mse.mayavi_scene)
        mlab.view(
            azimuth=135, elevation=135, distance=600, roll=-120,
            figure=self.prediction.mayavi_scene)

    @classmethod
    @current_figure_as_default
    def plot_uav_trajectory(cls, positions, figure):
        mlab.plot3d(
            *positions.T, tube_radius=1, line_width=0,
            color=cls.trajectory_color, figure=figure)
        mlab.points3d(
            *positions[-1], scale_factor=5, color=cls.marker_color,
            figure=figure)

    @classmethod
    @current_figure_as_default
    def plot_uav_trajectories(cls, trajectories, figure):
        for uav_positions in trajectories:
            cls.plot_uav_trajectory(uav_positions, figure)

    def calc_estimation(self, data):
        predictor = recorder.load_obj(data.root.gp)
        area = data.root.conf[0]['area']
        return predict_on_volume(predictor, area, [30, 30, 20])

    @staticmethod
    @current_figure_as_default
    def plot_volume((x, y, z), values, figure, vmin=None, vmax=None):
        volume = mlab.pipeline.volume(
            mlab.pipeline.scalar_field(
                x, y, z, values, figure=figure, colormap='Reds'),
            vmin=vmin, vmax=vmax, figure=figure)
        volume.lut_manager.show_scalar_bar = True
        return volume

    @staticmethod
    def _set_cutoff(volume, cutoff):
        range_min, range_max = volume.current_range
        vmin = range_min + cutoff * (range_max - range_min)

        otf = PiecewiseFunction()
        otf.add_point(range_min, 0.0)
        otf.add_point(vmin, 0.0)
        otf.add_point(range_max, 0.2)
        volume._otf = otf
        volume.volume_property.set_scalar_opacity(otf)

        ctf = ColorTransferFunction()
        ctf.range = volume.current_range
        ctf.add_rgb_point(range_min, 1.0, 1.0, 1.0)
        ctf.add_rgb_point((range_min + range_max) * (1.0 / 3.0), 0.0, 0.0, 1.0)
        ctf.add_rgb_point((range_min + range_max) * (2.0 / 3.0), 1.0, 0.0, 1.0)
        ctf.add_rgb_point(range_max, 1.0, 0.0, 0.0)
        volume._ctf = ctf
        volume.volume_property.set_color(ctf)
        set_lut(volume.lut_manager.lut, volume.volume_property)

    def _prediction_cutoff_changed(self):
        self.render_prediction_with_preview.abort_rendering()
        self._set_cutoff(self._prediction_volume, self.prediction_cutoff)
        self.render_prediction_with_preview()

    def _mse_cutoff_changed(self):
        self.render_mse_with_preview.abort_rendering()
        self._set_cutoff(self._mse_volume, self.mse_cutoff)
        self.render_mse_with_preview()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', nargs=1, type=int, help='Number of steps to visualize.')
    parser.add_argument('filename', nargs=1, type=str)
    args = parser.parse_args()

    with tables.openFile(args.filename[0], 'r') as data:
        end = None
        if args.t is not None:
            end = args.t[0]
        visualizer = PlumeVisualizer(data, end)
        visualizer.configure_traits()
