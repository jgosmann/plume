from __future__ import division

import argparse
import inspect
import numpy as np
from numpy.linalg import norm
import tables

from mayavi import mlab
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene
from tvtk.pyface.scene_editor import SceneEditor
from pyface.action.api import ToolBarManager
from mayavi.tools.camera import deg2rad, rad2deg

from mayavi.modules.api import Outline

from traits.api import Bool, Float, HasTraits, Instance, on_trait_change
from traitsui.api import Item, View, VSplit

from tvtk.api import tvtk
import vtk

from sklearn import gaussian_process


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


class PlumeVisualizer(HasTraits):
    trajectory_color = (0, 126 / 255, 203 / 255)
    marker_color = (0, 172 / 255, 1)

    prediction = Instance(MlabSceneModel, ())
    mse = Instance(MlabSceneModel, ())

    view = View(VSplit(
        Item(
            'prediction', show_label=False,
            editor=SceneEditor(scene_class=ThinToolbarEditor)),
        Item(
            'mse', show_label=False,
            editor=SceneEditor(scene_class=ThinToolbarEditor))),
        resizable=True, height=1.0, width=1.0)

    def __init__(self, data):
        HasTraits.__init__(self)

        self._init_scene(self.prediction)
        self._init_scene(self.mse)

        trajectories = data.root.positions.read()
        self.plot_uav_trajectories(
            trajectories, figure=self.prediction.mayavi_scene)
        self.plot_uav_trajectories(trajectories, figure=self.mse.mayavi_scene)

        pred, mse, positions = self.calc_estimation(data)
        self.plot_volume(positions, pred, self.prediction.mayavi_scene)
        self.plot_volume(positions, mse, self.mse.mayavi_scene)

        self.prediction.mayavi_scene.children[0].add_child(
            Outline(manual_bounds=True, bounds=[-140, 140, -140, 140, -80, 0]))
        self.mse.mayavi_scene.children[0].add_child(
            Outline(manual_bounds=True, bounds=[-140, 140, -140, 140, -80, 0]))

    @staticmethod
    def _init_scene(scene):
        scene.background = (1.0, 1.0, 1.0)
        scene.foreground = (0.0, 0.0, 0.0)

    @on_trait_change('prediction.activated, mse.activated')
    def init_camera(self):
        self.prediction.scene.interactor.interactor_style = \
            RotateAroundZInteractor()
        self.mse.scene.interactor.interactor_style = RotateAroundZInteractor()

        mlab.sync_camera(self.prediction.mayavi_scene, self.mse.mayavi_scene)
        mlab.sync_camera(self.mse.mayavi_scene, self.prediction.mayavi_scene)
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

    @staticmethod
    def calc_estimation(data):
        # FIXME ensure same predictor as in simulation
        gp = gaussian_process.GaussianProcess(nugget=0.5)
        gp.fit(
            data.root.positions.read()[0, :, :],
            data.root.plume_measurements.read()[0])

        # FIXME read from config
        x, y, z = np.mgrid[-140:141:5, -140:141:5, -80:1:5]
        pred, mse = gp.predict(
            np.column_stack((x.flat, y.flat, z.flat)), eval_MSE=True)
        pred = pred.reshape(x.shape)
        mse = mse.reshape(x.shape)
        return pred, mse, (x, y, z)

    @staticmethod
    @current_figure_as_default
    def plot_volume((x, y, z), values, figure, vmin=None, vmax=None):
        mlab.pipeline.volume(
            mlab.pipeline.scalar_field(x, y, z, values, figure=figure),
            vmin=vmin, vmax=vmax, figure=figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs=1, type=str)
    args = parser.parse_args()

    with tables.open_file(args.filename[0], 'r') as data:
        PlumeVisualizer(data).configure_traits()
