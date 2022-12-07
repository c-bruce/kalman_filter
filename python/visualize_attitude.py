# Date: 04/12/2022
# Author: Callum Bruce

from tvtk.api import tvtk
from tvtk.pyface.scene_editor import SceneEditor

from mayavi import mlab
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSplitter

from pyface.qt import QtGui

from traits.api import HasTraits, Instance
from traitsui.api import View, Item

import numpy as np
import time

from referenceframe import ReferenceFrame
from helpermath import euler2quaternion

# from ..helpermath.helpermath import *
class Visualization(HasTraits):
    """
    Visualization class.

    Notes:
        - tvtk.Actor() objects can be added to self.scene in the normal way.
    """
    scene3d = Instance(MlabSceneModel, ())

    def __init__(self, **traits):
        super(Visualization, self).__init__(**traits)
        mlab.pipeline.scalar_field([[0]], figure=self.scene3d.mayavi_scene) # Weird work around to get self.scene3d.mlab.orientation_axes() working
        self.scene3d.mlab.orientation_axes()

    view = View(Item('scene3d', editor=SceneEditor(scene_class=MayaviScene), height=500, width=500, show_label=False), resizable=True)

class MayaviQWidget(QtGui.QWidget):
    """
    MayaviQWidget class.

    Notes:
        - Can be added as a qt widget in the normal way.
    """
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        layout = QtGui.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.visualization = Visualization()
        # edit_traits call will generate the widget to embed
        self.ui = self.visualization.edit_traits(parent=self, kind='subpanel').control
        layout.addWidget(self.ui)
        self.ui.setParent(self)

class MainWidget(QSplitter):
    """
    MainWidget class.

    Main widget containing a QTreeView and MayaviQWidget.
    """
    def __init__(self, *args, **kwargs):
        super(MainWidget, self).__init__(*args, **kwargs)
        self.setOrientation(Qt.Vertical)

        # Setup MayaviQWidget
        self.viewer = MayaviQWidget()
        self.addWidget(self.viewer)

        # Setup actors
        self.actors = {}

        # Setup reference frames
        self.reference_frames = {'fixed' : ReferenceFrame(), 'moving' : ReferenceFrame()}
        self.plot_reference_frames()

    @mlab.animate(delay=100)
    def anim(self):
        while True:
            attitude = np.radians(np.loadtxt('attitude.csv', delimiter=','))
            #ang = np.random.randint(0, 90)
            self.rotate_reference_frame(attitude, self.reference_frames['moving'], 'moving')
            yield
    
    def plot_reference_frames(self):
        for key in self.reference_frames:
            self.plot_reference_frame(self.reference_frames[key], key, scale=1)
    
    def plot_reference_frame(self, reference_frame, actor_name, scale=1):
        reference_frame_actor = {}

        position = np.array([0, 0, 0])
        i, j, k = reference_frame.getIJK()
        i_points = np.append(np.expand_dims(position, axis=0), np.expand_dims(position + (i  * scale), axis=0), axis=0)
        j_points = np.append(np.expand_dims(position, axis=0), np.expand_dims(position + (j  * scale), axis=0), axis=0)
        k_points = np.append(np.expand_dims(position, axis=0), np.expand_dims(position + (k  * scale), axis=0), axis=0)
        i_source, j_source, k_source, i_actor, j_actor, k_actor = reference_frame.getActors()
        i_source.trait_set(points=i_points)
        j_source.trait_set(points=j_points)
        k_source.trait_set(points=k_points)
        reference_frame_actor['RF'] = [i_points, j_points, k_points, i_source, j_source, k_source, i_actor, j_actor, k_actor]

        # Add actors to the scene
        self.actors[actor_name] = reference_frame_actor
        self.viewer.visualization.scene3d.add_actor(self.actors[actor_name]['RF'][6])
        self.viewer.visualization.scene3d.add_actor(self.actors[actor_name]['RF'][7])
        self.viewer.visualization.scene3d.add_actor(self.actors[actor_name]['RF'][8])
    
    def rotate_reference_frame(self, euler, reference_frame, actor_name, scale=1):
        quaternion = euler2quaternion(euler)
        reference_frame.rotateAbs(quaternion)

        position = np.array([0, 0, 0])
        i, j, k = reference_frame.getIJK()
        i_points = np.append(np.expand_dims(position, axis=0), np.expand_dims(position + (i * scale), axis=0), axis=0)
        j_points = np.append(np.expand_dims(position, axis=0), np.expand_dims(position + (j * scale), axis=0), axis=0)
        k_points = np.append(np.expand_dims(position, axis=0), np.expand_dims(position + (k * scale), axis=0), axis=0)
        self.actors[actor_name]['RF'][3].trait_set(points=i_points)
        self.actors[actor_name]['RF'][4].trait_set(points=j_points)
        self.actors[actor_name]['RF'][5].trait_set(points=k_points)
        self.viewer.visualization.scene3d.render()

def main():
    fig = MainWidget()
    fig.anim()
    fig.showMaximized()
    mlab.show()

if __name__ == "__main__":
    main()
