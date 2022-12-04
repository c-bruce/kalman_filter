# Date: 04/12/2022
# Author: Callum Bruce

from tvtk.api import tvtk
from tvtk.pyface.scene_editor import SceneEditor

from mayavi import mlab
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSplitter, QPushButton

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

        # self.slider = QSlider(Qt.Horizontal)
        # self.slider.setFocusPolicy(Qt.StrongFocus)
        # self.slider.setTickPosition(QSlider.TicksBothSides)
        # self.slider.setTickInterval(10)
        # self.slider.setSingleStep(1)
        # self.slider.setMaximum(100)
        # self.slider.setMinimum(0)
        # self.slider.setValue(0)
        # self.slider.valueChanged.connect(self.sliderValueChange)
        # self.addWidget(self.slider)
        # self.button = QPushButton()
        # self.button.changeEvent(self.hello)
        # self.addWidget(self.button)

        # Setup actors
        self.actors = {}
    
    def hello(self):
        print('hello')
    
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

    def loadSystem(self, system):
        self.system = system
        self.slider.setMaximum(max(list(system.timesteps.keys())))
        timesteps = sorted(list(self.system.timesteps.keys()))
        # DateTime actor
        text = 'DateTime : ' + str(system.current.getDatetime()) + ', JulianDate : ' + str(system.current.getJulianDate())
        text_source = tvtk.TextSource(text=text)
        text_mapper = tvtk.PolyDataMapper2D(input_connection=text_source.output_port)
        text_actor = tvtk.Actor2D(mapper=text_mapper)
        self.actors['DateTime'] = [text_source, text_mapper, text_actor]
        self.viewer.visualization.scene3d.add_actor(self.actors['DateTime'][2])
        # CelestialBodies
        for celestial_body in self.system.current.celestial_bodies.values():
            celestial_body_actor = {}
            # Get base actor
            celestial_body_actor['base'] = celestial_body.getActor()
            # Get bodyRF actors
            radius = celestial_body.getRadius()
            position = celestial_body.getPosition()
            i, j, k = celestial_body.bodyRF.getIJK()
            i_points = np.append(np.expand_dims(position, axis=0), np.expand_dims(position + (i * radius * 1.5), axis=0), axis=0)
            j_points = np.append(np.expand_dims(position, axis=0), np.expand_dims(position + (j * radius * 1.5), axis=0), axis=0)
            k_points = np.append(np.expand_dims(position, axis=0), np.expand_dims(position + (k * radius * 1.5), axis=0), axis=0)
            i_source, j_source, k_source, i_actor, j_actor, k_actor = celestial_body.bodyRF.getActors()
            i_source.trait_set(points=i_points)
            j_source.trait_set(points=j_points)
            k_source.trait_set(points=k_points)
            celestial_body_actor['bodyRF'] = [i_points, j_points, k_points, i_source, j_source, k_source, i_actor, j_actor, k_actor]
            # Get trajectory actor
            points = np.empty([len(timesteps), 3])
            for i in range(0, len(timesteps)):
                points[i,:] = system.timesteps[timesteps[i]].celestial_bodies[celestial_body.name].getPosition()
            line_source = tvtk.LineSource(points=points) # Can modify line_source using line_source.trait_set(points=data)
            line_mapper = tvtk.PolyDataMapper(input_connection=line_source.output_port)
            p = tvtk.Property(line_width=2, color=(1, 1, 1))
            line_actor = tvtk.Actor(mapper=line_mapper, property=p)
            celestial_body_actor['trajectory'] = [points, line_source, line_actor]
            # Add actors to the scene
            self.actors[celestial_body.name] = celestial_body_actor
            self.viewer.visualization.scene3d.add_actor(self.actors[celestial_body.name]['base'])
            self.viewer.visualization.scene3d.add_actor(self.actors[celestial_body.name]['bodyRF'][6])
            self.viewer.visualization.scene3d.add_actor(self.actors[celestial_body.name]['bodyRF'][7])
            self.viewer.visualization.scene3d.add_actor(self.actors[celestial_body.name]['bodyRF'][8])
            self.viewer.visualization.scene3d.add_actor(self.actors[celestial_body.name]['trajectory'][2])
        # Vessels
        for vessel in self.system.current.vessels.values():
            vessel_actor = {}
            # Get bodyRF actors
            position = vessel.getPosition()
            i, j, k = vessel.bodyRF.getIJK()
            i_points = np.append(np.expand_dims(position, axis=0), np.expand_dims(position + (i * 100000), axis=0), axis=0)
            j_points = np.append(np.expand_dims(position, axis=0), np.expand_dims(position + (j * 100000), axis=0), axis=0)
            k_points = np.append(np.expand_dims(position, axis=0), np.expand_dims(position + (k * 100000), axis=0), axis=0)
            i_source, j_source, k_source, i_actor, j_actor, k_actor = vessel.bodyRF.getActors()
            i_source.trait_set(points=i_points)
            j_source.trait_set(points=j_points)
            k_source.trait_set(points=k_points)
            vessel_actor['bodyRF'] = [i_points, j_points, k_points, i_source, j_source, k_source, i_actor, j_actor, k_actor]
            # Get trajectory actor
            points = np.empty([len(timesteps), 3])
            for i in range(0, len(timesteps)):
                points[i,:] = system.timesteps[timesteps[i]].vessels[vessel.name].getPosition()
            line_source = tvtk.LineSource(points=points) # Can modify line_source using line_source.trait_set(points=data)
            line_mapper = tvtk.PolyDataMapper(input_connection=line_source.output_port)
            p = tvtk.Property(line_width=2, color=(1, 0, 1))
            line_actor = tvtk.Actor(mapper=line_mapper, property=p)
            vessel_actor['trajectory'] = [points, line_source, line_actor]
            # Add actors to the scene
            self.actors[vessel.name] = vessel_actor
            self.viewer.visualization.scene3d.add_actor(self.actors[vessel.name]['bodyRF'][6])
            self.viewer.visualization.scene3d.add_actor(self.actors[vessel.name]['bodyRF'][7])
            self.viewer.visualization.scene3d.add_actor(self.actors[vessel.name]['bodyRF'][8])
            self.viewer.visualization.scene3d.add_actor(self.actors[vessel.name]['trajectory'][2])

    def sliderValueChange(self):
        value = self.slider.value()
        if self.system is not None:
            # DateTime
            text = 'DateTime : ' + str(self.system.timesteps[value].getDatetime()) + ', JulianDate : ' + str(self.system.timesteps[value].getJulianDate())
            self.actors['DateTime'][0].trait_set(text=text)
            # CelestialBodies
            for celestial_body in self.system.current.celestial_bodies.values():
                # Update base actor
                position = self.system.timesteps[value].celestial_bodies[celestial_body.name].getPosition()
                orientation = np.rad2deg(quaternion2euler(self.system.timesteps[value].celestial_bodies[celestial_body.name].getAttitude()))
                self.actors[celestial_body.name]['base'].trait_set(position=position, orientation=(orientation + np.array([0, 0, 180])))
                # Update bodyRF actors
                radius = self.system.timesteps[value].celestial_bodies[celestial_body.name].getRadius()
                position = self.system.timesteps[value].celestial_bodies[celestial_body.name].getPosition()
                i, j, k = self.system.timesteps[value].celestial_bodies[celestial_body.name].bodyRF.getIJK()
                i_points = np.append(np.expand_dims(position, axis=0), np.expand_dims(position + (i * radius * 1.5), axis=0), axis=0)
                j_points = np.append(np.expand_dims(position, axis=0), np.expand_dims(position + (j * radius * 1.5), axis=0), axis=0)
                k_points = np.append(np.expand_dims(position, axis=0), np.expand_dims(position + (k * radius * 1.5), axis=0), axis=0)
                self.actors[celestial_body.name]['bodyRF'][3].trait_set(points=i_points)
                self.actors[celestial_body.name]['bodyRF'][4].trait_set(points=j_points)
                self.actors[celestial_body.name]['bodyRF'][5].trait_set(points=k_points)
                # Update trajectory actor
                self.actors[celestial_body.name]['trajectory'][1].trait_set(points=self.actors[celestial_body.name]['trajectory'][0][:value])
            # Vessels
            for vessel in self.system.current.vessels.values():
                # Update bodyRF actors
                position = self.system.timesteps[value].vessels[vessel.name].getPosition()
                i, j, k = self.system.timesteps[value].vessels[vessel.name].bodyRF.getIJK()
                i_points = np.append(np.expand_dims(position, axis=0), np.expand_dims(position + (i * 100000), axis=0), axis=0)
                j_points = np.append(np.expand_dims(position, axis=0), np.expand_dims(position + (j * 100000), axis=0), axis=0)
                k_points = np.append(np.expand_dims(position, axis=0), np.expand_dims(position + (k * 100000), axis=0), axis=0)
                self.actors[vessel.name]['bodyRF'][3].trait_set(points=i_points)
                self.actors[vessel.name]['bodyRF'][4].trait_set(points=j_points)
                self.actors[vessel.name]['bodyRF'][5].trait_set(points=k_points)
                # Update trajectory actor
                self.actors[vessel.name]['trajectory'][1].trait_set(points=self.actors[vessel.name]['trajectory'][0][:value])
        self.viewer.visualization.scene3d.render()

def main():
    ground_reference_frame = ReferenceFrame()
    quadcopter_reference_frame = ReferenceFrame()

    fig = MainWidget()
    fig.plot_reference_frame(ground_reference_frame, 'ground')
    fig.plot_reference_frame(quadcopter_reference_frame, 'quadcopter')
    fig.rotate_reference_frame([np.radians(5),0,0], quadcopter_reference_frame, 'quadcopter', scale=0.5)
    fig.showMaximized()
    mlab.show()
    

if __name__ == "__main__":
    main()
