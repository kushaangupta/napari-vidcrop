import napari
import napari_vidcrop as nv

ui = nv.VideoCrop()
viewer = napari.Viewer()
viewer.window.add_dock_widget(ui)

napari.run()
