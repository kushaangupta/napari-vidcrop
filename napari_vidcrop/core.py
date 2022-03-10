import os

import napari
import numpy as np

from magicclass import field, magicclass
from magicgui.widgets import ComboBox
from napari_video.napari_video import VideoReaderNP
from skimage import (
    color,
    filters,
    img_as_ubyte,
    measure,
    segmentation,
)


THRESHOLD_METHODS = ('isodata', 'li', 'mean', 'minimum',
                     'otsu', 'triangle', 'yen')


def _decode_fourcc(cc):
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])


def _largest_label_dims(labels, scale_factor):
    regions = measure.regionprops(labels)
    regions = sorted(regions, key=lambda region: region.area)
    miny, minx, maxy, maxx = regions[-1].bbox
    height, width = (maxy-miny, maxx-minx)

    miny = miny - int(scale_factor * .5 * height)
    minx = minx - int(scale_factor * .5 * width)
    maxy = maxy + int(scale_factor * .5 * height)
    maxx = maxx + int(scale_factor * .5 * width)
    height, width = (maxy-miny, maxx-minx)

    return (miny, minx, maxy, maxx), (height, width)


def _make_ffmpeg_cmd(
        in_dir, name, start_time, end_time, width, height,
        minx, miny, vidcodec, crf, fps, out_dir, extension
        ):
    cmd = f'ffmpeg -i "{os.path.join(in_dir, name)}" ' \
          f'-ss {start_time} -to {end_time} ' \
          f'-vf "crop={width}:{height}:{minx}:{miny}" ' \
          f'-c:v {vidcodec} -crf {crf} ' \
          f'-r {fps} -c:a copy "' \
          f'{os.path.join(out_dir, os.path.splitext(name)[0])}.{extension}"'

    return cmd

@magicclass(widget_type="scrollable")
class VideoCrop:
    IN_DIR = OUT_DIR = FILES = None
    vr = vid_width = vid_height = minx = miny = None
    start_keyframe = end_keyframe = 0

    @magicclass(widget_type="groupbox")
    class LoadDataset:
        @magicclass(widget_type="none")
        class LoadDIR:
            IN_DIR = field(str)
            OUT_DIR = field('./')

            def load_directory(self):
                FILES = os.listdir(self.IN_DIR.value)

                # ref to parent
                parent = self.__magicclass_parent__
                grandparent = parent.__magicclass_parent__
                grandparent.FILES = FILES
                grandparent.IN_DIR = self.IN_DIR.value
                grandparent.OUT_DIR = self.OUT_DIR.value

                parent.LoadVideo.VIDNUM.range = (0, len(FILES)-1)
                parent.LoadVideo._update_file()

        @magicclass(widget_type="none")
        class LoadVideo:
            VIDNUM = field(0)
            FILE = field(str)

            @VIDNUM.connect
            def _update_file(self):
                grandparent = self.__magicclass_parent__.__magicclass_parent__

                if grandparent.FILES is not None:
                    FILES = grandparent.FILES
                    self.FILE.value = FILES[self.VIDNUM.value]

            def load_video(self):
                parent = self.__magicclass_parent__
                grandparent = parent.__magicclass_parent__
                FILES = grandparent.FILES
                FILE = FILES[self.VIDNUM.value]
                vr = VideoReaderNP(os.path.join(parent.LoadDIR.IN_DIR.value, FILE))

                grandparent.vr = vr
                grandparent.vid_height = vr.shape[1]
                grandparent.vid_width = vr.shape[2]
                grandparent.minx = 0
                grandparent.miny = 0
                grandparent.end_keyframe = vr.number_of_frames-1
                grandparent.Trim.start_keyframe.range = (0, vr.number_of_frames-1)
                grandparent.Trim.end_keyframe.range = (0, vr.number_of_frames-1)
                grandparent.Trim.end_keyframe.value = vr.number_of_frames-1
                grandparent.Export.out_fps.value = int(vr.frame_rate)
                grandparent.Export.out_vidcodec.value = _decode_fourcc(vr.fourcc)

                self.parent_viewer.add_image(vr, name=FILE)

    @magicclass(widget_type="tabbed")
    class Crop:  # Region crop
        @magicclass(widget_type="none")
        class CropBox:
            def confirm_ROI(self):
                MazeLayer = self.parent_viewer.layers['Shapes'].data
                Maze = np.round(MazeLayer[0])
                ycoords, xcoords = Maze[:, -2].astype(int), Maze[:, -1].astype(int)
                minx, maxx = min(xcoords), max(xcoords)
                miny, maxy = min(ycoords), max(ycoords)

                parent = self.__magicclass_parent__
                parent.__magicclass_parent__.vid_width = maxx - minx
                parent.__magicclass_parent__.vid_height = maxy - miny
                parent.__magicclass_parent__.minx = minx
                parent.__magicclass_parent__.miny = miny

        @magicclass(widget_type="none")
        class FloodFillSegmentation:
            SCALE_FACTOR = field(.15)
            TOLERANCE = field(25)

            def segment(self):
                grandparent = self.__magicclass_parent__.__magicclass_parent__
                SCALE = self.SCALE_FACTOR.value
                TOL = self.TOLERANCE.value
                first_point_yx = self.parent_viewer.layers['Points'].data[0][-2:]
                COORDS = tuple(first_point_yx.astype(int))

                current_frame = self.parent_viewer.dims.current_step[0]
                frame = grandparent.vr[current_frame]
                if frame.ndim == 3:
                    frame = color.rgb2gray(frame)
                frame = img_as_ubyte(frame)

                flood_filled = segmentation.flood(frame, COORDS, tolerance=TOL)
                labels = measure.label(flood_filled)
                self.parent_viewer.add_labels(labels)

                bbox, (height, width) = _largest_label_dims(labels, SCALE)
                miny, minx, maxy, maxx = bbox

                self.parent_viewer.add_image(frame[miny:maxy, minx:maxx])

                grandparent.vid_width = maxx - minx
                grandparent.vid_height = maxy - miny
                grandparent.minx = minx
                grandparent.miny = miny

        @magicclass(widget_type="none")
        class ThresholdSegmentation:
            single_autothresh = ComboBox(choices=THRESHOLD_METHODS)
            SCALE_FACTOR = field(.10)

            def segment_largest(self):
                grandparent = self.__magicclass_parent__.__magicclass_parent__
                SCALE = self.SCALE_FACTOR.value

                current_frame = self.parent_viewer.dims.current_step[0]
                frame = grandparent.vr[current_frame]
                if frame.ndim == 3:
                    frame = color.rgb2gray(frame)
                frame = img_as_ubyte(frame)

                th_method = self.single_autothresh.value
                th = eval(f'filters.threshold_{th_method}(frame)')
                imbinary = frame > th
                labels = measure.label(imbinary)
                self.parent_viewer.add_labels(labels)

                # select largest label
                bbox, (height, width) = _largest_label_dims(labels, SCALE)
                miny, minx, maxy, maxx = bbox

                self.parent_viewer.add_image(frame[miny:maxy, minx:maxx])

                grandparent.vid_width = maxx - minx
                grandparent.vid_height = maxy - miny
                grandparent.minx = minx
                grandparent.miny = miny

    @magicclass(widget_type="groupbox")
    class Trim:  # Temporal Cropping (Trim)
        start_keyframe = field(0)
        end_keyframe = field(0)

        @start_keyframe.connect
        def _set_start_keyframe(self):
            keyframe = self.start_keyframe.value
            self.__magicclass_parent__.start_keyframe = keyframe
            self.start_keyframe.value = keyframe

        @end_keyframe.connect
        def _set_end_keyframe(self):
            keyframe = self.end_keyframe.value
            self.__magicclass_parent__.end_keyframe = keyframe
            self.end_keyframe.value = keyframe

        def select_start_keyframe(self):
            current_frame = self.parent_viewer.dims.current_step[0]
            self.__magicclass_parent__.start_keyframe = current_frame
            self.start_keyframe.value = current_frame

        def select_end_keyframe(self):
            current_frame = self.parent_viewer.dims.current_step[0]
            self.__magicclass_parent__.end_keyframe = current_frame
            self.end_keyframe.value = current_frame

    @magicclass(widget_type="groupbox")
    class Export:
        crf = field(23)
        out_fps = field(30)
        out_vidcodec = field('libx264')
        out_extension = field('mp4')

        def convert_current(self):
            parent = self.__magicclass_parent__
            start_timept = np.round(parent.start_keyframe / parent.vr.frame_rate)
            end_timept = np.round(parent.end_keyframe / parent.vr.frame_rate)
            F = parent.FILES[parent.LoadDataset.LoadVideo.VIDNUM.value]

            cmd = _make_ffmpeg_cmd(
                parent.IN_DIR, F, start_timept, end_timept, parent.vid_width,
                parent.vid_height, parent.minx, parent.miny, self.out_vidcodec.value,
                self.crf.value, self.out_fps.value, parent.OUT_DIR, self.out_extension.value
                )

            try:
                os.system(cmd)
            except Exception as err:
                print("Error:", F, ": ", err)

        def batch_convert(self):
            parent = self.__magicclass_parent__
            out_vidcodec = self.out_vidcodec.value
            out_fps = self.out_fps.value

            for i, F in enumerate(parent.FILES):
                vr = VideoReaderNP(os.path.join(parent.IN_DIR, F))
                start_timept = np.round(parent.start_keyframe / vr.frame_rate)
                end_timept = np.round(vr.number_of_frames-1 / vr.frame_rate)

                if parent.Crop.current_index == 1:  # FloodFill
                    layer_names = [layer.name for layer in self.parent_viewer.layers]
                    for layer_name in layer_names:
                        if layer_name != 'Points':
                            self.parent_viewer.layers.remove(layer_name)
                    parent.LoadDataset.LoadVideo.VIDNUM.value = i
                    parent.LoadDataset.LoadVideo.load_video()

                    # apply same settings
                    self.out_vidcodec.value = out_vidcodec
                    self.out_fps.value = out_fps
                    parent.Crop.FloodFillSegmentation.segment()

                if parent.Crop.current_index == 2:  # Threshold
                    self.parent_viewer.layers.clear()
                    parent.LoadDataset.LoadVideo.VIDNUM.value = i
                    parent.LoadDataset.LoadVideo.load_video()

                    # apply same settings
                    self.out_vidcodec.value = out_vidcodec
                    self.out_fps.value = out_fps
                    parent.Crop.ThresholdSegmentation.segment_largest()

                cmd = _make_ffmpeg_cmd(
                    parent.IN_DIR, F, start_timept, end_timept, parent.vid_width,
                    parent.vid_height, parent.minx, parent.miny, self.out_vidcodec.value,
                    self.crf.value, self.out_fps.value, parent.OUT_DIR, self.out_extension.value
                    )

                try:
                    os.system(cmd)
                except Exception as err:
                    print("Error:", F, ": ", err)
