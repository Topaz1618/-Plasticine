# -*- coding: utf-8 -*-
import os
import threading
import gc
from time import time, sleep

import numpy as np
import cv2
import wx

from slim_face import SlimFace

global_fps = 24

os.environ["UBUNTU_MENUPROXY"]="0"


class ImagePanel(wx.Panel):
    def __init__(self, parent, size):
        super(ImagePanel, self).__init__(parent, size=size)
        self.SetDoubleBuffered(True)

        self.size = size
        im = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        self.bmp = wx.BitmapFromBuffer(self.size[0], self.size[1], im)
        self.Bind(wx.EVT_PAINT, self.on_paint)

    def on_paint(self, evt):
        dc = wx.BufferedPaintDC(self)
        dc.DrawBitmap(self.bmp, 0, 0)

    def draw_frame(self, im):
        im_result = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_result = cv2.resize(im_result, self.size)
        self.bmp.CopyFromBuffer(im_result)
        self.Refresh()


class MainFrame(wx.Frame):
    def __init__(self, parent, size, title=''):
        super(MainFrame, self).__init__(parent, size=size, title=title)

        self.CreateMenu()
        self.InitUI()

        self.video_capture = None
        self.slim = SlimFace()
        self.slim_params = [0, 0, 0, 0, 0, 0]

        self.cover = None
        self.choose_frame = False

    def CreateMenu(self):

        menu_bar = wx.MenuBar()

        menu = wx.Menu()

        item_open = wx.MenuItem(menu, id=wx.ID_OPEN, text='open',
                                kind=wx.ITEM_NORMAL)
        menu.Append(item_open)
        menu.AppendSeparator()
        menu_bar.Append(menu, title='File')
        self.SetMenuBar(menu_bar)
        self.Bind(wx.EVT_MENU, self.menu_handler)

    def menu_handler(self, event):

        id = event.GetId()

        if id == wx.ID_OPEN:
            file_wildcard = 'Videos(*.mp4)|*.mp4|*.avi|*.mov'
            dlg = wx.FileDialog(self, "Open Video...",
                                os.getcwd(),
                                style=wx.FD_OPEN | wx.FD_CHANGE_DIR,
                                wildcard=file_wildcard)

            if dlg.ShowModal() == wx.ID_OK:
                self.open_video(dlg.GetPath())

            dlg.Destroy()

    def InitUI(self):
        hbox = wx.BoxSizer(wx.HORIZONTAL)

        self.left_panel = wx.Panel(self, size=(200, 450))
        self.create_sliders(self.left_panel)
        self.create_buttons(self.left_panel)
        self.create_next_frame(self.left_panel)
        hbox.Add(self.left_panel, 1, wx.FIXED_MINSIZE)

        self.image_panel = ImagePanel(self, size=(800, 450))
        hbox.Add(self.image_panel, 1, wx.EXPAND)

        self.SetSizer(hbox)

    def open_video(self, path):
        self.video_path = path
        if self.video_capture:
            self.video_capture.release()
        self.video_path = path
        self.video_capture = cv2.VideoCapture(path)


        _, im = self.video_capture.read()
        self.cover = im.copy()
        self.image_panel.draw_frame(self.cover)

    def get_next_frame(self, x):
        self.choose_frame = True
        _, im = self.video_capture.read()
        self.cover = im.copy()
        self.image_panel.draw_frame(self.cover)

    def create_sliders(self, parent):
        # from ipdb import set_trace; set_trace()

        for i in range(6):
            sl = wx.Slider(parent, size=(100, -1), pos=(20, i * 100),
                           value=0, minValue=-50, maxValue=50,
                           style=wx.SL_HORIZONTAL | wx.SL_LABELS,
                           name=str(i))
            sl.Bind(wx.EVT_SCROLL_CHANGED, self.OnSliderScroll)

    def create_buttons(self, parent):
        self.eval_button = wx.Button(parent, -1, "RUN", pos=(15, 400))
        self.eval_button.Bind(wx.EVT_BUTTON, self.OnBtnClick)
        self.msg_text = wx.StaticText(parent, -1, label="", pos=(15, 440))

    def create_next_frame(self, parent):
        self.eval_button1 = wx.Button(parent, -1, "Next", pos=(110, 400))
        self.eval_button1.Bind(wx.EVT_BUTTON, self.get_next_frame)

    def OnSliderScroll(self, e):
        slider = e.GetEventObject()
        name = slider.GetName()
        val = slider.GetValue()

        self.slim_params[int(name)] = val
        self.apply_slim_action()

    def apply_slim_action(self):
        self.slim.set_slim_strength(cheek_strength=self.slim_params[0] / 100 * 4,
                                    humerus_strength=self.slim_params[1] / 100 * 0.4,
                                    chin_strength=self.slim_params[2] / 100 * 3,
                                    forehead_strength=self.slim_params[3] / 100 * 1.5,
                                    pull_chin_strength=self.slim_params[4] / 100 * 1,
                                    pull_forehead_strength=self.slim_params[5] / 100 * 2.5,
                                    )
        self.slim.update_pixel_list(self.cover)
        res = self.slim.slim_handler(self.cover)
        self.image_panel.draw_frame(res)

    def OnBtnClick(self, event):
        if self.choose_frame:
            if self.video_capture:
                self.video_capture.release()
            self.video_capture = cv2.VideoCapture(self.video_path)
        st = threading.Thread(target=self.apply_video, args=(event, ))
        st.start()

    def apply_video(self, e):
        save_dir = os.path.basename(self.video_path)
        save_dir = os.path.splitext(save_dir)[0]
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        frame_count = self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        idx = 0
        while True:
            _, im = self.video_capture.read()
            if im is None:
                break

            print("idx >>>>>>> ", idx)
            res = self.slim.slim_handler(im)
            cv2.imwrite(os.path.join(save_dir, "%06d.png" % idx), res)

            # self.image_panel.draw_frame(res)
            wx.CallAfter(self.image_panel.draw_frame, res)
            # self.msg_text.SetLabel("%d/%d" % (idx, frame_count))
            wx.CallAfter(self.msg_text.SetLabel, "%d/%d" % (idx, frame_count))
            gc.collect()
            idx += 1


def main():
    app = wx.App()
    frame = MainFrame(None, size=(800 + 200, 450), title='Slim Face')
    frame.Centre()
    frame.Show()

    app.MainLoop()


if __name__ == '__main__':
    main()
    # slim = SlimFace()

    # im = cv2.imread('./input/捕获.PNG')
    # sp = time()
    # slim.set_slim_strength(cheek_strength=2, humerus_strength=2, chin_strength=3)
    # res = slim.slim_handler(im)
    # print(time() - sp)
    # cv2.imwrite('./xx.jpg', res)