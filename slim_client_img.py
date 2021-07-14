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
            file_wildcard = 'Videos(*.mp4)|*.mp4|*.avi|*.mov|*.jpg|*.png'
            dlg = wx.FileDialog(self, "Open Video Or Picture ... ",
                                os.getcwd(),
                                style=wx.FD_OPEN | wx.FD_CHANGE_DIR,
                                wildcard=file_wildcard)

            if dlg.ShowModal() == wx.ID_OK:
                self.open_video(dlg.GetPath())

            dlg.Destroy()

    def InitUI(self):
        hbox = wx.BoxSizer(wx.HORIZONTAL)

        self.left_panel = wx.Panel(self, size=(300, 600))
        self.create_sliders(self.left_panel)
        self.create_buttons(self.left_panel)
        self.create_next_frame(self.left_panel)
        hbox.Add(self.left_panel, 1, wx.FIXED_MINSIZE)

        self.image_panel = ImagePanel(self, size=(800, 600))
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

        # for i in range(1):
        for i in range(3):
        #     sl = wx.Slider(parent, size=(100, -1), pos=(60, i * 100),
        #                    value=0, minValue=-50, maxValue=50,
        #                    style=wx.SL_HORIZONTAL | wx.SL_LABELS,
        #                    name=str(i))

            sl = wx.Slider(parent, size=(100, -1), pos=(60, i * 100),
                           value=0, minValue=-50, maxValue=50,
                           style=wx.SL_HORIZONTAL,
                           name=str(i))

        sl.Bind(wx.EVT_SCROLL_CHANGED, self.OnSliderScroll)

    def create_buttons(self, parent):
        self.eval_button = wx.Button(parent, -1, "RUN", pos=(15, 548))
        self.eval_button.Bind(wx.EVT_BUTTON, self.OnBtnClick)
        self.msg_text = wx.StaticText(parent, -1, label="", pos=(15, 580))

    def create_next_frame(self, parent):
        self.eval_button1 = wx.Button(parent, -1, "Next", pos=(110, 548))
        self.eval_button1.Bind(wx.EVT_BUTTON, self.get_next_frame)

    def OnSliderScroll(self, e):
        slider = e.GetEventObject()
        name = slider.GetName()
        val = slider.GetValue()

        self.slim_params[int(name)] = val

        print("!!!!!!", name, val. self.slim_params)
        self.apply_slim_action()

    def apply_slim_action(self):
        self.slim.set_slim_strength(
            cheek_strength=-3.0,
            humerus_strength=-0.2,
            chin_strength=2
        )
        # self.slim.update_pixel_list(self.cover)
        res = self.slim.slim_handler(self.cover)
        self.image_panel.draw_frame(res)

    def OnBtnClick(self, event):
        # if self.choose_frame:
        #     if self.video_capture:
        #         self.video_capture.release()
        #     self.video_capture = cv2.VideoCapture(self.video_path)
        st = threading.Thread(target=self.apply_img, args=(event, ))
        st.start()

    def apply_img(self, e):
        print("!!!!")
        print(self.video_path)
        save_dir = os.path.basename(self.video_path)
        save_dir = os.path.splitext(save_dir)[0]
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        im = cv2.imread(self.video_path)
        if im is None:
            print("None img")

        idx=0
        print("idx >>>>>>>  Go handle", im)

        res = self.slim.slim_handler(im)
        cv2.imwrite(os.path.join(save_dir, "%06d.png" % idx), res)
        # self.image_panel.draw_frame(res)
        wx.CallAfter(self.image_panel.draw_frame, res)
        # self.msg_text.SetLabel("%d/%d" % (idx, frame_count))
        # wx.CallAfter(self.msg_text.SetLabel, "%d/%d" % (idx, frame_count))
        gc.collect()



def main():
    app = wx.App()
    frame = MainFrame(None, size=(1000 + 200, 600), title='Slim Face')
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