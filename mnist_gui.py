import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2
import threading
import numpy as np
from matplotlib import pyplot
from PIL import Image, ImageTk
import mnist_model_v34
import os

class main_window:

    def __init__(self):
        self.started = False
        self.detected = False
        self.img = None
        self.img_show = None
        self._create_wiget()


    def _create_wiget(self):
        self.mnist_model_emsembled = mnist_model_v34.mnist_model_emsembled()
        self.mnist_model_emsembled._restore()

        self.top = tk.Tk()
        self.top.title("Mnist demo")

        self.img_label = tk.Label(self.top)
        self.img_label.grid(row=0, column=0, columnspan=6)

        #self.canvas = tk.Canvas(self.img_label_frame, width=28, height=28)
        #self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW)
        #self.canvas.pack()

        self.path_var = tk.StringVar()
        self.path_var.set("")
        self.path_entry = tk.Entry(self.top, textvariable=self.path_var)
        self.path_entry.grid(row=1, column=0, columnspan=3)

        self.select_button = tk.Button(self.top, text="Select a photo", command=self._askfile)
        self.select_button.grid(row=1, column=3)

        self.start_take_button = tk.Button(self.top, text="Start Camera", command=self._start_camera)
        self.start_take_button.grid(row=1, column=4)

        self.detect_button = tk.Button(self.top, text="Detect", command=self._detect)
        self.detect_button.grid(row=2, column=2)

        self.result = tk.StringVar()
        self.result.set("Click detect to classify.")
        self.result_label = tk.Label(self.top, textvariable=self.result)
        self.result_label.grid(row=2, column=3)

        self.gamma = tk.DoubleVar()
        self.gamma.set(2.)
        self.scale = tk.Scale(self.top, from_=2., to=25, orient=tk.HORIZONTAL, variable=self.gamma, resolution=0.1)
        self.scale.grid(row=3, column=1)

        self.vis_button = tk.Button(self.top, text="Visualize activation", command=self._show_vis)
        self.vis_button.grid(row=2, column=4)

        self.top.mainloop()

    def _askfile(self):
        filename = filedialog.askopenfilename(title="Select a photo")
        self.path_var.set(filename)
        if(not self.path_var.get().endswith("jpg")):
            messagebox.showerror(title="Bad file", message="Please select a jpg image!")
            self.detected = False
        else:
            self.img = Image.open(self.path_var.get())
            self.img_show = ImageTk.PhotoImage(self.img)
            # must be set to class member, or it will be clean up and cannot be displayed!
            self.img_label.configure(image=self.img_show)

    def _scale_change(self, env=None):
        if self.img:
            self.img_show = np.array(self.img)
            self.img_show = self._augment_contrast(self.img_show)
            self.img_show = ImageTk.PhotoImage(Image.fromarray(self.img_show))
            self.img_label.configure(image=self.img_show)


    def _start_camera(self):
       if(self.started is None or self.started==False):
            record = Record()
            record.start()

    def _detect(self):
        if(not self.path_var.get().endswith("jpg")):
            messagebox.showerror("Bad file!", message="Please select a jpeg image file!")
        else:
            self.result.set("Detecting...")
            img = cv2.imread(self.path_var.get())
            img = self._preprocess(img, False)
            img = self._augment_contrast(img)
            pyplot.imshow(img)
            pyplot.show()
            img = img.reshape(1, 28, 28, 1)
            pred = self.mnist_model_emsembled.predict(img)
            res = np.argmax(pred)
            self.result.set("This number is " + str(res))
            self.mnist_model_emsembled.visualize_activation(img)
            self.detected = True

    def _show_vis(self):
        def get_all_pngs(dir):
            filenames_png = list()
            for dirpath, dirnames, filenames in os.walk(dir):
                for filename in filenames:
                    if filename.endswith(".png"):
                        filenames_png.append(os.path.join(dirpath, filename))
                for dirname in dirnames:
                    filenames_png = filenames_png + get_all_pngs(dirname)
            return filenames_png

        base_dir = r"C:\Users\Phree\PycharmProjects\mnist_demo\Output"
        filenames_png = get_all_pngs(base_dir)
        for filename in filenames_png:
            print(filename)
            cv2.namedWindow(filename)
            img = cv2.imread(filename)
            cv2.imshow(filename, img)

    def _preprocess(self, img, is_neg=True):
        """
        img must be a cv2 type image
        :param is_neg: whether to substracted by 255
        :return: a numpy array with rank 4
        """
        img = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.array(img)
        img = 255 - img
        return img

    def _augment_contrast(self, img):
        img_max = np.max(img)
        img = img/img_max
        img = np.power(img, self.gamma.get())
        img = img * 255
        return img


#    def _take_photo(self):



class Record(threading.Thread):
    def __init__(self):
        self.frame = None
        self.img_show = None
        self.done = False
        self.drawing = False
        self.x, self.y, self.x_e, self.y_e = (None, None, None, None)
        threading.Thread.__init__(self)




    def run(self):
        cap = cv2.VideoCapture(1)
        if cap.isOpened():
            while True:
                ret, self.frame = cap.read()
                if ret:
                    cv2.imshow("Capture", self.frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.img_show = np.copy(self.frame)

                    def _draw_triangle(event, x, y, flags, param):
                        if event == cv2.EVENT_LBUTTONDOWN:
                            self.x, self.y = x, y
                            self.drawing = True
                        elif event == cv2.EVENT_MOUSEMOVE:
                            self.img_show = np.copy(self.frame)
                            if self.drawing:
                                self.img_show[0:self.y, :, :] = 0
                                self.img_show[y:-1, :, :] = 0
                                self.img_show[:, 0:self.x, :] = 0
                                self.img_show[:, x:-1, :] = 0
                        elif event == cv2.EVENT_LBUTTONUP:
                            self.drawing = False
                            self.x_e, self.y_e = x, y
                            self.done = True

                    cv2.setMouseCallback("Capture", _draw_triangle)
                    while True:
                        cv2.imshow("Capture", self.img_show)
                        k = cv2.waitKey(1) & 0xFF
                        if k==27:
                            break
                        if self.done:
                            img = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
                            img = img[self.y:self.y_e, self.x:self.x_e, :]
                            img = Image.fromarray(img)
                            img.save("capture.jpg")
                            break
                    break
            cap.release()
            cv2.destroyAllWindows()



if __name__ == "__main__":
    app = main_window()
