import argparse
import threading
from tkinter import *
from pylsl import StreamInlet, resolve_stream

"""
Author: S.LEGEAY, intern at LaSEEB
e-mail: legeay.simon.sup@gmail.com


Create a Marker stream of a scenario described in a xml file specified in parser
Launch: python graz.py config.xml -v
"""

MARKERS = {
    'show_cross': 786,
    'show_rigth_arrow': 770,
    'show_left_arrow': 769,
    'hide_arrow': 781,
    'hide_cross': 800,
    'exit_': 1010,
}


def wait_marker_stream(verbose):
    if verbose:
        print('Waiting for Marker Stream')
    streams = resolve_stream('type', 'Markers')

    # open an inlet so we can read the stream's data (and meta-data)
    inlet = StreamInlet(streams[0])

    # get the full stream info (including custom meta-data) and dissect it
    info = inlet.info()
    if verbose:
        print("The stream's XML meta-data is: ")
        print(info.as_xml())
    return inlet


def get_inlet():
    streams = resolve_stream('type', 'Markers')

    # open an inlet so we can read the stream's data (and meta-data)
    inlet = StreamInlet(streams[0])
    return inlet


class CustomCanvas(Canvas):
    """Custom Canvas with function to update the content
    according the windows size, and plot functions"""

    def __init__(self, parent, **kwargs):
        Canvas.__init__(self, parent, **kwargs)
        self.pack(fill=BOTH, expand=1)
        self.bind("<Configure>", self.on_resize)

        self.height = self.winfo_reqheight()
        self.width = self.winfo_reqwidth()

    def show_cross(self):
        """Show cross on the window"""
        x1 = int(self.width / 3)
        x2 = int(self.width * 2 / 3)
        x3 = int(self.width / 2)
        y1 = int(self.height / 2)
        y2 = int(self.height / 3)
        y3 = int(self.height * 2 / 3)
        self.create_line(x1, y1, x2, y1,
                         fill="white", width=1, tags='cross1')
        self.create_line(x3, y2, x3, y3,
                         fill="white", width=1, tags='cross2')

    def show_left_arrow(self):
        """Show red arrow on the window"""
        x3 = int(self.width / 2)
        x4 = int(self.width * 7 / 18)
        y1 = int(self.height / 2)
        w = int(self.width / 25)
        s = f"{w} {w} {int(0.7 * w)}"
        self.create_line(x4, y1, x3, y1,
                         fill="red", width=w, tags='ra',
                         arrow='first', arrowshape=s)

    def show_rigth_arrow(self):
        """Show red arrow on the window"""
        x3 = int(self.width / 2)
        x5 = int(self.width * 11 / 18)
        y1 = int(self.height / 2)
        w = int(self.width / 25)
        s = f"{w} {w} {int(0.7 * w)}"
        self.create_line(x5, y1, x3, y1,
                         fill="red", width=w, tags='la',
                         arrow='first', arrowshape=s)

    def hide_arrow(self):
        """Hide all arrows on the window"""
        self.delete('la')
        self.delete('ra')

    def hide_cross(self):
        """Hide the cross on the window"""
        self.delete('cross1')
        self.delete('cross2')

    def on_resize(self, event):
        # determine the ratio of old width/height to new width/height
        wscale = float(event.width) / self.width
        hscale = float(event.height) / self.height
        self.width = event.width
        self.height = event.height
        # resize the canvas
        self.config(width=self.width, height=self.height)
        # rescale all the objects tagged with the "all" tag
        self.scale("all", 0, 0, wscale, hscale)


def app(verbose):
    # initialise a window.
    root = Tk()
    root.config(background='black')
    root.title('Graz Visualization')
    root.geometry("500x350")
    wait_marker_stream(verbose)

    # create a Frame
    myframe = Frame(root)
    myframe.pack(fill=BOTH, expand=YES)

    # add a canvas
    myCanvas = CustomCanvas(myframe, width=425, height=2,
                            bg="black", highlightthickness=0)
    myCanvas.pack(fill=BOTH, expand=YES)

    def plotter():
        inlet = get_inlet()
        bool_ = True
        while bool_:
            sample, timestamp = inlet.pull_sample()
            if verbose:
                print("got %s at time %s" % (sample[0], timestamp))
            if sample[0] == MARKERS['show_cross']:
                myCanvas.show_cross()
            elif sample[0] == MARKERS['show_rigth_arrow']:
                myCanvas.show_rigth_arrow()
            elif sample[0] == MARKERS['show_left_arrow']:
                myCanvas.show_left_arrow()
            elif sample[0] == MARKERS['hide_arrow']:
                myCanvas.hide_arrow()
            elif sample[0] == MARKERS['hide_cross']:
                myCanvas.hide_cross()
            elif sample[0] == MARKERS['exit_']:
                inlet.close_stream()
                print('Scenario is finished, close the window please')
                root.destroy()

    # add a thread to update the window
    threading.Thread(target=plotter).start()
    root.mainloop()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Launch visualization Graz in a new window")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode")
    args = parser.parse_args()

    app(args.verbose)
