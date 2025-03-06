# coding: utf-8

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

from PIL import Image, ImageTk

_image_extension = [
    "png",
    "jpg",
    "jpeg",
    "bmp",
    "gif",
]
_show_size = (1280, 1280)


class RectangleDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("Rectangle Drawer")

        button_area = tk.Frame(root)
        button_area.pack()
        self.load_button = tk.Button(button_area, text="載入圖片", command=self.load_image)
        self.load_button.pack(side=tk.LEFT)

        self.clear_button = tk.Button(button_area, text="清除圖片", command=self.reset_image)
        self.clear_button.pack(side=tk.LEFT)

        self.previous_step_button = tk.Button(button_area, text="退回前一步", command=self.previous_step)
        self.previous_step_button.pack(side=tk.LEFT)

        self.save_button = tk.Button(button_area, text="儲存標記", command=self.save_label)
        self.save_button.pack(side=tk.LEFT)

        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.rect_id = None
        self.start_x = None
        self.start_y = None
        self.rectangles = []

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.image_on_canvas = None

    def reset_image(self):
        self.canvas.config(width=self.tk_image.width(), height=self.tk_image.height())
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.rect_id = None
        self.start_x = None
        self.start_y = None

    def previous_step(self):
        self.reset_image()
        self.rectangles.pop()
        for rectangle in self.rectangles:
            rect_id = self.canvas.create_rectangle(*rectangle, outline="red", width=2)

    def save_label(self):
        data = list()
        for rectangle in self.rectangles:
            data.append(
                {
                    "x1": int(rectangle[0]),
                    "x2": int(rectangle[2]),
                    "y1": int(rectangle[1]),
                    "y2": int(rectangle[3]),
                    "range": [1, 1000],
                    "angle": False,
                    "choices": [],
                    "before_choices": [],
                    "after_choices": [],
                }
            )

        output_rootpath = Path(self.image_path).parent
        output_filename = Path(self.image_path).stem
        output_path = Path(output_rootpath, f"{str(output_filename)}.txt")
        with output_path.open(mode="w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "rotate_angle": 0.0,
                        "roles": data,
                    },
                    indent=4,
                    ensure_ascii=True,
                )
            )
        print(f"Success save to {output_path!s}")

    def load_image(self):
        image_path = filedialog.askopenfilename(
            title="Select an Image File",
            filetypes=[
                (
                    "Image Files",
                    " ".join([f"*.{e.lower()} *.{e.upper()}" for e in _image_extension]),
                ),
                ("All Files", "*.*"),
            ],
        )
        if not image_path:
            return

        try:
            self.image_path = image_path
            self.image = Image.open(image_path)
            self.image_scale = min(_show_size[0] / self.image.size[0], _show_size[1] / self.image.size[1])
            self.tk_image = ImageTk.PhotoImage(
                self.image.resize(
                    size=(
                        int(self.image.size[0] * self.image_scale),
                        int(self.image.size[1] * self.image_scale),
                    )
                )
            )

            self.reset_image()
            self.rectangles = []
        except Exception as e:
            print(f"Failed to load image: {e}")

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2
        )

    def on_mouse_drag(self, event):
        if self.rect_id:
            self.canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)

    def on_button_release(self, event):
        end_x, end_y = event.x, event.y
        end_x = self.image.width if end_x > self.image.width else max(0, end_x)
        end_y = self.image.height if end_y > self.image.height else max(0, end_y)
        if self.rect_id:
            _start_x = (self.start_x if self.start_x < end_x else end_x) / self.image_scale
            _end_x = (self.start_x if self.start_x > end_x else end_x) / self.image_scale
            _start_y = (self.start_y if self.start_y < end_y else end_y) / self.image_scale
            _end_y = (self.start_y if self.start_y > end_y else end_y) / self.image_scale
            self.rectangles.append((_start_x, _start_y, _end_x, _end_y))
            print(f"Rectangle coordinates: ({_start_x}, {_start_y}, {_end_x}, {_end_y})")
            self.rect_id = None


if __name__ == "__main__":
    root = tk.Tk()
    app = RectangleDrawer(root)
    root.mainloop()
