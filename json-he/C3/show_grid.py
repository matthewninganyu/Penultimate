"""Mac-side fullscreen numbered calibration dot grid.

Run on the LAPTOP whose screen is being calibrated (the tap surface):
    python3 show_grid.py

Shows the same dots (same order) capture_calib.py expects. Tap them IN ORDER
with the lit stylus; the Pi auto-captures each. Press Esc/q to close.
Uses tkinter (stdlib) so it needs no extra install and reports true screen size.
"""
import tkinter as tk

from calib_grid import grid_points

DOT_R = 14
FONT = ("Helvetica", 16, "bold")


def main():
    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.configure(bg="black")
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    cv = tk.Canvas(root, width=sw, height=sh, bg="black", highlightthickness=0)
    cv.pack()

    for i, (xn, yn) in enumerate(grid_points()):
        x, y = xn * sw, yn * sh
        cv.create_oval(x - DOT_R, y - DOT_R, x + DOT_R, y + DOT_R,
                       fill="white", outline="")
        cv.create_text(x + DOT_R + 10, y, text=str(i + 1),
                       fill="#888", font=FONT, anchor="w")

    root.bind("<Escape>", lambda e: root.destroy())
    root.bind("q", lambda e: root.destroy())
    print(f"grid on {sw}x{sh}. Tap dots 1..{len(grid_points())} in order. Esc to quit.")
    root.mainloop()


if __name__ == "__main__":
    main()
