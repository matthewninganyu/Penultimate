def camera_to_screen(cam_x, cam_y, cam_tl, cam_br, screen_w, screen_h):
    x1, y1 = cam_tl
    x2, y2 = cam_br

    screen_x = (cam_x - x1) / (x2 - x1) * screen_w
    screen_y = (cam_y - y1) / (y2 - y1) * screen_h

    return screen_x, screen_y
