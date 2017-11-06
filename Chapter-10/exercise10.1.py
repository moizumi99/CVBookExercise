from numpy import *
import cv2


def draw_flow(im, flow, step=16):
    """ Plot optical flow at sample points
    spaced step pixels apart. """

    h, w = im.shape[:2]
    y, x = mgrid[step/2:h:step, step/2:w:step].reshape(2, -1)
    fx, fy = flow[y, x].T

    # create line endpoints
    lines = vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = int32(lines)

    # create image and draw_flow
    vis = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for (x1, y1), (x2, y2) in lines:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def get_gesture(flow, step=16, thresh=8, nth=16):
    """ Calculate the chunk of motion. """
    
    h, w = flow.shape[:2]
    y, x = mgrid[step/2:h:step, step/2:w:step].reshape(2, -1)
    fx, fy = flow[y, x].T
    xp = sum(x>thresh for x in fx)
    xn = sum(x<-thresh for x in fx)
    yp = sum(y>thresh for y in fy)
    yn = sum(y<-thresh for y in fy)
    xd = 0
    if (xp>nth and xn<nth):
        xd = 1
    elif (xp<nth and xn>nth):
        xd = -1
    yd = 0
    if (yp>nth and yn<nth):
        yd = 1
    elif (yp<nth and yn>nth):
        yd = -1
    return xd, yd
    
# setup vide capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print "Camera is not opened"
    raise

ret, im = cap.read()
prev_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

while True:
    # get grayscale image
    ret, im = cap.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # compute flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prev_gray = gray

    # plot the flow vectors
    cv2.imshow('Optical flow', draw_flow(gray, flow))
    if cv2.waitKey(10) == 27:
        break

    xd, yd = get_gesture(flow, thresh=16, nth=32)
    str = ""
    if (xd==-1):
        str = "Left "
    elif (xd==1):
        str = "Right "
    if (yd==-1):
        str =  str + "Up"
    elif (yd==1):
        str = str + "Down"
    if not str == "":
        print str
