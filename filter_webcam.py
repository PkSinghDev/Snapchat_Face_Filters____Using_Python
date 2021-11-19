import cv2
import numpy as np

cap = cv2.VideoCapture(0)
eyes_cascade = cv2.CascadeClassifier("./third-party/frontalEyes35x16.xml")
nose_cascade = cv2.CascadeClassifier("./third-party/Nose18x15.xml")


def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:,:,:3] # Grab the BRG planes
    overlay_mask = overlay_t_img[:,:,3:]  # And the alpha plane
#     plt.imshow(overlay_mask)
#     plt.show()
    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask
#     plt.imshow(background_mask)
#     plt.show()
    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)
#     plt.imshow(overlay_mask)
#     plt.show()
#     plt.imshow(background_mask)
#     plt.show()

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
#     plt.imshow(face_part)
#     plt.show()
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))
#     plt.imshow(overlay_part)
#     plt.show()
    # And finally just add them together, and rescale it back to an 8bit integer image    
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


while True:
    ret, frame = cap.read()
    if ret == False:
        continue

    eyes = eyes_cascade.detectMultiScale(frame, 1.3, 5)
    if len(eyes) == 0: 
        continue
    eyes = sorted(eyes, key = lambda e: e[2] * e[3])
    x, y, w, h = eyes[-1]
    nose = nose_cascade.detectMultiScale(frame, 1.3, 5)
    if len(nose) == 0:
        continue
    nose = sorted(nose, key = lambda n: n[2] * n[3])
    x1, y1, w1, h1 = nose[-1]
    glasses = cv2.imread("./Train/glasses.png", cv2.IMREAD_UNCHANGED)
    glasses = cv2.resize(glasses, (w, h))
    mustache = cv2.imread("./Train/mustache.png", cv2.IMREAD_UNCHANGED)
    mustache = cv2.resize(mustache, (w1, h1))
    face_glass = blend_transparent(frame[y : y + h, x : x + w], glasses)
    frame[y: y + h, x : x + w] = face_glass
    mustache_blend = blend_transparent(frame[y1 + int(h1/2) : y1 + int(h1/2) + h1, x1 : x1 + w1], mustache)
    frame[y1 + int(h1/2) : y1 + int(h1/2) + h1, x1 : x1 + w1] = mustache_blend
    cv2.imshow("Video Frame", frame)

    # Wait for user input q then you will stop the loop
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()