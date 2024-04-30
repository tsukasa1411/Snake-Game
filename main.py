import enum
import math
import random
import cvzone
import cv2 as cv
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# object for detecting hand
detector = HandDetector(detectionCon=0.8, maxHands=1)


class snakegame:
    def __init__(self):
        self.points = []  # points of the snake
        self.lengths = []  # distance between the points
        self.currentLength = 0  # totalength of the snake
        self.allowedLength = 150  # total allowed Length
        self.previousHead = 0, 0  # previous head point
        self.score = 0  # points scored by the player
        self.gameover = False

        self.imgFood = cv.imread(
            "/Users/tsukasa/python/cvproj/red_ball.png", cv.IMREAD_UNCHANGED)
        self.imgFood = resized_image = cv.resize(
            self.imgFood, (70, 70), interpolation=cv.INTER_LINEAR)

        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodpos = 0, 0
        self.randomFoodLocation()

    def randomFoodLocation(self):
        self.foodpos = random.randint(100, 900), random.randint(100, 500)

    def update(self, imgMain, currentHead):
        px, py = self.previousHead
        cx, cy = currentHead

        self.points.append([cx, cy])
        distance = math.hypot(cx-px, cy-py)
        self.lengths.append(distance)
        self.currentLength += distance
        self.previousHead = cx, cy

        # length reduction
        if self.currentLength > self.allowedLength:
            toremove = 0
            for i, length in enumerate(self.lengths):
                self.currentLength -= length
                toremove = toremove+1
                if self.currentLength < self.allowedLength:
                    self.points = self.points[toremove:]
                    self.lengths = self.lengths[toremove:]
                    break

        # check if snake ate the food
        rx, ry = self.foodpos
        if rx-self.wFood//2 < cx < rx+self.wFood//2 and ry-self.hFood//2 < cy < ry+self.hFood//2:
            self.allowedLength += 35
            self.randomFoodLocation()
            self.score += 1

        # draw the snake
        if self.points:  # if it has some points
            for i, point in enumerate(self.points):
                if i != 0:
                    cv.line(imgMain, self.points[i-1],
                            self.points[i], (255, 0, 0), 20)
            cv.circle(imgMain, self.points[-1], 20, (200, 0, 200), cv.FILLED)

        # draw the food
        rx, ry = self.foodpos
        imgMain = cvzone.overlayPNG(imgMain, self.imgFood,
                                    (rx-self.wFood//2, ry-self.hFood//2))  # This tuple specifies the location where the top-left corner of the PNG image will be placed on so we are subracting from the center
        # set the score
        text = "CURRENT SCORE IS: "+str(self.score)
        position = (35, 35)
        font = cv.FONT_ITALIC
        font_scale = 1
        font_color = (0, 0, 0)  # White color in BGR format
        thickness = 4  # Thickness of the text

        # Add the text to the image using cv2.putText()
        cv.putText(imgMain, text, position, font,
                   font_scale, font_color, thickness)

    #    check for collision
        if len(self.points) > 4:
            pts = np.array(self.points[:-4], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv.polylines(imgMain, [pts], False, (0, 255, 0), 3)
            minDist = cv.pointPolygonTest(pts, (cx, cy), True)

            if -0.75 <= minDist <= 0.75:
                self.gameover = True
                self.points = []  # all points of the snake
                self.lengths = []  # distance between each point
                self.currentLength = 0  # total length of the snake
                self.allowedLength = 150  # total allowed Length
                self.previousHead = 0, 0  # previous head point
                self.randomFoodLocation()

        if self.gameover:
            finaltext = "GAME OVER"
            cv.putText(imgMain, finaltext, (120, 350), cv.FONT_ITALIC,
                       6, (0, 0, 0), 13)
            finaltext2 = "press any key to start again"
            cv.putText(imgMain, finaltext2, (200, 450), cv.FONT_ITALIC,
                       2, (0, 0, 0), 5)
        return imgMain, self.gameover


game = snakegame()

while True:
    success, img = cap.read()
    img = cv.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    hascollided = False
    if hands:
        lmlist = hands[0]['lmList']
        pointIndex = lmlist[8][0:2]
        img, hascollided = game.update(img, pointIndex)
    cv.imshow("image", img)
    if hascollided:
        game = snakegame()
        cv.waitKey(60000)
    cv.waitKey(1)
