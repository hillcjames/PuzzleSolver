
import cv2
from PIL import Image, ImageDraw
import numpy as np
import traceback


class PuzzlePiece:
    def __init__(self, contour):
        self.contour = contour
        self.contourDims = self.getXYMinAndWidthHeight()
        self.sharpCorners = None
        self.trueCorners = None


    def getContourShiftedNearOrigin(self, border):
        shift = np.array([self.contourDims[0] - border, self.contourDims[1] - border])
        return [p - shift for p in self.contour]

    def getXYMinAndWidthHeight(self):
        minX = 10000000
        maxX = 0
        minY = 10000000
        maxY = 0
        for p in self.contour:
            x, y = p[0][:]
            print(x, y)
            if x < minX:
                minX = x
            elif x > maxX:
                maxX = x
            elif y < minY:
                minY = y
            elif y > maxY:
                maxY = y

        w = maxX - minX
        h = maxY - minY

        return minX, minY, w, h


def main():
    rawImg = cv2.imread('./three_pieces.JPG')
    # rawImg = cv2.imread('./scoops.JPG')

    height, width = rawImg.shape[:2]

    img = cv2.resize(rawImg, (0,0), fx=0.2, fy=0.2)


    height, width = img.shape[:2]
    y1 = 0
    y2 = height//2
    x1 = 0
    x2 = width//2


    # smallChip = img[y1:y2, x1:x2]



    ## Gen lower mask (0-5) and upper maimgsk (175-180) of RED
    # mask1 = cv2.inRange(smallChip, (150,0,0), (255,50,50))
    # print(mask1.shape)
    # print(smallChip.shape)
    # print(mask1.max())
    red = img[:,:,2]

    super_threshold_indices = red < 130
    red[super_threshold_indices] = 0

    super_threshold_indices = red > 130
    red[super_threshold_indices] = 255

    cleanedBinImg = getConnectedComponents(red)

    contours = getListofPieceContours(cleanedBinImg)

    # outImGray = np.zeros(cleanedBinImg.shape)
    outImColor = cv2.cvtColor(cleanedBinImg, cv2.COLOR_GRAY2RGB)

    puzzlePieces = []
    allRealContours = []
    for c in contours:
        if c.size > 200:
            allRealContours.append(c)
            puzzlePieces.append(PuzzlePiece(c))


    chipImg = np.zeros((500, 500), np.uint8)
    # cv2.drawContours(outImGray, realContours, -1, (255,255,255), 3)
    cv2.drawContours(outImColor, allRealContours, -1, (255,0,0), 1)
    for piece in puzzlePieces:
        offsetX, offsetY, offsetW, offsetH = piece.contourDims[:]
        border = 50
        if (offsetW + border > chipImg.shape[0]) or (offsetH + border > chipImg.shape[1]):
            chipImg = np.zeros((offsetW + border*2, offsetH + border*2), np.uint8)
        try:
            # cv2.drawContours(chipImg, piece.getContourShiftedNearOrigin(border), -1, (255,255,255), 3)
            cv2.drawPolylines(chipImg, piece.getContourShiftedNearOrigin(border), -1, (255,255,255), 3)
        except:
            print("drawing contour on chip failed")
            traceback.print_exc()

        showNumpy(chipImg)
        return
        sharpCorners = cv2.goodFeaturesToTrack(chipImg, 20, qualityLevel=0.2, minDistance=0)
        for sharpCorner in sharpCorners:
            originalCornerPos = (int(offsetX + sharpCorner[0][0] - border), int(offsetY + sharpCorner[0][1] - border))
            cv2.circle(outImColor, originalCornerPos, radius=2, color=(0, 0, 255), thickness=3)

    #
    # for c in realContours:
    #     drawCorners(outImColor, c)

    #
    # for c in realContours:
    #     drawCorners(outImColor, c)

    showNumpy(outImColor)


def drawCorners(img, contour):

        # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        pts = box.reshape((-1, 1, 2))
        print(pts)
        cv2.polylines(img, [pts], True, (255, 0, 0), 3 )




def getListofPieceContours(binImg):
    contours, hierarchy = cv2.findContours(binImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours
    # img = np.zeros(binImg.shape)
    # cv2.drawContours(img, contours, -1, (255,255,255), 3)
    # showNumpy(img)




def getConnectedComponents(binImg):

    # do connected components processing
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binImg, None, None, None, 8, cv2.CV_32S)

    #get CC_STAT_AREA component as stats[label, COLUMN]
    areas = stats[1:,cv2.CC_STAT_AREA]

    result = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if areas[i] >= 100:   #keep
            result[labels == i + 1] = 255

    return result



def showNumpy(npImg):
    # npImg = cv2.cvtColor(npImg, cv2.COLOR_HSV2RGB)
    imPil = Image.fromarray(npImg)
    imPil.show()





if __name__ == "__main__":
    main()
