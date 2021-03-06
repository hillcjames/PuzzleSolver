
import cv2
from PIL import Image, ImageDraw
import numpy as np
import traceback
import math
from scipy import ndimage


class PuzzlePiece:
    def __init__(self, contour):
        self.contour = contour
        self.contourDims = self.getXYMinAndWidthHeight()
        self.sharpCorners = None
        self.corners = None
        self.trueCorners = None
        self.chip = None
        self.center = None

    def setChip(self, newChip):
        # x1, y1, w, h = self.contourDims[:]
        # self.chip = binImg[y1:y1+h, x1:x1+w]
        self.chip = newChip

    def getContourShiftedNearOrigin(self, border):
        shift = self.getOffset(border)
        return np.array([p - shift for p in self.contour])

    def getXYMinAndWidthHeight(self):
        minX = 10000000
        maxX = 0
        minY = 10000000
        maxY = 0
        for p in self.contour:
            x, y = p[0][:]
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

    def getOffset(self, border):
        return np.array([self.contourDims[0] - border, self.contourDims[1] - border])

    def getContourAsList(self, useOnlyEveryXValues=1, shifted=False, border=0):
        offset = self.getOffset(border)
        l = []
        i = 0
        for p in self.contour:
            if i % useOnlyEveryXValues == 0:
                l.append(p[0]-offset)
            i += 1
        return l

def main():
    rawImg = cv2.imread('./three_pieces.JPG')
    # rawImg = cv2.imread('./scoops.JPG')

    height, width = rawImg.shape[:2]

    img = cv2.resize(rawImg, (0,0), fx=0.2, fy=0.2)
    # img = cv2.resize(rawImg, (0,0), fx=0.6, fy=0.6)
    # img = rawImg

    height, width = img.shape[:2]
    y1 = 0
    y2 = height//2
    x1 = 0
    x2 = width//2


    # smallChip = img[y1:y2, x1:x2]



    ## Gen lower mask (0-5) and upper mask (175-180) of RED
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

    outImGray = np.zeros(cleanedBinImg.shape)
    outImColor = cv2.cvtColor(cleanedBinImg, cv2.COLOR_GRAY2RGB)
    # outImColor = np.zeros(cleanedBinImg.shape)

    puzzlePieces = []
    allRealContours = []
    for c in contours:
        if c.size > 200:
            allRealContours.append(c)
            puzzlePieces.append(PuzzlePiece(c))


    chipImg = np.zeros((800, 800), np.uint8)
    cv2.drawContours(outImGray, allRealContours, -1, (255,255,255), 1)
    cv2.drawContours(outImColor, allRealContours, -1, (255,0,255), 2)

    for piece in puzzlePieces:
        offsetX, offsetY, offsetW, offsetH = piece.contourDims[:]
        border = 300
        if (offsetW + border > chipImg.shape[1]) or (offsetH + border > chipImg.shape[0]):
            chipImg = np.zeros((offsetH + border*2, offsetW + border*2), np.uint8)
        cleanedBinChipImg = img[offsetY:offsetY+offsetH, offsetX:offsetX+offsetW]
        piece.setChip(cleanedBinChipImg)

        # cv2.drawContours(chipImg, piece.getContourShiftedNearOrigin(border), -1, (255,255,255), 3)

        shiftedContourTest = piece.getContourShiftedNearOrigin(border).reshape((-1, 1, 2))
        cv2.polylines(chipImg, [shiftedContourTest], True, (255, 255, 255), 1 )
        # sharpCorners = findCornersCV2(chipImg, 40, qualityLevel=0.1, minDistance=1)
        # sharpCorners = findCornersHarris(chipImg)[1:]
        piece.sharpCorners = piece.getContourAsList(4, shifted=True, border=border)

        # print(len(sharpCorners))
        for sharpCorner in piece.sharpCorners:
            originalCornerPos = (int(offsetX + sharpCorner[0] - border), int(offsetY + sharpCorner[1] - border))
            cv2.circle(outImColor, originalCornerPos, radius=2, color=(0, 0, 255), thickness=3)
            cv2.circle(chipImg, asTup(sharpCorner), radius=3, color=(0, 0, 255), thickness=3)
            # print(sharpCorner[0], sharpCorner[1])
        # print(chipImg.shape)
        findRealCorners(chipImg, piece)
        # showNumpy(chipImg)
        # # showNumpy(outImGray)
        # return
        chipImg.fill(0)

    exit()

    edges = getEdgesSortedByLength(puzzlePieces)


    # canvas = np.zeros((1200, 1200), np.uint8)
    for e1 in edges:
        for e2 in edges:
            print(abs(e1['sqrDist'] - e2['sqrDist']))
            if (e1['piece'] != e2['piece']) and abs(e1['sqrDist'] - e2['sqrDist']) < 300:
                piece1 = e1['piece']
                piece2 = e2['piece']
                chip1 = piece1.chip
                chip2 = piece2.chip

                angleFromCenterToFurtherCwCorner = getAngleOfLine(piece1.center, e1['edge'][0])

                # chip1 = ndimage.rotate(piece1.chip, 5)

                # piece1.
                print([e1['edge'][0]] - piece1.center + 100, piece1.contourDims)

                plotPoints(piece1.chip, [(100,100), e1['edge'][0]] - piece1.center, radius=2, color=(128, 255, 125), thickness=2)
                showNumpy(piece1.chip)
                showNumpy(piece2.chip)
                exit()

                chipImgColor = cv2.cvtColor(np.zeros((800, 800), np.uint8), cv2.COLOR_GRAY2RGB)

                offset = 200
                x1 = piece1.contourDims[0] + offset
                y1 = piece1.contourDims[1] + offset
                chipImgColor[x1:x1+piece1.chip.shape[0], y1:y1+piece1.chip.shape[1], :] = piece1.chip

                x2 = piece2.contourDims[0] + offset
                y2 = piece2.contourDims[1] + offset
                chipImgColor[x2:x2+piece2.chip.shape[0], y2:y2+piece2.chip.shape[1], :] = piece2.chip

                showNumpy(chipImgColor)
                exit()
        # break

def getAngleOfLine(p1, p2):
    diff = p2-p1
    return math.atan2(diff[0], diff[1])

def getEdgesSortedByLength(pieces):
    edges = []
    for piece in pieces:
        for i in range(0,4):
            edge = np.array([piece.corners[i], piece.corners[(i+1)%4]])
            sqrDist = (edge[0][0] - edge[1][0])**2 + (edge[0][1] - edge[1][1])**2
            edges.append({"edge": edge, "sqrDist": sqrDist, "piece":piece})

    return sorted(edges, key=lambda e:e["sqrDist"])



def findRealCorners(img, piece):
    imgColor = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    center = np.array([0.0,0.0])
    for c in piece.sharpCorners:
        center += c
    center /= len(piece.sharpCorners)
    piece.center = center
    cv2.circle(imgColor, asTup(center), radius=3, color=(0, 255, 0), thickness=3)
    steps = 24
    bestRect = None
    bestEdge = None
    mostPointsOnABorder = []
    for i in range(0, steps):
        theta = 2*math.pi*(i%steps)/steps
        fakeOutlier = center + piece.contourDims[2]*2*np.array([math.cos(theta), math.sin(theta)])
        cv2.circle(imgColor, asTup(fakeOutlier), radius=3, color=(0, 0, 255), thickness=3)
        # print(piece.sharpCorners)
        newPoints = np.float32(np.vstack([piece.sharpCorners, fakeOutlier]))

        # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
        rect = cv2.minAreaRect(newPoints)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        pts = box.reshape((-1, 1, 2))
        cv2.polylines(imgColor, [pts], True, (125, 0, 0), 1 )
        pntsOnBorder, matchingEdge = getPointsWithinDistOfOneLineOfRect(piece.sharpCorners, box, 16)
        # print(i, len(pntsOnBorder))
        if len(pntsOnBorder) > len(mostPointsOnABorder):
            mostPointsOnABorder = pntsOnBorder
            bestEdge = matchingEdge
            bestRect = box

        # break
    c2, c3 = getClosestPointsToCorners(mostPointsOnABorder, bestEdge)[:]
    c1, pntsOnc1c2Edge = getBestNextCorner(c2, c3, piece.sharpCorners)
    c4, pntsOnc1c2Edge = getBestNextCorner(c3, c2, piece.sharpCorners)

    corners = validateCornersByDist(c1, c2, c3, c4, piece.sharpCorners)

    # cv2.circle(imgColor, asTup(c1), radius=3, color=(255, 0, 255), thickness=4)
    # cv2.circle(imgColor, asTup(c4), radius=3, color=(255, 255, 0), thickness=4)
    # cv2.circle(imgColor, asTup(c2), radius=5, color=(0, 255, 255), thickness=4)
    # cv2.circle(imgColor, asTup(c3), radius=5, color=(0, 255, 0), thickness=4)
    for c in corners:
        cv2.circle(imgColor, asTup(c), radius=3, color=(255, 0, 255), thickness=4)

    showNumpy(imgColor)
    piece.corners = corners

def validateCornersByDist(c1, c2, c3, c4, otherPoints):
    bestCorner = []
    bestAngle = 0
    worstAngle = 90
    corners = [c1, c2, c3, c4]
    for i in range(3):
        # print(corners)
        angle = closeToRightAngle(corners)
        if abs(abs(angle) - 90) < abs(abs(bestAngle) - 90):
            bestAngle = angle
            bestCorner = corners[:3]
        if abs(abs(angle) - 90) > abs(abs(worstAngle) - 90):
            worstAngle = angle


        shiftedCorners = corners[1:]
        shiftedCorners.append(corners[0])
        corners = shiftedCorners
    # print(abs(abs(worstAngle) - 90))
    if abs(abs(worstAngle) - 90) < 15:
        return corners

    remainingCorner = calcRemainingCorner(bestCorner)

    bestCorner.append(getClosestPointToPoint(otherPoints, remainingCorner))
    return bestCorner

def validateCornersByAngle(c1, c2, c3, c4, otherPoints):
    bestCorner = []
    bestAngle = 0
    worstAngle = 90
    corners = [c1, c2, c3, c4]
    for i in range(3):
        # print(corners)
        angle = closeToRightAngle(corners)
        if abs(abs(angle) - 90) < abs(abs(bestAngle) - 90):
            bestAngle = angle
            bestCorner = corners[:3]
        if abs(abs(angle) - 90) > abs(abs(worstAngle) - 90):
            worstAngle = angle


        shiftedCorners = corners[1:]
        shiftedCorners.append(corners[0])
        corners = shiftedCorners
    # print(abs(abs(worstAngle) - 90))
    if abs(abs(worstAngle) - 90) < 15:
        return corners

    remainingCorner = calcRemainingCorner(bestCorner)

    bestCorner.append(getClosestPointToPoint(otherPoints, remainingCorner))
    return bestCorner

def calcRemainingCorner(threePoints):
    return (threePoints[0] - threePoints[1]) + threePoints[2]

def closeToRightAngle(points):
    ba = points[0] - points[1]
    bc = points[2] - points[1]

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def getBestNextCorner(pivotCorner, otherKnownCorner, points):
    numPoints = len(points)
    pivotCornerIndex = getIndexOf(points, pivotCorner)
    otherKnownCornerIndex = getIndexOf(points, otherKnownCorner)
    # print("t", otherKnownCornerIndex, pivotCornerIndex)
    deltaIndex = otherKnownCornerIndex - pivotCornerIndex
    if abs(deltaIndex) > numPoints//2:
        if deltaIndex > 0:
            deltaIndex -= numPoints
        else:
            deltaIndex += numPoints

    if deltaIndex > 0:
        directionAwayFromPivotOppositeOther = -1
    else:
        directionAwayFromPivotOppositeOther = 1

    mostPointsOnEdge = []
    bestCorner = None
    for i in range(1, numPoints//3):
        index = (pivotCornerIndex + i*directionAwayFromPivotOppositeOther) % numPoints
        possibleCorner = points[index]
        pointsOnEdge = getPointsWithinDistOfLine(points, [pivotCorner, possibleCorner], 10)
        if len(pointsOnEdge) >= len(mostPointsOnEdge):
            mostPointsOnEdge = pointsOnEdge
            bestCorner = possibleCorner
        # print(index, len(pointsOnEdge), len(mostPointsOnEdge), possibleCorner)

    # So bestCorner is the point that creates the best line (where the line isn't confined to the two containing points)
    # So we actually need to return the furthest of the points within mostPointsOnEdge
    bestCorner = getFurthestPoint(mostPointsOnEdge, pivotCorner)
    return bestCorner, mostPointsOnEdge


def getPointsWithinDistOfLine(points, line, tolerance):
    pointsOnEdge = []
    for p in points:
        dist = distFromPointToLine(p, line[0], line[1])
        if dist < tolerance:
            pointsOnEdge.append(p)
    return pointsOnEdge

def getFurthestPoint(points, target):
    furthestDistSqr = 0
    furthestPoint = None
    for p in points:
        sqrDist = (p[0] - target[0])**2 + (p[1] - target[1])**2
        if sqrDist > furthestDistSqr:
            furthestDistSqr = sqrDist
            furthestPoint = p
    return furthestPoint


def getIndexOf(points, p):
    for i in range(len(points)):
        if p[0] == points[i][0] and p[1] == points[i][1]:
            return i
    return -1

def plotPoints(img, points, radius, color, thickness):
    i = 0
    step = 255//len(points)
    for p in points:
        color = (0, 20+ i*step, 255-i*step)
        cv2.circle(img, asTup(p), radius=radius, color=color, thickness=thickness)
        i+=1

def getClosestPointsToCorners(points, corners):
    c1 = getClosestPointToPoint(points, corners[0])
    c2 = getClosestPointToPoint(points, corners[1])
    return np.array([c1, c2])

def getClosestPointToPoint(points, target):
    closestP = None
    closestDistSqr = 1000000000
    for p in points:
        sqrDist = (p[0] - target[0])**2 + (p[1] - target[1])**2
        if sqrDist < closestDistSqr:
            closestDistSqr = sqrDist
            closestP = p
    return closestP


def getPointsWithinDistOfOneLineOfRect(points, rect, tolerance):
    mostPntsOnAnEdge = []
    bestEdge = ()
    for i in range(4):
        rectC1 = rect[i]
        rectC2 = rect[(i+1)%4]
        pointsOnEdge = []
        for p in points:
            dist = distFromPointToLine(p, rectC1, rectC2)
            if dist < tolerance:
                pointsOnEdge.append(p)
        if len(pointsOnEdge) > len(mostPntsOnAnEdge):
            mostPntsOnAnEdge = pointsOnEdge
            bestEdge = (rectC1, rectC2)
    return mostPntsOnAnEdge, bestEdge
#
# def distFromPointToRectPerimeter(p, rect):
#     d1 = distFromPointToLine(p, rect[0], rect[1])
#     d2 = distFromPointToLine(p, rect[1], rect[2])
#     d3 = distFromPointToLine(p, rect[2], rect[3])
#     d4 = distFromPointToLine(p, rect[3], rect[0])
#     b1 = d1 if d1 < d2 else d2
#     b2 = d3 if d3 < d4 else d4
#     return b1 if b1 < b2 else b2


def distFromPointToLine(p, p1, p2):
    if (p1==p2).all():
        raise ZeroDivisionError("Line endpoints equivalent")
    return np.abs(np.cross(p2-p1, p1-p)) / np.linalg.norm(p2-p1)


def asTup(l):
    return (int(l[0]), int(l[1]))

def findCornersCV2(img, maxPoints, qualityLevel, minDistance):
    cv2Corners = cv2.goodFeaturesToTrack(img, maxPoints, qualityLevel=qualityLevel, minDistance=minDistance)
    corners = []
    for cCV2 in cv2Corners:
        corners.append(cCV2[0])
    return corners

def findCornersHarris(grayImg):
    threshhold = 0.05
    gray = np.float32(grayImg)
    # dst = cv2.cornerHarris(gray, 5, 3, 0.04)
    dst = cv2.cornerHarris(gray, 5, 3, 0.04)
    ret, dst = cv2.threshold(dst,threshhold*dst.max(),255,0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.04)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    # for i in range(1, len(corners)):
    #     print(corners[i])
    # print(len(corners))
    # img = np.array([[[s,s,s] for s in p] for p in grayImg],dtype="u1")
    # # img = cv2.cvtColor(grayImg, cv2.COLOR_GRAY2RGB)
    # img[dst>threshhold*dst.max()]=[0,0,255]
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows
    return corners



# def drawContourBoundingBox(img, contour):
    # return drawContourBoundingBox(img, contour, np.zeros((2,2)))

def drawContourBoundingBox(img, contour):
    box = getBoundingBox(contour)
    pts = box.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, (255, 0, 0), 3 )

def getBoundingBox(points):
    # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)


def getListofPieceContours(binImg):
    contours, hierarchy = cv2.findContours(binImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours




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
