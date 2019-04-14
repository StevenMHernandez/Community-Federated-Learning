import math
import random
import enum


# printEnabled - Function to control all print statements in Node.py
#                Set printStatementsEnabled = False to disable print statements.
#                   If disabled, only the exportResult method will print the resulting file name when called.
def printEnabled():
    printStatementsEnabled = False
    return printStatementsEnabled


# Enum to give nodes easy status setting/checking.
class NodeStatus(enum.Enum):
    traveling = 1  # Default.
    paused = 2  # Temporary status.


class Node:
    # Self-explanatory variables.
    identifier = 0
    x = 0.0
    y = 0.0
    angle = 0.0
    speed = 0.0
    minSpeed = 0.0
    maxSpeed = 0.0
    initialPause = 0.0
    remainingPause = 0.0
    minPause = 0.0
    maxPause = 0.0
    initialTravelTime = 0.0
    remainingTravelTime = 0.0
    maxTravel = 0.0
    minTravel = 0.0
    status = NodeStatus.traveling

    # r - Transmission radius of node.
    r = 0.0

    # regionHeight, regionWidth - Simulation area size.
    regionHeight = 0
    regionWidth = 0

    def __init__(self, id, transmissionRadius, regionLength, regionWidth, minSpeed, maxSpeed, minPause, maxPause,
                 minTravel, maxTravel):
        self.identifier = id
        self.r = transmissionRadius
        (self.regionHeight, self.regionWidth) = regionLength, regionWidth
        (self.x, self.y) = self.getRandomLocation()
        (self.minSpeed, self.maxSpeed) = minSpeed, maxSpeed
        (self.minPause, self.maxPause) = minPause, maxPause
        (self.minTravel, self.maxTravel) = minTravel, maxTravel
        if printEnabled():
            print("Initialized: id=" + str(id)
                  + ", x=" + str("%.2f" % self.x)
                  + ", y=" + str("%.2f" % self.y)
                  + ", transmission=" + str(self.r)
                  + ", regionHeight=" + str(self.regionHeight)
                  + ", regionWidth=" + str(self.regionWidth)
                  + ", minSpeed=" + str(self.minSpeed)
                  + ", maxSpeed=" + str(self.maxSpeed)
                  + ", minPause=" + str(self.minPause)
                  + ", maxPause=" + str(self.maxPause)
                  + ", minTravel=" + str(self.minTravel)
                  + ", maxTravel=" + str(self.maxTravel))
        self.randomizeMovementParameters()

    # initializeNodes - Returns a list of nodes with all values initialized.
    #                   Logs will be printed to the console of all parameters set.
    def initializeNodes(numberNodes, transmissionRadius,
                        regionLength, regionWidth,
                        minSpeed, maxSpeed,
                        minPause, maxPause,
                        minTravel, maxTravel):
        nodes = []
        for identifier in range(0, numberNodes):
            nodes.append(
                Node(identifier, transmissionRadius, regionLength, regionWidth, minSpeed, maxSpeed, minPause, maxPause,
                     minTravel, maxTravel))
        return nodes

    # moveNodes - Takes array of nodes as a parameter, updates each node's position and status accordingly.
    #             This method assumes that it is being called from a function in which the list of nodes that is passed
    #             to the function is declared. (I know that is a big difficult to comprehend.)
    #
    #             We can modify this to return the list again or something if we really want to...
    def moveNodes(nodes):
        for node in nodes:
            node.move()

    # checkForNeighbors - Takes array of nodes and the current time interval as parameters. It then goes through each
    #                     node and calculates the distance between it and the other nodes.
    #                     If the distance between them is less than or equal to the transmission size, then we create
    #                     a string if "time,id1,id2" and store it temporarily.
    #                     The function ends by returning a list of all nodes that communicated
    #                     within the given time interval.
    #
    #                     We can alter it so that the function returns an dictionary of nodes that communicated
    #                     or whatever you see fit, Steven.
    def checkForNeighbors(nodes, currentTime):
        communicationStringList = []

        for nodeOne in nodes:
            for nodeTwo in nodes:
                if nodeOne != nodeTwo:
                    distance = nodeOne.euclidianDistance(nodeTwo)
                    if nodeOne.sees(nodeTwo):
                        # Currently this string consists of... time, node1Id, node2Id
                        communicationString = str(currentTime) + "," + str(nodeOne.id) + "," + str(nodeTwo.id)
                        communicationStringList.append(communicationString)
                        if printEnabled():
                            print("Communication: time=" + str(currentTime)
                                  + ", id=" + str(nodeOne.id)
                                  + ", id=" + str(nodeTwo.id)
                                  + ", distance=" + str("%.2f" % distance)
                                  + ", transmissionRadius=" + str(nodeOne.transmissionRadius)
                                  + ", regionSize=" + str(nodeOne.regionHeight) + "x" + str(nodeOne.regionWidth)
                                  + ", communication=True")
                    else:
                        if printEnabled():
                            print("Communication: time=" + str(currentTime)
                                  + ", id=" + str(nodeOne.id)
                                  + ", id=" + str(nodeTwo.id)
                                  + ", distance=" + str("%.2f" % distance)
                                  + ", transmissionRadius=" + str(nodeOne.transmissionRadius)
                                  + ", regionSize=" + str(nodeOne.regionHeight) + "x" + str(nodeOne.regionWidth)
                                  + ", communication=False")
        return communicationStringList

    # getRandomLocation - Generates a random coordinate within the simulation region.
    #                     This function is only called when the nodes are initialized.
    def getRandomLocation(self):
        if self.regionHeight > 0 and self.regionWidth > 0:
            return random.randint(1, self.regionHeight), random.randint(1, self.regionWidth)
        else:
            return 0, 0

    # randomizeMovementParameters - Sets the node this function was called on to "traveling" status. It also
    #                               randomizes the traveling and pausing parameters within their respective
    #                               bounds.
    def randomizeMovementParameters(self):
        self.travelStatus()
        self.angle = self.setAngle()
        self.speed = self.setSpeed()
        (self.initialPause, self.remainingPause) = self.setPause()
        (self.initialTravelTime, self.remainingTravelTime) = self.setTravel()
        if printEnabled():
            print("Randomize: id=" + str(self.identifier)
                  + ", angle=" + str("%.2f" % self.angle)
                  + ", speed=" + str(self.speed)
                  + ", pause=" + str(self.initialPause)
                  + ", travel=" + str(self.initialTravelTime))

    # pauseStatus - Sets the node this function was called on to "pause" status.
    def pauseStatus(self):
        self.status = NodeStatus.paused

    # decrementRemainingPause - Decrements the remainingPause counter for node this function was called on.
    def decrementRemainingPause(self):
        self.remainingPause = self.remainingPause - 1

    # pauseRemaining - Returns a boolean value indicating if the remainingPause variable for the node this
    #                  function was called on is greater than 0.
    def pauseRemaining(self):
        if self.remainingPause > 0:
            return True
        else:
            return False

    # travelStatus - Sets the node this function was called on to "traveling" status.
    def travelStatus(self):
        self.status = NodeStatus.traveling

    # decrementRemainingTravel - Decrements the remainingTravel counter for node this function was called on.
    def decrementRemainingTravel(self):
        self.remainingTravelTime = self.remainingTravelTime - 1

    # travelRemaining - Returns a boolean value indicating if the remainingTravel variable for the node this
    #                  function was called on is greater than 0.
    def travelRemaining(self):
        if self.remainingTravelTime > 0:
            return True
        else:
            return False

    # setAngle - Sets the angle of travel for the node this function was called on.
    def setAngle(self):
        return random.uniform(0, math.pi)

    # setSpeed = Sets the speed of the node this function was called on.
    def setSpeed(self):
        return random.random() * (self.maxSpeed - self.minSpeed) + self.minSpeed

    # setPause - Returns a randomly generated integer between the min and max pause time.
    #            This function specifically returns two copies of the randomPause variable.
    #            The reason for this is because, where it is used within the randomizeMovementParameters
    #            function, it is setting both the initialPause as well as remainingPause variables for the given node.
    def setPause(self):
        randomPause = random.randint(self.minPause, self.maxPause)
        return randomPause, randomPause

    # setTravel - Returns a randomly generated integer between the min and max travel time.
    #             This function specifically returns two copies of the randomTravel variable.
    #             The reason for this is because, where it is used within the randomizeMovementParameters
    #             function, it is setting both the initialTravelTime as well as remainingTravelTime variables
    #             for the given node.
    def setTravel(self):
        randomTravel = random.randint(self.minTravel, self.maxTravel)
        return randomTravel, randomTravel

    # setCurrentCoordinates - Sets x and y passed as parameters as the nodes new x and y position
    #                         for the node this function was called on.
    def setCurrentCoordinates(self, x, y):
        self.x = x
        self.y = y

    # getCurrentCoordinates - Gets the coordinates of the node this function was called on.
    def getCurrentCoordinates(self):
        return self.x, self.y

    # exists - Checks if the coordinates passed as parameters exist within the simulation region
    #          and returns True or False accordingly.
    def exists(self, x, y):
        if x >= 0 and y >= 0 and x <= self.regionWidth and y <= self.regionHeight:
            return True
        return False

    # determineLinesIntersectionPoint - Given two lines in the format [ [ x, y ], [ x, y ] ], [ [ x, y ], [ x, y ] ]
    #                                   it returns the point in which the two lines intersect.
    def determineLinesIntersectionPoint(self, line1, line2):
        # print("Line1: " + str(line1))
        # print("Line2: " + str(line2))

        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])  # Typo was here

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('No intersection!')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

    # findClosestIntersection - This method returns the intersection coordinates of the border intersection directly
    #                           within the node's path. It compares all of the border intersections passed to it
    #                           and returns the one closest to the node's current position.
    #                           This function ensures it returns the correct border intersection because this function
    #                           is only called when we have determined that, during this interval, the node will step
    #                           outside of the simulation region.
    def findClosestIntersection(self, intersections, x, y):
        intersectionDistanceMap = []
        for intersection in intersections:
            intersectionDistanceMap.append([intersection, math.hypot(intersection[0] - x, intersection[1] - y)])

        # print("IntersectionDistanceMap: " + str(intersectionDistanceMap))

        bestIntersection = intersectionDistanceMap[0][0]
        bestLength = intersectionDistanceMap[0][1]
        for intersection in intersectionDistanceMap:
            # print("BestIntersectionLength=" + str(bestLength) + ", IntersectionLength=" + str(intersection[1]))
            if intersection[1] < bestLength:
                bestIntersection = intersection[0]
                bestLength = intersection[1]

        return bestIntersection

    # determineBorderIntersection - This function returns the point on the border in which the node's current path
    #                               intersects with the simulation region's border.
    #                               It finds all border intersection points.
    #                               If they exist within the simulation region, they are passed on to determine which
    #                               of the intersections is closes to the node's current position.
    #                               The closes border intersection is returned.
    def determineBorderIntersection(self, x1, y1, x2, y2):
        bottomLeft = [0, 0]
        bottomRight = [0, self.regionWidth]
        topLeft = [self.regionHeight, 0]
        topRight = [self.regionHeight, self.regionWidth]

        bottom = [bottomLeft, bottomRight]
        top = [topLeft, topRight]
        left = [bottomLeft, topLeft]
        right = [bottomRight, topRight]

        borders = [bottom, top, left, right]
        path = [[x1, y1], [x2, y2]]

        intersections = []

        for border in borders:
            intersection = self.determineLinesIntersectionPoint(path, border)
            # print("Intersection: " + str(intersection))
            if self.exists(intersection[0], intersection[1]):
                # print("Intersection exists!")
                intersections.append(intersection)
            # else:
            # print("Intersection doesn't exist!")

        # Now, pick the intersection closest to where the node would
        # have ended up if it didn't want to stop at the border.
        finalIntersection = self.findClosestIntersection(intersections, x2, y2)

        # print("Final Intersection: " + str(finalIntersection))

        return finalIntersection[0], finalIntersection[1]

    # moveWithoutTimePenalty - Moves the node without reducing any of the time penalties.
    #                          This is used in special cases. Nodes are still set to their appropriate
    #                          statuses, just no time costs are incurred.
    def moveWithoutTimePenalty(self):
        angleInRadians = math.radians(self.angle)

        cosValue = math.cos(angleInRadians)
        sinValue = math.sin(angleInRadians)

        newX = self.x + self.speed * cosValue
        newY = self.y + self.speed * sinValue

        # Check if the new points are within the region
        if self.exists(newX, newY):
            if printEnabled():
                print("Move: id=" + str(self.identifier)
                      + ", x=" + str("%.2f" % self.x)
                      + ", y=" + str("%.2f" % self.y)
                      + ", speed=" + str(self.speed)
                      + ", newX=" + str("%.2f" % newX)
                      + ", newY=" + str("%.2f" % newY)
                      + ", exists=True")
            self.x = newX
            self.y = newY
        else:
            # Need to set nodes point to intersection of path and the region border
            # Node also needs to be set to "paused" and randomize its traveling parameters
            newX, newY = self.determineBorderIntersection(self.x, self.y, newX, newY)
            if printEnabled():
                print("Move: id=" + str(self.identifier)
                      + ", x=" + str("%.2f" % self.x)
                      + ", y=" + str("%.2f" % self.y)
                      + ", speed=" + str(self.speed)
                      + ", newX=" + str("%.2f" % newX)
                      + ", newY=" + str("%.2f" % newY)
                      + ", exists=False")
            self.x = newX
            self.y = newY
            self.pauseStatus()
            self.decrementRemainingPause()

        pass

    # move - This function does several things...
    #        It first checks if the node is paused and decrements its counter or
    #        transitions it to traveling if its pause timer is up.
    #        If the node isn't paused, it simply calculates a new position on the node's path,
    #        and tries to move the node.
    #        If the new position is outside of the simulation, the node currently is set to pause
    #        and repeat the process once its pause time is up.
    def move(self):
        if self.status == NodeStatus.paused and self.pauseRemaining():
            self.decrementRemainingPause()
            if printEnabled():
                print("Pause: id=" + str(self.identifier)
                      + ", pauseRemaining=" + str(self.remainingPause))
            return

        if self.status == NodeStatus.paused and not self.pauseRemaining():
            self.travelStatus()
            self.randomizeMovementParameters()
            self.moveWithoutTimePenalty()
            if printEnabled():
                print("Transition: id=" + str(self.identifier))
            return

        angleInRadians = math.radians(self.angle)

        cosValue = math.cos(angleInRadians)
        sinValue = math.sin(angleInRadians)

        newX = self.x + self.speed * cosValue
        newY = self.y + self.speed * sinValue

        # Check if the new points are within the region
        if self.exists(newX, newY) and self.travelRemaining() and self.status == NodeStatus.traveling:
            if printEnabled():
                print("Move: id=" + str(self.identifier)
                      + ", x=" + str("%.2f" % self.x)
                      + ", y=" + str("%.2f" % self.y)
                      + ", speed=" + str(self.speed)
                      + ", angle=" + str("%.2f" % self.angle)
                      + ", newX=" + str("%.2f" % newX)
                      + ", newY=" + str("%.2f" % newY)
                      + ", exists=True")
            self.x = newX
            self.y = newY
            self.decrementRemainingTravel()
        else:
            # Need to set nodes point to intersection of path and the region border
            # Node also needs to be set to "paused" and randomize its traveling parameters
            newX, newY = self.determineBorderIntersection(self.x, self.y, newX, newY)
            if printEnabled():
                print("Move: id=" + str(self.identifier)
                      + ", x=" + str("%.2f" % self.x)
                      + ", y=" + str("%.2f" % self.y)
                      + ", speed=" + str(self.speed)
                      + ", angle=" + str("%.2f" % self.angle)
                      + ", newX=" + str("%.2f" % newX)
                      + ", newY=" + str("%.2f" % newY)
                      + ", exists=False")
            self.x = newX
            self.y = newY
            self.pauseStatus()
            # self.decrementRemainingPause()

        pass

    def euclidianDistance(self, node):
        return math.sqrt(pow(self.x - node.x, 2) + pow(self.y - node.y, 2))

    def sees(self, node):
        return self.euclidianDistance(node) < self.r

    #  exportResults - Creates a CSV file output containing the information from the provided array of strings.
    def exportResults(data, fileName):
        print("\nExportingData: ")
        with open('Output_' + fileName, "w+") as outputFile:
            for row in data:
                print("   row=" + row)
                outputFile.write(row + "\n")
        return True


'''
        # Set some simulation parameters...
        numberNodes = 5
        regionLength = 100
        regionWidth = 100
        simulationTime = 100;

        # Start Simulation

        # Node Parameters
        transmissionRadius = 10
        minSpeed = 1
        maxSpeed = 10
        minPause = 1
        maxPause = 5
        minTravel = 1
        maxTravel = simulationTime

        print(" - Initializing Nodes - ")
        nodes = Simulation.initializeNodes(numberNodes,
                                     transmissionRadius,
                                     regionLength, regionWidth,
                                     minSpeed, maxSpeed,
                                     minPause, maxPause,
                                     minTravel, maxTravel)

        #distributeData() - Distribute data to the nodes for training...

        print("\n - Starting Simulation - ")

        # Debugging printing in the for loop below
        for node in nodes:
            print("Move: id=" + str(node.id)
                  + ", x=" + str("%.2f" % node.x)
                  + ", y=" + str("%.2f" % node.y)
                  + ", speed=" + str(node.speed))

        print("\n")

        simulationCommunicationLog= []

        # Simulation's main loop
        for i in range(0, simulationTime):
            print("Iteration: time=" + str(i + 1))
            #trainNodes() - Train nodes

            iterationsCommunications = Simulation.checkForNeighbors(nodes, i)

            for string in iterationsCommunications:
                simulationCommunicationLog.append(string)

            Simulation.moveNodes(nodes)

            #exchangeKnowledge() - Exchange learning delta?

            print("\n")

        # Export the communications between nodes logged during the simulation...
        # The resulting file will be named "Output_CommunicationLog.csv"
        exportResults(simulationCommunicationLog, "CommunicationLog.csv")
'''
