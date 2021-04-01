#Definition of all functions  used in the project
#Last revision - Oct 2019
import faulthandler
import math
import time
faulthandler.enable()
import copy
import itertools
import resource, sys
resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)
from my_classes import *
from docplex.mp.model import Model
import numpy as np
from decimal import *
from timeout import *
import timeout_decorator
from my_learning_classes import *
from my_classes import *
import operator
import warnings
import random
from torch_geometric.data import Data
warnings.filterwarnings("ignore")
getcontext().prec = 6

#Parameters
pr_wait = 0.5
child_timeout = 3
child_attempts = 25
or_depth_order_lim = 60
and_depth_order_lim = 0
max_window_size = 256
visitedChildrenStorageThreshold = 20


delta = Decimal(20)
upper_bound = 100000000000
cplex_time_limit = 3
gen_prob = 0.9
node_limit = 300
depth_pen_fact = 10
depth_order_lim = 10

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = NetVariableClasses()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.eval()

def boundDetectionProp(a, b, propagatedTimepoint, minTab, listOfProcessedConjuncts,
                       sumLb, sumUb, remainingFreeConstraints):

    if sumLb > b:
        return

    listOfDisjuncts = remainingFreeConstraints[propagatedTimepoint]
    listOfConj = []
    for disjunct in listOfDisjuncts:
        for conjunct in disjunct:
            if isDistance(conjunct):
                if not conjunct in listOfProcessedConjuncts:
                    t1 = conjunct[0]
                    t2 = conjunct[1]
                    d1 = conjunct[2]
                    d2 = conjunct[3]
                    if t1 == propagatedTimepoint and d1 >=0:
                        listOfConj.append(conjunct)


    if listOfConj == []:
        return

    for conjunct in listOfConj:
        t1 = conjunct[0]
        t2 = conjunct[1]
        d1 = conjunct[2]
        d2 = conjunct[3]
        nlb = sumLb + d1
        nub = sumUb + d2

        if a > nlb:
            if a - nlb > 0 and a - nlb < minTab[0]:
                minTab[0] = a - nlb
        if a > nub:
            if a - nub > 0 and a - nub < minTab[0]:
                minTab[0] = a - nub
        if b > nlb:
            if b - nlb > 0 and b - nlb < minTab[0]:
                minTab[0] = b - nlb
        if b > nub:
            if b - nub > 0 and b - nub < minTab[0]:
                minTab[0] = b - nub
        if not conjunct in listOfProcessedConjuncts:
            listOfProcessedConjuncts.append(conjunct)
        boundDetectionProp(a, b, t2, minTab, listOfProcessedConjuncts,
                       nlb, nub, remainingFreeConstraints)
        if conjunct in listOfProcessedConjuncts:
            listOfProcessedConjuncts.remove(conjunct)








#Bound detection
#Inspects remaining constraints and tries to find a a bound happens before scheduled wait. Minimum bound
#Is returned, which should be the new wait time
def boundDetection(remainingFreeConstraints, activatedUncontrollables):
    min = upper_bound
    for key in remainingFreeConstraints.keys():
        for disjunct in remainingFreeConstraints[key]:
            for conjunct in disjunct:
                if isBounds(conjunct):
                    timePointNumber1 = conjunct[0]
                    a = conjunct[1]
                    b = conjunct[2]
                    if a < min and a > 0:
                        min = a
                    if b < min and b > 0:
                        min = b

                    minTab = [1000000000]
                    boundDetectionProp(a, b, timePointNumber1, minTab, [], 0, 0, copy.deepcopy(remainingFreeConstraints))
                    if minTab[0] < min and minTab[0] > 0:
                        min = minTab[0]
                    '''
                    for disjunct1 in remainingFreeConstraints[timePointNumber1]:
                        for conjunct1 in disjunct1:
                            if isDistance(conjunct1):
                                t1 = conjunct1[0]
                                t2 = conjunct1[1]
                                ap = conjunct1[2]
                                bp = conjunct1[3]
                                if t1 == timePointNumber1:
                                    if a - bp < min and a - bp > 0:
                                        min = a - bp
                                    if b - ap < min and b - ap > 0:
                                        min = b - ap

                                if t2 == timePointNumber1:
                                    if a + ap < min and a + ap > 0:
                                        min = a + ap
                                    if b + bp < min and b + bp > 0:
                                        min = b + bp
                    
                    '''

                else:
                    timePointNumber1 = conjunct[0]
                    timePointNumber2 = conjunct[1]
                    a = conjunct[2]
                    b = conjunct[3]

                    if timePointNumber1 in activatedUncontrollables.keys():
                        for bound in activatedUncontrollables[timePointNumber1]:
                            x = bound[0]
                            y = bound[1]
                            if x <= 0 and y >= 0:
                                if (b - a > 0) and (b - a < min):
                                    min = b - a

                    if timePointNumber2 in activatedUncontrollables.keys():
                        for bound in activatedUncontrollables[timePointNumber2]:
                            x = bound[0]
                            y = bound[1]
                            if x <= 0 and y >= 0:
                                if (b - a > 0) and (b - a < min):
                                    min = b - a


    for key in activatedUncontrollables.keys():
        for bound in activatedUncontrollables[key]:
            a = bound[0]
            b = bound[1]
            if a < min and a > 0:
                min = a
            if b < min and b > 0:
                min = b

    return min

def notExistCombination(orNode):
    listOfScheduledPoints = []
    currentNode = orNode
    while True:
        if currentNode.parentNode != None:
            if currentNode.parentNode.nodeType == 0:
                if currentNode.parentNode.lastControllableScheduled != None:
                    listOfScheduledPoints.append(currentNode.parentNode.lastControllableScheduled)
                currentNode = currentNode.parentNode

            elif currentNode.parentNode.nodeType == 1:
                currentNode = currentNode.parentNode

            elif currentNode.parentNode.nodeType == 2:
                break

        else:
            break

    listOfScheduledPoints.sort()

    return str(listOfScheduledPoints), currentNode

def notExistCombinationPoints(subDtnu):


    listOfScheduledPoints = []
    listOfScheduledPoints.append(subDtnu.lastControllableScheduled)
    currentNode = subDtnu

    while True:
        if currentNode.parentNode != None:
            if currentNode.parentNode.nodeType == 0:
                if currentNode.parentNode.lastControllableScheduled != None:
                    listOfScheduledPoints.append(currentNode.parentNode.lastControllableScheduled)
                currentNode = currentNode.parentNode

            elif currentNode.parentNode.nodeType == 1:
                currentNode = currentNode.parentNode

            elif currentNode.parentNode.nodeType == 2:
                break

        else:
            break

    listOfScheduledPoints.sort()

    return str(listOfScheduledPoints), currentNode

#Returns true if conjunct is distance, false if bounds
def isDistance(conjunct):
    if len(conjunct) == 4:
        return True
    else:
        return False
def isBounds(conjunct):
    if len(conjunct) == 3:
        return True
    else:
        return False


# Cplex DTNU solving
def cplexSolve(dtnu):
    isSolved = False
    x_vars = {}

    mdl = Model(name='dtnu')
    mdl.set_time_limit(cplex_time_limit)
    freeConstraints = dtnu.remainingFreeConstraints

    #freeConstraints = {0: [[[0, 1, 1, 2], [0, 2, 5, 6]],[[1, 3, 4]],[[2, 10, 12]],[[0,0,20]]]}
    bin = []
    for key in freeConstraints.keys():
        for disjunct in freeConstraints[key]:
            if disjunct != []:
                i = 0
                sizeOfDisjunct = len(disjunct)
                binaryVariables = []
                for k in range(sizeOfDisjunct):
                    binaryVariables.append(mdl.binary_var())
                binaryVariables = np.array(binaryVariables)
                mdl.add_constraint(np.sum(binaryVariables) == 1)
                bin.append(binaryVariables)
                for conjunct in disjunct:

                    row = []
                    for k in range(sizeOfDisjunct):
                        if k == i:
                            row.append((1 - binaryVariables[k]) * upper_bound)
                        else:
                            row.append(binaryVariables[k] * upper_bound)
                    row = np.array(row)

                    bound = np.sum(row)

                    if isDistance(conjunct):
                        timePointNumber1 = conjunct[0]
                        timePointNumber2 = conjunct[1]
                        a = float(conjunct[2])
                        b = float(conjunct[3])
                        if not timePointNumber1 in x_vars:
                            x_vars[timePointNumber1] = mdl.continuous_var()
                        if not timePointNumber2 in x_vars:
                            x_vars[timePointNumber2] = mdl.continuous_var()

                        mdl.add_constraint(x_vars[timePointNumber1] - x_vars[timePointNumber2] + bound >= a)
                        mdl.add_constraint(x_vars[timePointNumber1] - x_vars[timePointNumber2] <= b + bound)
                    else:
                        timePointNumber1 = conjunct[0]
                        a = float(conjunct[1])
                        b = float(conjunct[2])
                        if not timePointNumber1 in x_vars:
                            x_vars[timePointNumber1] = mdl.continuous_var()

                        mdl.add_constraint(x_vars[timePointNumber1] + bound >= a)
                        mdl.add_constraint(x_vars[timePointNumber1] <= b + bound)

                    i = i + 1



    for key in x_vars.keys():
        mdl.add_constraint(x_vars[key] >= 0)
        mdl.add_constraint(x_vars[key] <= 100000)
    mdl.minimize(0)
    if mdl.solve():
        for key in x_vars:
            dtnu.relativeOccurenceTimes[key] = [Decimal(x_vars[key].solution_value), Decimal(x_vars[key].solution_value)]
            dtnu.globalOccurenceTimes[key] = [Decimal(x_vars[key].solution_value) + dtnu.globalTime,
                                              Decimal(x_vars[key].solution_value) + dtnu.globalTime]

        return True, x_vars
    else:
        return False, x_vars


def isSolved(dtnu, train, start_time, time_out):
    remainingConstraints = dtnu.remainingFreeConstraints
    remainingUncontrollables = dtnu.remainingUncontrollables
    noUncontrollablesLeft = False
    if len(remainingUncontrollables) == 0:
        noUncontrollablesLeft = True

    if not noUncontrollablesLeft:
        for k in remainingConstraints.keys():
            for disjunct in remainingConstraints[k]:
                if disjunct != []:
                    return 2
        return 1
    else:
        #MAKE THE CPLEX CHECK
        truth = True
        for k in remainingConstraints.keys():
            for disjunct in remainingConstraints[k]:
                if disjunct != []:
                    truth = False

        if truth:
            return 1
        else:
            try:
                if time.time() - start_time < time_out - cplex_time_limit:
                    return cplexSolve(dtnu)[0]
                else:
                    return 2
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                raise timeout_decorator.timeout_decorator.TimeoutError
                return 2

def isThereNonDistanceConstraint(dtnu):
    for key in dtnu.remainingFreeConstraints.keys():
        for disjunct in dtnu.remainingFreeConstraints[key]:
            for conjunct in disjunct:
                if isBounds(conjunct):
                    return True
    if dtnu.activatedUncontrollables != {}:
        return True
    return False

#Returns 1 if the node can be cut, 0 otherwise. Exececuted at beginning of node
def parentCheckCut(treeNode):
    if treeNode.parentNode != None:
        parentTruth = treeNode.parentNode.truthValue
        if parentTruth == 1 or parentTruth == 0:
            return True
        else:
            return False
    return False

#Returns 1 if the current node can be cut, 0 otherwise. Executed once after each sub-DTNU is processed
def currentCheckCut(treeNode):
    truthValue = treeNode.truthValue
    if truthValue == 1 or truthValue == 0:
        return True
    else:
        return False

def checkOrVal(treeNode):
    if treeNode.listOfChildren != None:
        FalseCount = 0
        for child in treeNode.listOfChildren:
            if child.truthValue == 1:
                treeNode.truthValue = 1
                return
            elif child.truthValue == 0:
                FalseCount = FalseCount + 1
        if FalseCount == treeNode.numberOfChildren:
            treeNode.truthValue = 0
            return
        else:
            treeNode.truthValue = 2
            return





#Propagates True to the parents of the child node
def propagateTrue(childNode):
    propagate = True
    propagateNode = childNode
    while propagate:
        propagate = False
        parent = propagateNode.parentNode

        if parent == None:
            return

        if parent.truthValue == 2:

            #If parent is a DTNU
            if parent.nodeType == 0:
                parent.truthValue = 1

            #If parent is OR node
            elif parent.nodeType == 1:
                parent.truthValue = 1
                parent.validatedChild = propagateNode

            #If parent is AND node
            elif parent.nodeType == 2:
                if not propagateNode in parent.listOfTrueChildren:
                    parent.numberOfTrueChildren = parent.numberOfTrueChildren + 1
                    parent.listOfTrueChildren.append(propagateNode)
                if parent.numberOfTrueChildren == parent.numberOfChildren:
                    parent.truthValue = 1

            #If parent is WAIT node
            elif parent.nodeType == 3:
                parent.truthValue = 1

        if parent.truthValue == 1:
            propagate = True
        propagateNode = parent
    return

#Propagates False to the parents of child node
def propagateFalse(childNode):
    propagate = True
    propagateNode = childNode
    while propagate:
        propagate = False
        parent = propagateNode.parentNode
        if parent == None:
            return

        if parent.truthValue == 2:

            #If parent is a DTNU
            if parent.nodeType == 0:
                parent.truthValue = 0

            #If parent is OR node
            elif parent.nodeType == 1:
                if not propagateNode in parent.listOfFalseChildren:
                    parent.numberOfFalseChildren = parent.numberOfFalseChildren + 1
                    parent.listOfFalseChildren.append(propagateNode)
                if parent.numberOfFalseChildren == parent.numberOfChildren:
                    parent.truthValue = 0

            #If parent is AND node
            elif parent.nodeType == 2:
                parent.truthValue = 0

            #If parent is WAIT node
            elif parent.nodeType == 3:
                parent.truthValue = 0

        if parent.truthValue == 0:
            propagate = True
        propagateNode = parent
    return

def buildReacts(dtnu, waitDuration):
    remainingFreeConstraints = dtnu.remainingFreeConstraints
    activatedUncontrollables = dtnu.activatedUncontrollables
    remainingControllables = dtnu.remainingControllables
    instantReacts = {}

    for u in activatedUncontrollables.keys():
        for disjunctiveConstraint in remainingFreeConstraints[u]:
            for conjunct in disjunctiveConstraint:
                if isDistance(conjunct):
                    timePointNumber1 = conjunct[0]
                    timePointNumber2 = conjunct[1]
                    a = conjunct[2]
                    b = conjunct[3]
                    if timePointNumber1 in activatedUncontrollables and timePointNumber2 in remainingControllables:
                        if a == 0:
                            for bound in activatedUncontrollables[timePointNumber1]:
                                if waitDuration > bound[0] and bound[1] > 0:
                                    if not timePointNumber2 in instantReacts:
                                        instantReacts[timePointNumber2] = timePointNumber1
                    if timePointNumber1 in remainingControllables and timePointNumber2 in activatedUncontrollables:
                        if b == 0:
                            for bound in activatedUncontrollables[timePointNumber2]:
                                if waitDuration > bound[0] and bound[1] > 0:
                                    if not timePointNumber1 in instantReacts:
                                        instantReacts[timePointNumber1] = timePointNumber2

    return instantReacts



def subDtnuMaker(dtnu, elapsedTime, listOfTasks, instantReacts):

    #Get last DTNU information
    #Dynamic lists and dicts
    remainingFreeConstraints = copy.deepcopy(dtnu.remainingFreeConstraints)
    remainingContingencyLinks = copy.deepcopy(dtnu.remainingContingencyLinks)
    activatedUncontrollables = copy.deepcopy(dtnu.activatedUncontrollables)
    relativeOccurenceTimes = copy.deepcopy(dtnu.relativeOccurenceTimes)
    globalOccurenceTimes = copy.deepcopy(dtnu.globalOccurenceTimes)
    remainingControllables = copy.deepcopy(dtnu.remainingControllables)
    remainingUncontrollables = copy.deepcopy(dtnu.remainingUncontrollables)
    lastControllableScheduled = None
    occuredUncontrollablesDuringLastWait = None
    lastWaitTime = elapsedTime
    instantReacts = instantReacts
    childOr = None
    tasksElapsedToCheck = []
    controllableSetTimeAfterUncontrollable = copy.deepcopy(dtnu.controllableSetTimeAfterUncontrollable)
    scheduledCombinations = {}
    scheduledPointsCombination = {}
    parentNode = None
    truthValue = 2
    nodeType = 0
    globalTime = dtnu.globalTime

    #Static lists and dicts
    globalFreeConstraints = dtnu.globalFreeConstraints
    setOfControllables = dtnu.setOfControllables
    setOfUncontrollables = dtnu.setOfUncontrollables

    #Update time
    if elapsedTime != 0:
        for key in remainingFreeConstraints.keys():
            for disjunct in remainingFreeConstraints[key]:
                for conjunct in disjunct:
                    if not isDistance(conjunct):
                        if key == conjunct[0]:
                            conjunct[1] = conjunct[1] - elapsedTime
                            conjunct[2] = conjunct[2] - elapsedTime
                            if conjunct[2] < 0 and not key in tasksElapsedToCheck:
                                tasksElapsedToCheck.append(key)

        for key in activatedUncontrollables:
            for bound in activatedUncontrollables[key]:
                bound[0] = bound[0] - elapsedTime
                bound[1] = bound[1] - elapsedTime

        for key in relativeOccurenceTimes:
            relativeOccurenceTimes[key][0] = relativeOccurenceTimes[key][0] - elapsedTime
            relativeOccurenceTimes[key][1] = relativeOccurenceTimes[key][1] - elapsedTime

        globalTime = globalTime + elapsedTime

    #Update tasks scheduled
    for task in listOfTasks:

        x = -elapsedTime
        y = 0

        '''
        if task in instantReacts:
            uncon = instantReacts[task]
            if uncon in activatedUncontrollables:
                unconlb = activatedUncontrollables[uncon][0]
                unconub = activatedUncontrollables[uncon][1]
                if unconlb > x:
                    x = unconlb
                if unconub < y:
                    y = unconub
        '''

        if task in remainingContingencyLinks:
            contingent = remainingContingencyLinks[task]
            del remainingContingencyLinks[task]
            v = contingent[0]
            listOfBounds = contingent[1]
            activatedUncontrollables[v] = []
            for bound in listOfBounds:
                lb = bound[0]
                ub = bound[1]
                activatedUncontrollables[v].append([x + lb, y + ub])

        relativeOccurenceTimes[task] = [x,y]

        globalOccurenceTimes[task] = [globalTime - elapsedTime, globalTime]

    for task in listOfTasks:

        if task in activatedUncontrollables:
            del activatedUncontrollables[task]

        if task in remainingControllables:
            remainingControllables.remove(task)

        if task in remainingUncontrollables:
            remainingUncontrollables.remove(task)

    return Dtnu(remainingFreeConstraints, globalFreeConstraints,remainingContingencyLinks,
                 activatedUncontrollables, relativeOccurenceTimes, globalOccurenceTimes,
                 remainingControllables, setOfControllables,
                 remainingUncontrollables, setOfUncontrollables, lastControllableScheduled,
                 occuredUncontrollablesDuringLastWait, lastWaitTime, instantReacts, childOr,
                tasksElapsedToCheck, controllableSetTimeAfterUncontrollable, scheduledCombinations,
                scheduledPointsCombination, parentNode, truthValue, nodeType, globalTime)

def possibilitiesMaker(dtnu):
    #Ins react
    return True

#Returns true if conjunct true at this node, returns false if false, returns simplified conjunct if unknown at this point
#This is a relative check ! Bounds are supposed to be >0 since simplifications are done at each tree node
def relativeCheckConjunctSimplify(conjunct, relativeOccurenceTimes, instantReacts,
                                  controllableSetTimeAfterUncontrollable):
    if isDistance(conjunct):
        timePointNumber1 = conjunct[0]
        timePointNumber2 = conjunct[1]
        a = conjunct[2]
        b = conjunct[3]

        #Case 1: Neither are known
        if not timePointNumber1 in relativeOccurenceTimes and not timePointNumber2 in relativeOccurenceTimes:
            return conjunct

        #Case 2: Both are known (two uncontrollables which happened), return true or false:
        if timePointNumber1 in relativeOccurenceTimes and timePointNumber2 in relativeOccurenceTimes:
            #Check if instant reaction solved conjunct
            if timePointNumber1 in instantReacts:
                if instantReacts[timePointNumber1] == timePointNumber2:
                    if b == 0:
                        return True
            if timePointNumber2 in instantReacts:
                if instantReacts[timePointNumber2] == timePointNumber1:
                    if a == 0:
                        return True

            timePointNumber1Occurence = relativeOccurenceTimes[timePointNumber1]
            x1 = timePointNumber1Occurence[0]
            y1 = timePointNumber1Occurence[1]
            timePointNumber2Occurence = relativeOccurenceTimes[timePointNumber2]
            x2 = timePointNumber2Occurence[0]
            y2 = timePointNumber2Occurence[1]

            lowerTruth = False
            upperTruth = False

            if x2 >= y1 - b:
                lowerTruth = True

            if y2 <= x1 - a:
                upperTruth = True

            if lowerTruth and upperTruth:
                return True
            else:
                return False

        #Case 3: one is known
        if timePointNumber1 in relativeOccurenceTimes and not timePointNumber2 in relativeOccurenceTimes:
            timePointNumber1Occurence = relativeOccurenceTimes[timePointNumber1]
            x1 = timePointNumber1Occurence[0]
            y1 = timePointNumber1Occurence[1]

            '''
            #case where a controllable happens a set time after an uncontrollable happens (or reactive)
            if x1 != y1 and a == b and a < 0:
                relativeOccurenceTimes[timePointNumber2] = [x1 - a, y1 - a]
                controllableSetTimeAfterUncontrollable[timePointNumber1] = [timePointNumber2, -a]
                return True
                Also activate contin links
                simplify before using cplex
            '''

            if x1 - a < 0:
                return False
            else:
                return [timePointNumber2, y1 - b, x1 - a]

        if not timePointNumber1 in relativeOccurenceTimes and timePointNumber2 in relativeOccurenceTimes:
            timePointNumber2Occurence = relativeOccurenceTimes[timePointNumber2]
            x2 = timePointNumber2Occurence[0]
            y2 = timePointNumber2Occurence[1]

            '''
            #case where a controllable happens a set time after an uncontrollable happens (or reactive)
            if x2 != y2 and a == b and a > 0:
                relativeOccurenceTimes[timePointNumber1] = [x2 + a, y2 + a]
                controllableSetTimeAfterUncontrollable[timePointNumber2] = [timePointNumber1, a]
                return True
            '''

            if x2 + b < 0:
                return False
            else:
                return [timePointNumber1, y2 + a, x2 + b]

    else:
        timePointNumber = conjunct[0]
        a = conjunct[1]
        b = conjunct[2]

        if b < 0:
            return False

        if timePointNumber in relativeOccurenceTimes:
            timePointrelativeOccurenceTime = relativeOccurenceTimes[timePointNumber]
            x = timePointrelativeOccurenceTime[0]
            y = timePointrelativeOccurenceTime[1]
            lowerTruth = False
            upperTruth = False

            if x >= a:
                lowerTruth = True

            if y <= b:
                upperTruth = True

            if lowerTruth and upperTruth:
                return True
            else:
                return False

        else:
            return conjunct

#Returns 2 if no violation found, Returns 1 if solved, Returns 0 if a clear violation is found
def freeConstraintsCheckForTasksSimplify(dtnu, listofTasks, train, start_time, time_out):

    listofTasks = listofTasks + dtnu.tasksElapsedToCheck

    for taskNumber in listofTasks:
        constraintsToCheck = dtnu.remainingFreeConstraints[taskNumber]
        i = 0
        disjunctConstraintsToRemoveInd = []
        for disjunctiveConstraint in constraintsToCheck:
            j = 0
            conjunctConstraintsToRemoveInd = []
            isValidatedDisjunct = False
            for conjunct in disjunctiveConstraint:
                checkResult = relativeCheckConjunctSimplify(conjunct, dtnu.relativeOccurenceTimes,
                                                            dtnu.instantReacts,
                                                            dtnu.controllableSetTimeAfterUncontrollable)

                if checkResult == True:
                    disjunctConstraintsToRemoveInd.append(i)
                    isValidatedDisjunct = True
                    break

                elif checkResult == False:
                    conjunctConstraintsToRemoveInd.append(j)

                else:
                    simplifiedConjunct = checkResult
                    disjunctiveConstraint[j] = simplifiedConjunct
                j = j + 1

            if len(disjunctiveConstraint) == 0:
                isValidatedDisjunct = True
            # Remove false conjuncts to simplify
            if not isValidatedDisjunct:
                for k1 in range(len(conjunctConstraintsToRemoveInd)):
                    ind = conjunctConstraintsToRemoveInd[k1]
                    del disjunctiveConstraint[ind]
                    for k2 in range(len(conjunctConstraintsToRemoveInd)):
                        conjunctConstraintsToRemoveInd[k2] = conjunctConstraintsToRemoveInd[k2] - 1
                if len(disjunctiveConstraint) == 0:
                    #DTNU is found to be False
                    dtnu.truthValue = 0
                    return 0

            i = i + 1


        # remove disjuncts to simplify
        for i in range(len(disjunctConstraintsToRemoveInd)):
            ind = disjunctConstraintsToRemoveInd[i]
            while len(constraintsToCheck[ind]) > 0:
                del constraintsToCheck[ind][0]

    res = isSolved(dtnu, train, start_time, time_out)
    if res == 1:
        #DTNU is found to be True
        dtnu.truthValue = 1
        return 1
    elif res == 2:
        #DTNU is still unknown
        return 2
    else:
        dtnu.truthValue = 0
        return 0

#Tests all free contraints. Simplifies the DTNU
def completeCheckSimplify(dtnu):
    return True

#Tests constraints for remaining controllables. Must be done on a copy of DTNU before creating sub-DTNUs. Does not simplify
def controllablesCheckAfterWaitSimplify(dtnu):
    return True

# Tests constraints for uncontrollables. Must be done on a sub-DTNU. Simplifies the sub-DTNU
def checkAfterWaitSimplify(dtnu):
    #First check if a scheduled controllable in a FC has a negative occurence interval

    #Then check each sub-DTNU


    return True

# Tests constraints after a controllable is scheduled. Must be done on a sub-DTNU. Simplifies the sub-DTNU
# Returns True if no violation, False is clear violation
def taskScheduledCheckSimplify(dtnu, train, start_time, time_out):
    taskNumber = dtnu.lastControllableScheduled
    return freeConstraintsCheckForTasksSimplify(dtnu, [taskNumber], train, start_time, time_out)

def possibleOutcomes(dtnu, waitDuration):
    return

def explore(treeNode, window, visited_nodes_num, visited_or_nodes_num, visited_and_nodes_num, explore_prob, train, order, start_time, time_out, or_depth, and_depth):

    if time.time() - start_time > time_out:
        raise timeout_decorator.timeout_decorator.TimeoutError

    cut = parentCheckCut(treeNode)
    if cut:
        if random.random() - explore_prob >= 0 and not (train and or_depth == 0):
            return

    visited_nodes_num[0] = visited_nodes_num[0] + 1

    #If node is DTNU
    if treeNode.nodeType == 0:
        childOr = OrNode(None, None, 0, [], 0, [], treeNode, 2, 1, treeNode.globalTime)
        treeNode.childOr = childOr
        explore(childOr, window, visited_nodes_num, visited_or_nodes_num, visited_and_nodes_num, explore_prob, train, order, start_time, time_out, or_depth, and_depth)
        return


    #If node is OR
    if treeNode.nodeType == 1:
        #If parent is DTNU
        if treeNode.parentNode.nodeType == 0:
            explored_false_children = False
            or_depth = or_depth + 1
            if train and or_depth == 3:
                explore_prob = 0
            visited_or_nodes_num[0] = visited_or_nodes_num[0] + 1
            parentDtnu = treeNode.parentNode
            #For learning part, order tasks in tasksToCheckAndExpllore
            tasksToCheckAndExplore = []
            for task in parentDtnu.remainingControllables:
                subDtnu = subDtnuMaker(parentDtnu, 0, [task], {})
                subDtnu.lastControllableScheduled = task
                subDtnu.parentNode = treeNode
                treeNode.numberOfChildren = treeNode.numberOfChildren + 1
                treeNode.listOfChildren.append(subDtnu)
                tasksToCheckAndExplore.append(subDtnu)
            if isThereNonDistanceConstraint(parentDtnu):
                dynamicWait = delta
                min = boundDetection(parentDtnu.remainingFreeConstraints, parentDtnu.activatedUncontrollables)
                if min < dynamicWait:
                    dynamicWait = min
                w = WaitNode(dynamicWait, None, treeNode, 2, 3, parentDtnu.globalTime)
                treeNode.numberOfChildren = treeNode.numberOfChildren + 1
                treeNode.listOfChildren.append(w)
                tasksToCheckAndExplore.append(w)

            if order == True and or_depth < or_depth_order_lim:
                X, edge_index, edge_attr, timepoint_list, timepoint_to_node, \
                timepoint_count = dtnuToGraph(treeNode.parentNode)
                data = Data(x = X, edge_index = edge_index, edge_attr = edge_attr)

                data = data.to(device)
                sig_x, x_hat = model(data)
                sig_x = torch.t(sig_x)[0]
                probList = []
                for childNode in tasksToCheckAndExplore:
                    if childNode.nodeType == 0:
                        controllable = childNode.lastControllableScheduled
                        controllable_ind = timepoint_to_node[controllable]
                        probList.append(sig_x[controllable_ind].item())
                    else:
                        controllable_ind = timepoint_to_node['wait']
                        #probList.append(sig_x[controllable_ind].item())
                        probList.append(pr_wait)
                #print(probList)


                tasksToCheckAndExplore = [x for _, x in sorted(zip(probList, tasksToCheckAndExplore),
                                                               reverse=True, key = operator.itemgetter(0))]



                data = 0
                sig_x = 0
                x_hat = 0

            if train and not order:
                if random.random() > 0.5:
                    random.shuffle(tasksToCheckAndExplore)
                else:
                    tasksToCheckAndExplore.reverse()

            for childNode in tasksToCheckAndExplore:
                if childNode.nodeType == 0:
                    checkResult = freeConstraintsCheckForTasksSimplify(childNode, [childNode.lastControllableScheduled],
                                                                       train, start_time, time_out)
                    if checkResult == 1:
                        propagateTrue(childNode)
                        childNode.visitedChildrenNumber = 0
                    elif checkResult == 0:
                        propagateFalse(childNode)
                        childNode.visitedChildrenNumber = 0
                    else:
                        if train and or_depth == 0:
                            for k in range(child_attempts):
                                #print("\n child try = %d"%k)
                                rcp = copy.deepcopy(childNode)
                                try:
                                    #print("cur c = %d"%childNode.truthValue)
                                    #print("cur = %d" % treeNode.truthValue)
                                    vb = visited_nodes_num[0]
                                    tb = time.time()
                                    combination, cParent = notExistCombinationPoints(rcp)
                                    if not combination in cParent.scheduledPointsCombination:
                                        explore(rcp, window, visited_nodes_num, visited_or_nodes_num,
                                                visited_and_nodes_num, 0, train, False, time.time(),
                                                child_timeout, or_depth, and_depth)
                                        cParent.scheduledPointsCombination[combination] = [rcp.truthValue, rcp]
                                    else:
                                        [rcp.truthValue, pointing_dtnu] = cParent.scheduledPointsCombination[combination]
                                        if rcp.truthValue:
                                            propagateTrue(rcp)
                                        else:
                                            propagateFalse(rcp)

                                    va = visited_nodes_num[0]
                                    ta = time.time()
                                    print('child finished on attempt %d, value = %d visited nodes = %d, time = %.2f'%(k, rcp.truthValue, va - vb, float(ta - tb)))
                                    childNode.visitedChildrenNumber = va - vb
                                    childNode.truthValue = rcp.truthValue
                                    if childNode.truthValue:
                                        propagateTrue(childNode)
                                    else:
                                        propagateFalse(childNode)
                                    break

                                except timeout_decorator.timeout_decorator.TimeoutError:
                                    #print("Child Timer expired")
                                    childNode.childOr = None

                            if childNode.truthValue == 2:
                                print("No simulation led to a solution for this child, all timeouts")
                                childNode.truthValue = 0
                                explored_false_children = True
                                propagateFalse(childNode)
                                childNode.visitedChildrenNumber = visited_nodes_num[0] - vb


                        else:
                            if train:
                                vb = visited_nodes_num[0]
                                combination, cParent = notExistCombinationPoints(childNode)
                                if not combination in cParent.scheduledPointsCombination:
                                    explore(childNode, window, visited_nodes_num, visited_or_nodes_num,
                                            visited_and_nodes_num, 0, train, False, start_time,
                                            time_out, or_depth, and_depth)
                                    cParent.scheduledPointsCombination[combination] = [childNode.truthValue, childNode]
                                else:
                                    [childNode.truthValue, pointing_dtnu] = cParent.scheduledPointsCombination[combination]
                                    if childNode.truthValue:
                                        propagateTrue(childNode)
                                    else:
                                        propagateFalse(childNode)
                                va = visited_nodes_num[0]

                            else:
                                combination, cParent = notExistCombinationPoints(childNode)
                                if not combination in cParent.scheduledPointsCombination:
                                    explore(childNode, window, visited_nodes_num, visited_or_nodes_num,
                                            visited_and_nodes_num, explore_prob, train, order,
                                            start_time, time_out, or_depth, and_depth)
                                    cParent.scheduledPointsCombination[combination] = [childNode.truthValue, childNode]
                                else:
                                    [childNode.truthValue, pointing_dtnu] = cParent.scheduledPointsCombination[combination]
                                    if childNode.truthValue:
                                        propagateTrue(childNode)
                                    else:
                                        propagateFalse(childNode)
                                    if treeNode.validatedChild == childNode:
                                        treeNode.validatedChild = pointing_dtnu
                                    if childNode in treeNode.listOfFalseChildren:
                                        treeNode.listOfFalseChildren.remove(childNode)
                                        treeNode.listOfFalseChildren.append(pointing_dtnu)
                                    if childNode in treeNode.listOfChildren:
                                        treeNode.listOfChildren.remove(childNode)
                                        treeNode.listOfChildren.append(pointing_dtnu)

                else:
                    if train and or_depth == 0:
                        for k in range(child_attempts):
                            rcp = copy.deepcopy(childNode)
                            try:
                                vb = visited_nodes_num[0]
                                tb = time.time()
                                combination, cParent = notExistCombination(treeNode)
                                if not combination in cParent.scheduledCombinations:
                                    #cParent.scheduledCombinations[combination] = 2
                                    explore(rcp, window, visited_nodes_num, visited_or_nodes_num, visited_and_nodes_num,
                                            0, train, False, time.time(), child_timeout, or_depth, and_depth)
                                    cParent.scheduledCombinations[combination] = [rcp.truthValue, rcp]
                                else:
                                    [rcp.truthValue, pointing_dtnu] = cParent.scheduledCombinations[combination]
                                    if rcp.truthValue:
                                        propagateTrue(rcp)
                                    else:
                                        propagateFalse(rcp)
                                va = visited_nodes_num[0]
                                ta = time.time()
                                print('child finished on try %d, value = %d visited nodes = %d, time = %.2f'%(k, rcp.truthValue, va - vb, float(ta - tb)))
                                childNode.truthValue = rcp.truthValue
                                childNode.visitedChildrenNumber = va - vb
                                if childNode.truthValue:
                                    propagateTrue(childNode)
                                else:
                                    propagateFalse(childNode)
                                break

                            except timeout_decorator.timeout_decorator.TimeoutError:
                                #print("Child Timer expired")
                                childNode.childOr = None
                        if childNode.truthValue == 2:
                            print("No simulation led to a solution for this child, all timeout")
                            childNode.truthValue = 0
                            explored_false_children = True
                            propagateFalse(childNode)
                            childNode.visitedChildrenNumber = visited_nodes_num[0] - vb

                    else:
                        if train:
                            vb = visited_nodes_num[0]
                            combination, cParent = notExistCombination(treeNode)
                            if not combination in cParent.scheduledCombinations:
                                #cParent.scheduledCombinations[combination] = 2
                                explore(childNode, window, visited_nodes_num, visited_or_nodes_num, visited_and_nodes_num,
                                        0, train, False, start_time, time_out, or_depth, and_depth)
                                cParent.scheduledCombinations[combination] = [childNode.truthValue, childNode]
                            else:
                                [childNode.truthValue, pointing_dtnu] = cParent.scheduledCombinations[combination]
                                if childNode.truthValue:
                                    propagateTrue(childNode)
                                else:
                                    propagateFalse(childNode)
                            va = visited_nodes_num[0]

                        else:
                            combination, cParent = notExistCombination(treeNode)
                            if not combination in cParent.scheduledCombinations:
                                #cParent.scheduledCombinations[combination] = 2
                                explore(childNode, window, visited_nodes_num, visited_or_nodes_num, visited_and_nodes_num,
                                        explore_prob, train, order, start_time, time_out, or_depth, and_depth)
                                cParent.scheduledCombinations[combination] = [childNode.truthValue, childNode]
                            else:
                                [childNode.truthValue, pointing_dtnu] = cParent.scheduledCombinations[combination]
                                if childNode.truthValue:
                                    propagateTrue(childNode)
                                else:
                                    propagateFalse(childNode)
                                if treeNode.validatedChild == childNode:
                                    treeNode.validatedChild = pointing_dtnu
                                if childNode in treeNode.listOfFalseChildren:
                                    treeNode.listOfFalseChildren.remove(childNode)
                                    treeNode.listOfFalseChildren.append(pointing_dtnu)
                                if childNode in treeNode.listOfChildren:
                                    treeNode.listOfChildren.remove(childNode)
                                    treeNode.listOfChildren.append(pointing_dtnu)

                cut = currentCheckCut(treeNode)
                if cut:
                    if random.random() - explore_prob >= 0:
                        if train and (((random.random() * math.exp(
                                - math.sqrt(or_depth + 100000) + 0)) > random.random()) or or_depth == 0):
                            if or_depth < 2:
                                genData(treeNode, window)
                                print(or_depth)
                                print(treeNode.truthValue)
                            else:
                                if (random.random() * math.exp(- 3 * math.sqrt(or_depth + 1) + 0)) > random.random() or\
                                        (treeNode.truthValue == 1 and (random.random() * math.exp(- 2 * math.sqrt(or_depth + 1) + 5)) > random.random()):
                                    genData(treeNode, window)
                                    print(or_depth)
                                    print(treeNode.truthValue)
                        if train:
                            treeNode.listOfChildren = None
                            treeNode.listOfFalseChildren = None
                        return

            if train and (((random.random() * math.exp(- math.sqrt(or_depth + 100000) + 0)) > random.random()) or or_depth == 0)\
                    and ((treeNode.truthValue == 1 and explored_false_children) or
                         ((treeNode.truthValue == 1 or treeNode.truthValue == 0) and random.random() > 0.8)):
                    if or_depth < 2:
                        genData(treeNode, window)
                        print(or_depth)
                        print(treeNode.truthValue)
                    else:
                        if (random.random() * math.exp(- or_depth)) > random.random():
                            genData(treeNode, window)
                            print(or_depth)
                            print(treeNode.truthValue)

            if train:
                treeNode.listOfChildren = None
                treeNode.listOfFalseChildren = None
            return

        #If parent is wait, inst react
        if treeNode.parentNode.nodeType == 3:
            instantReacts = treeNode.instantReacts
            combination_of_reacts = []
            for i in range(len(instantReacts.keys())+1):
                combination_of_reacts = combination_of_reacts + list(
                    map(dict, itertools.combinations(instantReacts.items(), i)))

            #For each reaction combination
            andNodesToExplore = []
            for combination in combination_of_reacts:

                childAnd = AndNode(combination, 0, [], 0, [], treeNode, 2, 2, treeNode.parentNode.parentNode.parentNode.globalTime)
                treeNode.numberOfChildren = treeNode.numberOfChildren + 1
                treeNode.listOfChildren.append(childAnd)
                andNodesToExplore.append(childAnd)

            for andNode in andNodesToExplore:

                explore(andNode, window, visited_nodes_num, visited_or_nodes_num, visited_and_nodes_num, explore_prob,
                        train, order, start_time, time_out, or_depth, and_depth)
                cut = currentCheckCut(treeNode)
                if cut:
                    return
            return

    #If tree node is AND node
    if treeNode.nodeType == 2:
        and_depth = and_depth + 1
        visited_and_nodes_num[0] = visited_and_nodes_num[0] + 1
        waitDuration = treeNode.parentNode.parentNode.waitDuration
        parentDtnu = treeNode.parentNode.parentNode.parentNode.parentNode
        # For learning part, order possible sub dtnu in possibleSubDtnusToExplore
        possibleSubDtnusToExplore = []
        possibleUncontrollables = []
        certainUncontrollables = []


        #Case where reactive controllable activates uncontrollable link which is also in the wait duratio, To be done later

        for key in parentDtnu.activatedUncontrollables:
            listOfBounds = parentDtnu.activatedUncontrollables[key]
            for bound in listOfBounds:
                lb = bound[0]
                ub = bound[1]
                if lb < waitDuration:
                    if ub > waitDuration:
                        if not key in possibleUncontrollables and not key in certainUncontrollables:
                            possibleUncontrollables.append(key)
                    elif ub > 0:
                        max = 0
                        for bound2 in listOfBounds:
                            if bound2[1] > max:
                                max = bound2[1]
                        if ub == max:
                            if not key in certainUncontrollables:
                                certainUncontrollables.append(key)
                                if key in possibleUncontrollables:
                                    del possibleUncontrollables[key]
                        else:
                            if not key in possibleUncontrollables and not key in certainUncontrollables:
                                possibleUncontrollables.append(key)

        combination_of_uncontrollables = []
        for i in range(len(possibleUncontrollables) + 1):
            combination_of_uncontrollables = combination_of_uncontrollables + list(
                map(list, list(itertools.combinations(possibleUncontrollables, i))))

        for i in range(len(combination_of_uncontrollables)):
            combination_of_uncontrollables[i] = combination_of_uncontrollables[i] + certainUncontrollables

        subDtnusToCheck = []
        subDtnusToExplore = []
        for combination in combination_of_uncontrollables:
            # Add controllables which reacted to the list of tasks to process
            for key in treeNode.instantReacts.keys():
                if treeNode.instantReacts[key] in combination and key not in combination:
                    combination.append(key)
            subDtnu = subDtnuMaker(parentDtnu, waitDuration, combination, treeNode.instantReacts)
            subDtnu.occuredUncontrollablesDuringLastWait = combination
            subDtnu.parentNode = treeNode
            treeNode.numberOfChildren = treeNode.numberOfChildren + 1
            treeNode.listOfChildren.append(subDtnu)
            subDtnusToCheck.append(subDtnu)

        for subDtnu in subDtnusToCheck:
            checkResult = freeConstraintsCheckForTasksSimplify(subDtnu,
                                                               subDtnu.occuredUncontrollablesDuringLastWait,
                                                               train, start_time, time_out)

            if checkResult == 1:
                propagateTrue(subDtnu)
            elif checkResult == 0:
                propagateFalse(subDtnu)
            else:
                subDtnusToExplore.append(subDtnu)

            cut = currentCheckCut(treeNode)
            if cut:
                return
        
        if train and not order:
            random.shuffle(subDtnusToExplore)

        if order == True and len(subDtnusToExplore) > 1 and and_depth < and_depth_order_lim:
            probList = []
            for subDtnu in subDtnusToExplore:
                X, edge_index, edge_attr, timepoint_list, timepoint_to_node, \
                timepoint_count = dtnuToGraph(subDtnu)
                data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr)
                data = data.to(device)
                sig_x, x_hat = model(data)
                sig_x = torch.t(sig_x)[0]
                prob_full_zero = 1

                for available_action in subDtnu.remainingControllables:
                    prob_full_zero = prob_full_zero * (1 - sig_x[timepoint_to_node[available_action]].item())
                if isThereNonDistanceConstraint(subDtnu):
                    prob_full_zero = prob_full_zero * (1 - sig_x[timepoint_to_node['wait']].item())
                probList.append(1 - prob_full_zero)

                data = 0
                sig_x = 0
                x_hat = 0

            subDtnusToExplore = [x for _, x in sorted(zip(probList, subDtnusToExplore), key=operator.itemgetter(0))]

        for subDtnu in subDtnusToExplore:
            explore(subDtnu, window, visited_nodes_num, visited_or_nodes_num, visited_and_nodes_num, explore_prob,
                    train, order, start_time, time_out, or_depth, and_depth)

            cut = currentCheckCut(treeNode)
            if cut:
                return

        return

    #If node is a wait node
    if treeNode.nodeType == 3:
        parentDtnu = treeNode.parentNode.parentNode
        waitDuration = treeNode.waitDuration
        instantReacts = buildReacts(parentDtnu, waitDuration)
        childOr = OrNode(instantReacts, None, 0, [], 0, [], treeNode, 2, 1, parentDtnu.globalTime)
        treeNode.childOr = childOr
        explore(childOr, window, visited_nodes_num, visited_or_nodes_num, visited_and_nodes_num, explore_prob, train,
                order, start_time, time_out, or_depth, and_depth)
        return

    #CUT check: check parents
    #DETERMINE node type

    #completion check






    # use taskScheduledCheckSimplify(dtnu), if True, ok, if False, Propagate False !
    #Check if node is solved with isSolved also, if True, propagate True !


    #If AND: CHECK ALL possible children which can be obtained from parent before exploring any !!!!
    #So create a list of them, check them all, the explore list

    #use possibilitesMaker, include ins react

    #When considering children
    #First make children woth sub dtnu maker, then check them to simplify and cut, only then explore of valid

    #If DTNU go to OR
    #if

    #Structure is like this:
    #
    # Example for OR:
    #Create list of children sub DNTNU
    #Add them to the list of children of current node (also update number of childre,) for each of them add this node as parent
    # FOR EACH SUB DTNU
        # Run FreeConstraintCheckSimplify.
        # If 1 returned, propagateTruth(sub-DTNU)

        # if 0 propagateFAlse(sub-DTNU),

        # if 2, explore(sub-dtnu)
        # Run checkCurrentNodeCut

    # Example for AND:
    #Create list of children sub DNTNU
    #Add them to the list of children of current node (update num of children), for each of them add this node as parent
    # FOR EACH SUB DTNU
        # Run FreeConstraintCheckSimplify.
        # If 1 returned, propagateTruth(sub-DTNU)

        # if 0 propagateFAlse(sub-DTNU),

        # if 2, add sub-dtnu to list to explore
        # Run checkCurrentNodeCut

    #For dtnu in explore list
        #epxlore (dtnu)

def createDNTU(setOfControllables, setOfUncontrollables, freeConstraints, contingencyLinks):

    remainingFreeConstraints = {}
    remainingContingencyLinks = {}
    activatedUncontrollables = {}
    relativeOccurenceTimes = {}
    globalOccurenceTimes = {}
    remainingControllables = []
    remainingUncontrollables = []
    lastControllableScheduled = None
    occuredUncontrollablesDuringLastWait = None
    lastWaitTime = 0
    instantReacts = {}
    childOr = None
    tasksElapsedToCheck = []
    controllableSetTimeAfterUncontrollable = {}
    scheduledCombinations = {}
    scheduledPointsCombination = {}
    parentNode = None
    truthValue = 2
    nodeType = 0
    globalTime = 0

    #Static lists and dicts
    globalFreeConstraints = {}

    for controllable in setOfControllables:
        remainingFreeConstraints[controllable] = []

    for uncontrollable in setOfUncontrollables:
        remainingFreeConstraints[uncontrollable] = []

    for disjunct in freeConstraints:
        for conjunct in disjunct:
            if isDistance(conjunct):
                timePoint1 = conjunct[0]
                timePoint2 = conjunct[1]
                if not disjunct in remainingFreeConstraints[timePoint1]:
                    remainingFreeConstraints[timePoint1].append(disjunct)
                if not disjunct in remainingFreeConstraints[timePoint2]:
                    remainingFreeConstraints[timePoint2].append(disjunct)

            else:
                timePoint1 = conjunct[0]
                if not disjunct in remainingFreeConstraints[timePoint1]:
                    remainingFreeConstraints[timePoint1].append(disjunct)

    globalFreeConstraints = copy.deepcopy(remainingFreeConstraints)

    remainingContingencyLinks = contingencyLinks

    remainingControllables = copy.deepcopy(setOfControllables)

    remainingUncontrollables = copy.deepcopy(setOfUncontrollables)

    return Dtnu(remainingFreeConstraints, globalFreeConstraints,remainingContingencyLinks,
                 activatedUncontrollables, relativeOccurenceTimes, globalOccurenceTimes,
                 remainingControllables, setOfControllables,
                 remainingUncontrollables, setOfUncontrollables, lastControllableScheduled,
                 occuredUncontrollablesDuringLastWait, lastWaitTime, instantReacts, childOr,
                tasksElapsedToCheck, controllableSetTimeAfterUncontrollable, scheduledCombinations,
                scheduledPointsCombination, parentNode, truthValue, nodeType, globalTime)

#@timeout(10)
#@timeout_decorator.timeout(20)
def treeSearch(originalDtnu, window, visited_nodes_num, visited_or_nodes_num, visited_and_nodes_num, explore_prob,
               train, order, start_time, time_out):

    prelimCheck = isSolved(originalDtnu, train, start_time, time_out)
    if prelimCheck == 1:
        print("\n DTN solved")
        originalDtnu.truthValue = 1
    elif prelimCheck == 0:
        print("\n DTN solved")
        originalDtnu.truthValue = 0
    else:
        explore(originalDtnu, window, visited_nodes_num, visited_or_nodes_num, visited_and_nodes_num, explore_prob, train, order, start_time, time_out, -1, -1)

    if originalDtnu.truthValue == 1:
        print("\nStrategy found")
    else:
        print("\nNo strategy found")

def varReader(var_name):
    var_tab = var_name.split('_')
    var_ind = int(var_tab[0][1:])
    var_start_end = var_tab[1]
    if var_start_end == 'start':
        return 2 * var_ind
    else:
        return 2 * var_ind + 1

def readAllandroData(filename):

    setOfControllables = []
    setOfUncontrollables = []
    freeConstraints = []
    contingencyLinks = {}

    try:
        f = open(filename, 'r')
    except:
        print("File not found")
    f1 = f.readlines()
    sp = f1[0].replace("\n","").split(" ")
    var_num = int(sp[0])
    const_num = int(sp[1])

    listOfVars = f1[1: var_num + 1]
    listOfConstraints = f1[var_num + 1:]

    for var in listOfVars:
        var = var.replace("\n","")
        var = var.split(" ")
        var_name = var[0]
        var_type = var[1]

        var_ind = varReader(var_name)
        if var_type == 'c' and not var_ind in setOfControllables:
            setOfControllables.append(var_ind)
        if var_type == 'u' and not var_ind in setOfUncontrollables:
            setOfUncontrollables.append(var_ind)

    for constraint in listOfConstraints:
        constraint = constraint.replace("\n", "")
        init_space = constraint.split(" ")
        number_of_conjuncts = int(init_space[0])
        t = init_space[1]
        if t == 'f':
            remain = init_space[2:]
            disjunct = []
            for i in range(number_of_conjuncts):
                if i != 0:
                    remain = remain[4:]
                x_start = varReader(remain[0])
                x_end = varReader(remain[1])
                a = Decimal(remain[2])
                b = Decimal(remain[3])
                disjunct.append([x_end, x_start,a,b])
            freeConstraints.append(disjunct)

        else:
            remain = init_space[2:]
            for i in range(number_of_conjuncts):
                if i != 0:
                    remain = remain[4:]
                x_start = varReader(remain[0])
                x_end = varReader(remain[1])
                a = Decimal(remain[2])
                b = Decimal(remain[3])

                if not x_start in contingencyLinks:
                    contingencyLinks[x_start] = [x_end,[]]

                contingencyLinks[x_start][1].append([a,b])



    return createDNTU(setOfControllables, setOfUncontrollables, freeConstraints, contingencyLinks)

def distanceToClass(distance, max):

    if distance <=0:
        return 0

    if max > 0.1:
        cate = distance / max * 10
    else:
        return 0

    if cate > 0 and cate <= 1:
        return 0
    if cate > 1 and cate <= 2:
        return 1
    if cate > 2 and cate <= 3:
        return 2
    if cate > 3 and cate <= 4:
        return 3
    if cate > 4 and cate <= 5:
        return 4
    if cate > 5 and cate <= 6:
        return 5
    if cate > 6 and cate <= 7:
        return 6
    if cate > 7 and cate <= 8:
        return 7
    if cate > 8 and cate <= 9:
        return 8
    if cate > 9:
        return 9

def dtnuToGraph(dtnu):
    # node features
    # 0 START
    # 1 Disjunction node
    # 2 Controllable
    # 3 Uncontrollable

    # edge features
    # 0 - 9 : distance classes
    #10 - sign, 1 if -
    # 11: Normal link
    # 12: Constraint link
    # 13: Disjunction link
    # 14: Contingency link
    # 15: connected to wait (every real node)

    freeConstraints = dtnu.remainingFreeConstraints
    activatedUncontrollables = dtnu.activatedUncontrollables
    remainingContingencyLinks = dtnu.remainingContingencyLinks
    controllables = dtnu.remainingControllables

    X = []
    edge_index = [[],[]]
    edge_attr = []
    max_val = 0
    node_index = []
    timepoint_to_node = {}
    list_of_disjuncts = []
    timepoint_count = 0
    timepoint_list = []

    for key in freeConstraints.keys():
        for disjunct in freeConstraints[key]:
            if not disjunct in list_of_disjuncts:
                list_of_disjuncts.append(disjunct)
                for conjunct in disjunct:
                    if isBounds(conjunct):
                        if not conjunct[0] in timepoint_list:
                            timepoint_list.append(conjunct[0])
                            timepoint_to_node[conjunct[0]] = timepoint_count
                            timepoint_count = timepoint_count + 1
                            if conjunct[0] in controllables:
                                X.append([0,0,1,0])
                            else:
                                X.append([0,0,0,1])

                        a = conjunct[1]
                        b = conjunct[2]
                        if a > max_val and a < 100000:
                            max_val = a
                        if b > max_val and b < 100000:
                            max_val = b

                    elif isDistance(conjunct):
                        if not conjunct[0] in timepoint_list:
                            timepoint_list.append(conjunct[0])
                            timepoint_to_node[conjunct[0]] = timepoint_count
                            timepoint_count = timepoint_count + 1
                            if conjunct[0] in controllables:
                                X.append([0,0,1,0])
                            else:
                                X.append([0,0,0,1])
                        if not conjunct[1] in timepoint_list:
                            timepoint_list.append(conjunct[1])
                            timepoint_to_node[conjunct[1]] = timepoint_count
                            timepoint_count = timepoint_count + 1
                            if conjunct[1] in controllables:
                                X.append([0,0,1,0])
                            else:
                                X.append([0,0,0,1])

                        a = conjunct[2]
                        b = conjunct[3]
                        if a > max_val and a < 100000:
                            max_val = a
                        if b > max_val and b < 100000:
                            max_val = b

    for element in remainingContingencyLinks.keys():
        u = remainingContingencyLinks[element][0]
        l = remainingContingencyLinks[element][1]

        if not element in timepoint_list:
            timepoint_list.append(element)
            timepoint_to_node[element] = timepoint_count
            timepoint_count = timepoint_count + 1
            if element in controllables:
                X.append([0, 0, 1, 0])
            else:
                X.append([0, 0, 0, 1])
        if not u in timepoint_list:
            timepoint_list.append(u)
            timepoint_to_node[u] = timepoint_count
            timepoint_count = timepoint_count + 1
            if u in controllables:
                X.append([0, 0, 1, 0])
            else:
                X.append([0, 0, 0, 1])
        '''
        for conjunct in l:
            a = conjunct[0]
            b = conjunct[1]
            if a > max_val and a < 100000:
                max_val = a
            if b > max_val and b < 100000:
                max_val = b
        '''
    for element in activatedUncontrollables.keys():

        if not element in timepoint_list:
            timepoint_list.append([element])
            timepoint_to_node[element] = timepoint_count
            timepoint_count = timepoint_count + 1
            if element in controllables:
                X.append([0, 0, 1, 0])
            else:
                X.append([0, 0, 0, 1])

        disjunct = activatedUncontrollables[element]

        for conjunct in disjunct:
            a = conjunct[0]
            b = conjunct[1]
            if a > max_val and a < 100000:
                max_val = a
            if b > max_val and b < 100000:
                max_val = b

    for element in controllables:
        if not element in timepoint_list:
            timepoint_list.append([element])
            timepoint_to_node[element] = timepoint_count
            timepoint_count = timepoint_count + 1
            if element in controllables:
                X.append([0, 0, 1, 0])
            else:
                X.append([0, 0, 0, 1])

    timepoint_list.append('wait')
    X.append([1, 0, 0, 0])
    timepoint_to_node['wait'] = timepoint_count
    timepoint_count = timepoint_count + 1
    disjunct_nodes_ind = timepoint_count

    for i in range(len(timepoint_list) - 1):
        edge_index[0].append(i)
        edge_index[1].append(len(timepoint_list) - 1)
        edge_attr.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        edge_index[0].append(len(timepoint_list) - 1)
        edge_index[1].append(i)
        edge_attr.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])


    for disjunct in list_of_disjuncts:
        list_of_disjunctive_nodes = []
        for conjunct in disjunct:
            newDisjunct = disjunct_nodes_ind
            X.append([0, 1, 0, 0])
            list_of_disjunctive_nodes.append(newDisjunct)
            disjunct_nodes_ind = disjunct_nodes_ind + 1
            if isBounds(conjunct):
                x = conjunct[0]
                a = conjunct[1]
                b = conjunct[2]

                edge_index[0].append(timepoint_to_node[x])
                edge_index[1].append(newDisjunct)
                edge_attr.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

                edge_index[0].append(newDisjunct)
                edge_index[1].append(timepoint_to_node[x])
                edge_attr.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

                edge_index[0].append(newDisjunct)
                edge_index[1].append(timepoint_to_node['wait'])
                feature = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]
                feature[distanceToClass(a, max_val)] = 1
                edge_attr.append(feature)

                edge_index[0].append(timepoint_to_node['wait'])
                edge_index[1].append(newDisjunct)
                feature = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
                feature[distanceToClass(b, max_val)] = 1
                edge_attr.append(feature)

            elif isDistance(conjunct):
                x = conjunct[0]
                y = conjunct[1]
                a = conjunct[2]
                b = conjunct[3]

                edge_index[0].append(timepoint_to_node[x])
                edge_index[1].append(newDisjunct)
                edge_attr.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

                edge_index[0].append(newDisjunct)
                edge_index[1].append(timepoint_to_node[x])
                edge_attr.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

                edge_index[0].append(newDisjunct)
                edge_index[1].append(timepoint_to_node[y])
                feature = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]
                feature[distanceToClass(a, max_val)] = 1
                edge_attr.append(feature)

                edge_index[0].append(timepoint_to_node[y])
                edge_index[1].append(newDisjunct)
                feature = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
                feature[distanceToClass(b, max_val)] = 1
                edge_attr.append(feature)

        for element in itertools.combinations(list_of_disjunctive_nodes, 2):
            edge_index[0].append(element[0])
            edge_index[1].append(element[1])
            edge_attr.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
            edge_index[0].append(element[1])
            edge_index[1].append(element[0])
            edge_attr.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

    for u in activatedUncontrollables.keys():
        disjunct = activatedUncontrollables[u]
        list_of_disjunctive_nodes = []
        for conjunct in disjunct:
            newDisjunct = disjunct_nodes_ind
            X.append([0, 1, 0, 0])
            list_of_disjunctive_nodes.append(newDisjunct)
            disjunct_nodes_ind = disjunct_nodes_ind + 1
            a = conjunct[0]
            b = conjunct[1]

            edge_index[0].append(timepoint_to_node[u])
            edge_index[1].append(newDisjunct)
            edge_attr.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

            edge_index[0].append(newDisjunct)
            edge_index[1].append(timepoint_to_node[u])
            edge_attr.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

            edge_index[0].append(newDisjunct)
            edge_index[1].append(timepoint_to_node['wait'])
            feature = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
            feature[distanceToClass(a, max_val)] = 1
            edge_attr.append(feature)

            edge_index[0].append(timepoint_to_node['wait'])
            edge_index[1].append(newDisjunct)
            feature = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
            feature[distanceToClass(b, max_val)] = 1
            edge_attr.append(feature)

        for element in itertools.combinations(list_of_disjunctive_nodes, 2):
            edge_index[0].append(element[0])
            edge_index[1].append(element[1])
            edge_attr.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
            edge_index[0].append(element[1])
            edge_index[1].append(element[0])
            edge_attr.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

    for element in remainingContingencyLinks.keys():
        u = remainingContingencyLinks[element][0]
        l = remainingContingencyLinks[element][1]
        list_of_disjunctive_nodes = []
        for conjunct in l:
            newDisjunct = disjunct_nodes_ind
            X.append([0, 1, 0, 0])
            list_of_disjunctive_nodes.append(newDisjunct)
            disjunct_nodes_ind = disjunct_nodes_ind + 1
            a = conjunct[0]
            b = conjunct[1]

            edge_index[0].append(timepoint_to_node[element])
            edge_index[1].append(newDisjunct)
            feature = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
            feature[distanceToClass(b, max_val)] = 1
            edge_attr.append(feature)

            edge_index[0].append(newDisjunct)
            edge_index[1].append(timepoint_to_node[element])
            feature = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
            feature[distanceToClass(a, max_val)] = 1
            edge_attr.append(feature)

            edge_index[0].append(newDisjunct)
            edge_index[1].append(timepoint_to_node[u])
            edge_attr.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

            edge_index[0].append(timepoint_to_node[u])
            edge_index[1].append(newDisjunct)
            edge_attr.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

        for element in itertools.combinations(list_of_disjunctive_nodes, 2):
            edge_index[0].append(element[0])
            edge_index[1].append(element[1])
            edge_attr.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
            edge_index[0].append(element[1])
            edge_index[1].append(element[0])
            edge_attr.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

    return torch.tensor(X, dtype=torch.float), torch.tensor(edge_index, dtype=torch.long), torch.tensor(edge_attr, dtype=torch.float), timepoint_list, timepoint_to_node, timepoint_count

def genData(node, window):
    if node.nodeType == 1:
        if node.parentNode.nodeType == 0:
            children = node.listOfChildren
            X, edge_index, edge_attr, timepoint_list, timepoint_to_node, timepoint_count = dtnuToGraph(node.parentNode)
            y = torch.zeros(timepoint_count)
            act = torch.zeros(timepoint_count)
            for child in children:
                if child.visitedChildrenNumber > visitedChildrenStorageThreshold:
                    print("Storing, because %d"%child.visitedChildrenNumber)
                    if child.nodeType == 0:
                        controllable = child.lastControllableScheduled
                        controllable_ind = timepoint_to_node[controllable]
                        value = child.truthValue
                        if value == 1:
                            y[controllable_ind] = 1
                            act[controllable_ind] = 1
                        elif value == 0:
                            y[controllable_ind] = 0
                            act[controllable_ind] = 1

                    else:
                        controllable_ind = timepoint_to_node['wait']
                        value = child.truthValue
                        if value == 1:
                            y[controllable_ind] = 1
                            act[controllable_ind] = 1
                        elif value == 0:
                            y[controllable_ind] = 0
                            act[controllable_ind] = 1

                else:
                    print("Not Storing, because %d" % child.visitedChildrenNumber)


            data = Data(x = X, edge_index = edge_index, edge_attr = edge_attr, y = y,
                        num_true_nodes = timepoint_count, activatedIndices = act)
            window.append(data)
            '''
            if len(window) > max_window_size:
                while len(window) > max_window_size:
                    window.pop(0)
            '''
            print(y)
            print(act)

    return

def randomBound(max_scale):

    a = Decimal(random.randint(0, max_scale-1))
    b = Decimal(random.randint(a+1, max_scale))
    return [a, b]

def randomSetOfBounds(max_scale, number_of_bounds):

    max_val = 0
    list_of_bounds = []
    for i in range(number_of_bounds):
        a = Decimal(random.randint(max_val, max_scale-1))
        b = Decimal(random.randint(a+1, max_scale))
        max_val = b
        list_of_bounds.append([a,b])
        if max_val >= max_scale-1:
            break

    return list_of_bounds

def genRandomDtnu(timepoint_num, uncontrollable_num, min_conjuncts, max_conjuncts):

    max_scale = 100

    setOfControllables = []
    setOfUncontrollables = []
    freeConstraints = []
    contingencyLinks = {}

    invoked_points = []
    #Only invoke

    for i in range(timepoint_num):
        setOfControllables.append(i)

    for i in range(timepoint_num, timepoint_num + uncontrollable_num):
        setOfUncontrollables.append(i)

    controllable_temp = copy.deepcopy(setOfControllables)

    for u in setOfUncontrollables:
        chosenControllable = random.choice(controllable_temp)
        invoked_points.append(chosenControllable)
        controllable_temp.remove(chosenControllable)
        number_of_bounds = random.randint(1, 5)
        contingencyLinks[chosenControllable] = [u, randomSetOfBounds(max_scale, number_of_bounds)]

    for chosenPoint in setOfControllables + setOfUncontrollables:
        if (not chosenPoint in invoked_points) or (random.random() > 0.8):
            disjunct = []
            for j in range(random.randint(min_conjuncts, max_conjuncts)):
                typeOfConjunct = random.random()
                if typeOfConjunct >= 0 and typeOfConjunct < 0.1 and len(setOfControllables) > 0:
                    #controllableBound
                    if j == 0:
                        chosenPoint1 = chosenPoint
                    else:
                        chosenPoint1 = random.choice(setOfControllables)
                    if chosenPoint1 in invoked_points:
                        chosenPoint1 = random.choice(setOfControllables)
                    bound = randomBound(max_scale)
                    disjunct.append([chosenPoint1, bound[0], bound[1]])
                    invoked_points.append(chosenPoint1)

                elif typeOfConjunct >= 0.1 and typeOfConjunct < 0.2 and len(setOfUncontrollables) > 0:
                    #UncontrollableBound
                    if j == 0:
                        chosenPoint1 = chosenPoint
                    else:
                        chosenPoint1 = random.choice(setOfUncontrollables)
                    if chosenPoint1 in invoked_points:
                        chosenPoint1 = random.choice(setOfUncontrollables)
                    bound = randomBound(max_scale)
                    disjunct.append([chosenPoint1, bound[0], bound[1]])
                    invoked_points.append(chosenPoint1)


                elif typeOfConjunct >= 0.2 and typeOfConjunct < 0.35 and \
                        len(setOfControllables) > 0 and len(setOfUncontrollables) > 0:
                    #controllable Uncontrollable
                    chosenPoint1 = random.choice(setOfControllables)
                    if chosenPoint1 in invoked_points:
                        chosenPoint1 = random.choice(setOfControllables)
                    if j == 0:
                        chosenPoint2 = chosenPoint
                    else:
                        chosenPoint2 = random.choice(setOfUncontrollables)
                    if chosenPoint2 in invoked_points:
                        chosenPoint2 = random.choice(setOfUncontrollables)
                    bound = randomBound(max_scale)
                    disjunct.append([chosenPoint1, chosenPoint2, bound[0], bound[1]])
                    invoked_points.append(chosenPoint1)
                    invoked_points.append(chosenPoint2)

                elif typeOfConjunct >= 0.35 and typeOfConjunct < 0.5 and \
                        len(setOfControllables) > 0 and len(setOfUncontrollables) > 0:
                    #uncontrollable controllable
                    if j == 0:
                        chosenPoint1 = chosenPoint
                    else:
                        chosenPoint1 = random.choice(setOfUncontrollables)
                    if chosenPoint1 in invoked_points:
                        chosenPoint1 = random.choice(setOfUncontrollables)
                    chosenPoint2 = random.choice(setOfControllables)
                    if chosenPoint2 in invoked_points:
                        chosenPoint2 = random.choice(setOfControllables)
                    bound = randomBound(max_scale)
                    disjunct.append([chosenPoint1, chosenPoint2, bound[0], bound[1]])
                    invoked_points.append(chosenPoint1)
                    invoked_points.append(chosenPoint2)

                elif typeOfConjunct >= 0.5 and len(setOfControllables) > 1:
                    #controllable controllable
                    controllable_temp = copy.deepcopy(setOfControllables)
                    if j == 0:
                        chosenPoint1 = chosenPoint
                    else:
                        chosenPoint1 = random.choice(controllable_temp)
                    if chosenPoint1 in invoked_points:
                        chosenPoint1 = random.choice(controllable_temp)
                    if chosenPoint1 in controllable_temp:
                        controllable_temp.remove(chosenPoint1)
                    chosenPoint2 = random.choice(controllable_temp)
                    if chosenPoint2 in invoked_points:
                        chosenPoint2 = random.choice(controllable_temp)
                    controllable_temp.remove(chosenPoint2)
                    bound = randomBound(max_scale)
                    disjunct.append([chosenPoint1, chosenPoint2, bound[0], bound[1]])
                    invoked_points.append(chosenPoint1)
                    invoked_points.append(chosenPoint2)

            freeConstraints.append(disjunct)


    return createDNTU(setOfControllables, setOfUncontrollables, freeConstraints, contingencyLinks)

def trainModel(model, num_controllables, num_uncontrollables,
               num_of_examples, time_out, explore_prob, window):

    for i in range(num_of_examples):

        a = genRandomDtnu(
            random.randint(num_uncontrollables, num_controllables),
            random.randint(1, num_uncontrollables), 1, random.randint(1,5))

        visited_nodes_num = [0]
        visited_or_nodes_num = [0]
        visited_and_nodes_num = [0]
        print("\n Training on example %d" % i)
        start = time.time()
        last = 0
        if len(window) > 0:
            last = window[-1]

        try:
            treeSearch(copy.deepcopy(a), window, visited_nodes_num, visited_or_nodes_num, visited_and_nodes_num, explore_prob, True, False, time.time(), time_out)
        except timeout_decorator.timeout_decorator.TimeoutError:
            print("Timer expired")
        end = time.time()
        print("length")


        print(len(window))
        if len(window) > 0:
            if True or window[-1] != last:
                #soft_random_train(model, window, 1, 64, optimizer, device)
                print("30r")
            else:
                print("no data added")
        print(len(window))

def diff(li1, li2):
    return (list(set(li1) - set(li2)))

def printStrategy(treeNode, depth):
    tab = "\t"
    if treeNode.nodeType == 0:
        if treeNode.parentNode != None:
            if treeNode.parentNode.nodeType == 1:
                pkey = treeNode.parentNode.parentNode.relativeOccurenceTimes.keys()
                ckey = treeNode.relativeOccurenceTimes.keys()
                occurredPoints = diff(ckey, pkey)
                print(tab * depth + "Execute %s at current time t = %.2f, "%(occurredPoints, treeNode.globalTime))


            if treeNode.parentNode.nodeType == 2:
                gkey = treeNode.parentNode.parentNode.parentNode.parentNode.parentNode.globalOccurenceTimes.keys()
                pkey = treeNode.parentNode.parentNode.parentNode.parentNode.parentNode.activatedUncontrollables.keys()
                ckey = treeNode.activatedUncontrollables.keys()
                occurredPoints = diff(pkey, ckey)
                print(tab * (depth - 1) + "If these points occurred: %s "%occurredPoints)

        if treeNode.childOr != None:
            printStrategy(treeNode.childOr, depth)
        else:
            hkey = treeNode.globalOccurenceTimes.keys()
            npo = diff(diff(hkey, gkey), occurredPoints)
            for el in npo:
                if treeNode.relativeOccurenceTimes[el][0] >= 0:
                    print(tab * depth + "Schedule %d at given time: %s"%(el, treeNode.globalOccurenceTimes[el]))
            print(tab * depth + "Problem solved")



    elif treeNode.nodeType == 1:
        printStrategy(treeNode.validatedChild, depth)

    elif treeNode.nodeType == 2:
        print("reactive strategy: %s,"%treeNode.instantReacts)
        for child in treeNode.listOfChildren:
            printStrategy(child, depth + 1)


    elif treeNode.nodeType == 3:
        print(tab * depth + "Wait %s units at current time t = %.2f with " %(treeNode.waitDuration, treeNode.globalTime), end = '')
        printStrategy(treeNode.childOr, depth)

    return
