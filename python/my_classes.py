#The class used for every node of the tree, with common attributes
#parentNode: parent node of current tree node
#truthValue: 0 if current node is not DC, 1 otherwise. 2 if unknown
#nodeType: 0 if DTNU, 1 if OR node, 2 if AND node, 3 if WAIT node
#globalTime: Total time waited since t=0


import copy


class TreeNode():
    def __init__(self, parentNode, truthValue, nodeType, globalTime):
        self.parentNode = parentNode
        self.truthValue = truthValue
        self.nodeType = nodeType
        self.globalTime = globalTime
        return

#The DTNU class is a specialization of the TreeNode class
#remainingFreeConstraints: remaining free constaints left to satisfy.
# All removed constraints have already been satisfied completely
#remainingContingencyLinks: remaining contingency links left to be activated.
#activatedUncontrollables: current list of uncontrollables which are activated
#and set to take place in a defined window
#scheduledTasks: Tasks already scheduled, and their execution time
#setOfControllables: The static set of controllables
#setOfUncontrollables: The static set of uncontrollables
#childOr: The OR child node of the DTNU
class Dtnu(TreeNode):
    def __init__(self, remainingFreeConstraints, globalFreeConstraints,remainingContingencyLinks,
                 activatedUncontrollables, relativeOccurenceTimes, globalOccurenceTimes,
                 remainingControllables, setOfControllables,
                 remainingUncontrollables, setOfUncontrollables, lastControllableScheduled,
                 occuredUncontrollablesDuringLastWait, lastWaitTime, instantReacts, childOr, tasksElapsedToCheck,
                 controllableSetTimeAfterUncontrollable, scheduledCombinations, scheduledPointsCombination, *args, **kwargs):
        self.remainingFreeConstraints = remainingFreeConstraints
        self.globalFreeConstraints = globalFreeConstraints
        self.remainingContingencyLinks = remainingContingencyLinks
        self.activatedUncontrollables = activatedUncontrollables
        self.relativeOccurenceTimes = relativeOccurenceTimes
        self.globalOccurenceTimes = globalOccurenceTimes
        self.remainingControllables = remainingControllables
        self.setOfControllables = setOfControllables
        self.remainingUncontrollables = remainingUncontrollables
        self.setOfUncontrollables = setOfUncontrollables
        self.lastControllableScheduled = lastControllableScheduled
        self.occuredUncontrollablesDuringLastWait = occuredUncontrollablesDuringLastWait
        self.lastWaitTime = lastWaitTime
        self.instantReacts = instantReacts
        self.childOr = childOr
        self.tasksElapsedToCheck = tasksElapsedToCheck
        self.controllableSetTimeAfterUncontrollable = controllableSetTimeAfterUncontrollable
        self.scheduledCombinations = scheduledCombinations
        self.scheduledPointsCombination = scheduledPointsCombination
        super(Dtnu, self).__init__(*args, **kwargs)

#The OrNode class is a specialization of the TreeNode class.
#validatedChild: The first child which has been found to have 1 as truth value
#numberOfFalseChildren: Counter which keeps track of the number of false children
#numberOfChildren: Number of children
#listOfChildren: List of children
class OrNode(TreeNode):
    def __init__(self, instantReacts, validatedChild, numberOfFalseChildren, listOfFalseChildren, numberOfChildren,
                 listOfChildren, *args, **kwargs):
        self.instantReacts = instantReacts
        self.validatedChild = validatedChild
        self.numberOfFalseChildren = numberOfFalseChildren
        self.listOfFalseChildren = listOfFalseChildren
        self.numberOfChildren = numberOfChildren
        self.listOfChildren = listOfChildren
        super(OrNode, self).__init__(*args, **kwargs)

#The AndNode class is a specialization of the TreeNode class
#numberOfTrueChildren: Counter which keeps track of the number of true children
#numberOfChildren: Number of children
#listOfChildren: List of children
class AndNode(TreeNode):
    def __init__(self, instantReacts, numberOfTrueChildren, listOfTrueChildren, numberOfChildren,
                 listOfChildren, *args, **kwargs):
        self.instantReacts = instantReacts
        self.numberOfTrueChildren = numberOfTrueChildren
        self.listOfTrueChildren = listOfTrueChildren
        self.numberOfChildren = numberOfChildren
        self.listOfChildren = listOfChildren
        super(AndNode, self).__init__(*args, **kwargs)

#The waitNode class is a specialization of the TreeNode class
#waitDuration: the amount of time waited in this node
#childAnd: the AND child of the wait node
class WaitNode(TreeNode):
    def __init__(self, waitDuration, childOr, *args, **kwargs):
        self.waitDuration = waitDuration
        self.childOr = childOr
        super(WaitNode, self).__init__(*args, **kwargs)

#test = Dtnu(1,2,3,4,5,6,7,8,9,10,11)




