using RecursiveArrayTools, Tracker, Test

x = [5.0]
a = [Tracker.TrackedArray(x)]
b = [Tracker.TrackedArray(copy([5.2]))]
RecursiveArrayTools.recursivecopy!(a, b)
@test a[1][1] == 5.2
