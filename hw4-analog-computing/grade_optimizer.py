# practice using sympy
import pulp
from enum import Enum
import random


# get practice writing a constraint problem
def grade_optimizer(assignInfo):

    # create an integer linear program that maximizes some expression (our grade), subject 
    # to a bunch of constraints
    gradeMax = pulp.LpProblem("MaximizeGrade", pulp.LpMaximize)
    
    # constants you may find useful
    latePenalty = 0.08
    availGraceDays = 3

    grades = {}
    # an expression that will eventually count the total number of grace days
    totalGraceDays = 0
    # an expression that contains the total grade 
    totalGrade = 0

    for assign, stats in assignInfo.items():
        # we will create a new integer variable named grace{assignmentname} for each assignment
        # this variable will be set to the number of grac days used for that assignment.
        asgnGraceDays = pulp.LpVariable(f"grace{assign}",lowBound=0, cat="Integer")
        assignmentGrade = (stats["grade"] - latePenalty*(stats["daysLate"] - asgnGraceDays))

        gradeMax.addConstraint(asgnGraceDays <= stats["daysLate"])
        gradeMax.addConstraint(assignmentGrade >= 0)

        totalGrade += assignmentGrade*stats["totalGrade"]
        totalGraceDays +=asgnGraceDays 

    gradeMax.addConstraint(totalGraceDays <= availGraceDays)
    gradeMax.setObjective(totalGrade)
    gradeMax.solve()

    #reports the optimizer status
    print("==== Status ====")
    print(gradeMax.status)

    # prints out the solution
    print("=== Solution ===")
    varsdict = {}
    for v in gradeMax.variables():
        varsdict[v.name] = v.varValue
        print("%s = %d grace days" % (v,varsdict[v.name]))
    
    print("=== Maximum Grade ===")
    print("Maximum Grade: %s" % gradeMax.objective.value())


# totalGrade is the fraction of the total grade allocated to each assignment/project
# daysLate is the number of days late the assignment was submitted
# grade is the grade that was scored on the project/assignment.
assignInfo = {
    "hw1": {"grade":1.0, "daysLate":1, "totalGrade":0.1},
    "hw2": {"grade":1.0, "daysLate":1, "totalGrade":0.1},
    "hw3": {"grade":1.0, "daysLate":1, "totalGrade":0.1},
    "hw4": {"grade":0.95, "daysLate":3, "totalGrade":0.1},
    "bib": {"grade":1.0, "daysLate":1, "totalGrade":0.05},
    "survey": {"grade":0.80, "daysLate":2, "totalGrade":0.25},
    "propdraft": {"grade":1.0, "daysLate":0, "totalGrade":0.18},
    "final": {"grade":1.0, "daysLate":1, "totalGrade":0.12},
}
grade_optimizer(assignInfo)
