# Grading System:

marks = float(input("Enter the marks: "))

if marks >= 90:
    grade = "O"
elif marks >= 80:
    grade = "E"
elif marks >= 70:
    grade = "A"
elif marks >= 60:
    grade = "B"
elif marks >= 50:
    grade = "C"
elif marks >= 40:
    grade = "D"
elif marks < 40:
    grade = "F"

print(f"Your grade is {grade} !")
