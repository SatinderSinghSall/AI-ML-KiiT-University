# Find simple interest using Python.

principleAmount = float(input("Enter the principal amount: "))
rate = float(input("Enter the rate: "))
time = float(input("Enter the time: "))

SimpleInterest = principleAmount * rate / time

print("The simple interest is", SimpleInterest)
