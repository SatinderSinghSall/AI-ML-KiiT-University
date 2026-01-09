# Find compound interest using Python.

principal = float(input("Enter the principal amount: "))
rate = float(input("Enter the rate amount: "))
time = float(input("Enter the time amount: "))

amount = principal*(pow((1 + rate / 100), time))
CI = amount - principal

print("Compound Interest: ", CI)
