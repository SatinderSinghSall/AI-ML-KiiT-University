# Python program to find Largest of three numbers.

number1 = float(input("Enter the 1st number: "))
number2 = float(input("Enter the 2nd number: "))
number3 = float(input("Enter the 3rd number: "))

if number1 > number2 and number1 > number3:
    print(f"{number1} > {number2} and {number3}")
elif number2 > number1 and number2 > number3:
    print(f"{number2} > {number1} and {number3}")
else:
    print(f"{number3} > {number1} and {number2}")
