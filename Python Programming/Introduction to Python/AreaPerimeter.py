# Find the Area and Perimeter of a Rectangle using Python.

length = float(input("Enter the length of the rectangle: "))
width = float(input("Enter the width of the rectangle: "))

area = length * width
perimeter = 2 * (length + width)

print(f"The perimeter of the rectangle is {perimeter}")
print(f"The area of the rectangle is {area}")
