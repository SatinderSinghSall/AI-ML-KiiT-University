from datetime import datetime

current_hour = datetime.now().hour

if 5 <= current_hour < 12:
    greeting = "Good morning"
elif 12 <= current_hour < 18:
    greeting = "Good day"
else:
    greeting = "Good evening"

print(greeting)
