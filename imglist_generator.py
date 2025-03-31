import random

# Generate 650 random numbers between 0 and 1195
random_numbers = random.sample(range(0, 1196), 350)

# Format numbers to ensure they have 4 digits
formatted_numbers = [f"{num:04d}" for num in random_numbers]

# Sort the numbers in ascending order
formatted_numbers.sort()

# Write the sorted numbers to a text file
with open("LINEMOD/data/05/med_train.txt", "w") as file:
    for number in formatted_numbers:
        file.write(number + "\n")

print("File 'med_train.txt' has been created.")
