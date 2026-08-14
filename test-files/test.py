# weight = int(input('weight: '))
# unit = input('(K)g or (L)bs: ')
# if unit.upper() == "K": 
#     converted_weight = weight * 2.205
#     print(f"You are {converted_weight} pounds")
# else:
#     converted_weight = weight / 2.205
#     print(f"You are {converted_weight} kilos")


# def lbs_to_kg(weight):
#     return weight / 2.205

# def kg_to_lbs(weight):
#     return weight * 2.205
# Guessed_number = int(input("Guess: "))
# number=9
# i = 1 
# while i<= 3:
#     if Guessed_number != number:

#      i+= 1 
#      print("wrong number try again")
#     elif Guessed_number == number:
#     print("You guessed the right number")
# prices = [10, 20, 30]
# total = 0
# for item in prices:
#  total += item
#  print(item)
# print(f"TOtal: {total}")
# print("""+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++""")
# is_hot = False
# if is_hot:
#     print("It's a hot day")
# else:
#     print("It's a cold day")



                    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    # print("Welcome to AI Implementation and Model Development Training")
                    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

                    # company = "Abay Bank"
                    # sales = 50000
                    # print(company)
                    # print(sales)
                    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

                    # name = "Sara"        # text (string)
                    # age = 25             # number (integer)
                    # profit = 1200.50     # decimal (float)
                    # is_active = True     # true/false (boolean)

                    # print(name)
                    # print(age)
                    # print(profit)
                    # print(is_active)

                    # print(type(name))
                    # print(type(age))
                    # print(type(profit))
                    # print(type(is_active))

                    # new_sales = sales + 10000
                    # print("New Sales:", new_sales)
                    # print("Company:", company)
                    # print("Total Sales:", new_sales)
# import math

# math.sqrt(16)
# x=-2.9
# print(math.ceil(x))
# user_input = input('Enter your name: ')
# print("Hello,", user_input, "! Welcome to AI Implementation and Model Development Training.")
# print(user_input.find("a"))
# if user_input.find("a") != -1:
    # print("The letter 'a' is present in your name. at" ,user_input.find("a") +1) 
# user_input1 = input('Enter your favorite color: ')
# print("Hello,", user_input[1:2], "! Welcome to AI Implementation and Model Development Training. and favorite color is ", user_input1)
# bame ="hellowlowl"
#   #    0123456789  
# print(bame[5:-4]
# x = (2+3)**10
# print(x)


# matrix = [
#     [1, 2, 3], 
#     [4, 5, 6], 
#     [7, 8, 9]
#  ]
# for row in matrix:
#     for item in row:
#         print(item)


import numpy as np
scores = np.array([70, 80, 90, 85, 75])

print("Average:", np.mean(scores))
print("Highest:", np.max(scores))


print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
sales = np.array([200, 250, 300, 400])

growth = sales * 1.1

print(growth)


yield_data = np.array([2.5, 3.0, 3.2, 2.8])

print("Average Yield:", np.mean(yield_data))


image = np.array([
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255]
])